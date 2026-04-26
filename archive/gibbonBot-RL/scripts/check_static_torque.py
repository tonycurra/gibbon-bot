"""
Static torque feasibility check.

Checks whether the real hardware can physically hold a standing pose.
Uses qfrc_bias (gravity compensation torques) with a pose where the
arm grippers are placed AT the floor — equivalent to the static case
where each arm pushes down and supports the torso.

The key insight: qfrc_bias[i] at zero velocity = joint torque required
to hold the current pose against gravity. We test the pose with grippers
near the floor so the arm geometry is realistic for a push-up configuration.
For the free joint (torso), qfrc_bias[2] = total robot weight — this
confirms the mass is correct.

Usage:
    .venv/bin/python scripts/check_static_torque.py
    .venv/bin/python scripts/check_static_torque.py --target-z 0.15
    .venv/bin/python scripts/check_static_torque.py --no-dual-drive
"""

import argparse
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
SCENE = ROOT / "data" / "scene.xml"


def get_actuator_limits(model):
    limits = np.zeros(model.nu)
    for i in range(model.nu):
        if model.actuator_forcelimited[i]:
            limits[i] = model.actuator_forcerange[i, 1]
        else:
            limits[i] = np.inf
    return limits


def actuator_to_joint_names(model):
    names = []
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-z", type=float, default=0.20,
                    help="Torso height to test (m)")
    ap.add_argument("--torso-mass", type=float, default=None,
                    help="Override torso mass (kg). Default: use scene.xml value")
    ap.add_argument("--arm-mass-scale", type=float, default=1.0,
                    help="Multiplier for all arm-body masses (default: scene.xml values)")
    ap.add_argument("--no-dual-drive", action="store_true",
                    help="Disable dual-drive shoulder correction (use single-servo limit)")
    ap.add_argument("--shoulder", type=float, default=1.0,
                    help="Shoulder joint angle for test pose (rad)")
    ap.add_argument("--elbow", type=float, default=1.5,
                    help="Elbow joint angle for test pose (rad)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    tree_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tree")

    if args.torso_mass is not None:
        ratio = args.torso_mass / model.body_mass[torso_id]
        model.body_mass[torso_id] = args.torso_mass
        model.body_inertia[torso_id] *= ratio

    if args.arm_mass_scale != 1.0:
        for b in range(model.nbody):
            if b in (0, torso_id, tree_id):
                continue
            model.body_mass[b] *= args.arm_mass_scale
            model.body_inertia[b] *= args.arm_mass_scale

    dual_drive = not args.no_dual_drive
    limits = get_actuator_limits(model)
    if dual_drive:
        for i, n in enumerate(actuator_to_joint_names(model)):
            if "link1_to_link2" in n:
                limits[i] = 5.886  # 2× ST3235 dual-drive
        print("Shoulder: dual-drive (2× ST3235) → 5.886 Nm")
    else:
        print("Shoulder: single-servo (ST3235) → 2.943 Nm")

    arm_total = sum(model.body_mass[b] for b in range(model.nbody)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                    .startswith(("fl_", "fr_", "bl_", "br_")))
    total_mass = model.body_mass[torso_id] + arm_total

    print(f"Scene:       {SCENE}")
    print(f"Torso mass:  {model.body_mass[torso_id]:.3f} kg")
    print(f"Arms total:  {arm_total:.3f} kg  ({arm_total/4:.3f} kg/arm)")
    print(f"Robot total: {total_mass:.3f} kg  ({total_mass*9.81:.1f} N)")
    print()

    # Build test pose: torso at target_z, arms in push-down configuration
    qpos = np.zeros(model.nq)
    qpos[0:3] = [0.0, 0.0, args.target_z]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    arm_pose = np.array([0.0, args.shoulder, args.elbow, 0.0, 0.0, 0.0])
    for i in range(4):
        qpos[7 + i*6 : 7 + (i+1)*6] = arm_pose

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # qfrc_bias at v=0 = pure gravity compensation torques (no Coriolis)
    # Free joint dofs 0-5: dof 2 (z) = total robot weight → sanity check
    gravity_torques = data.qfrc_bias.copy()
    print(f"Sanity: free-joint z-force = {gravity_torques[2]:.2f} N  "
          f"(expected {total_mass*9.81:.1f} N, ignoring contact)")
    print()

    # Joint torques are dofs 6..6+nu
    joint_torques = gravity_torques[6:6+model.nu]
    names = actuator_to_joint_names(model)

    print(f"NOTE: These are torques to hold ARM WEIGHT at this pose.")
    print(f"Torso weight ({model.body_mass[torso_id]*9.81:.1f} N) acts on free joint, not arm joints.")
    print(f"Torso-support torque per shoulder ≈ "
          f"{model.body_mass[torso_id]*9.81/4 * 0.15:.2f} Nm  "
          f"(torso_weight/4 × 0.15m moment arm estimate)")
    print()

    print(f"{'JOINT':40s} {'HOLD_ARM (Nm)':>14s} {'LIMIT (Nm)':>10s} {'UTIL':>8s} {'OK?':>6s}")
    print("-" * 84)
    worst = 0.0
    n_over = 0
    for name, torque, lim in zip(names, joint_torques, limits):
        util = abs(torque) / lim if lim > 0 else np.inf
        worst = max(worst, util)
        ok = "OK" if util <= 1.0 else "FAIL"
        if util > 1.0:
            n_over += 1
        if abs(torque) > 0.01 or util > 0.1:  # skip near-zero rows
            print(f"{name:40s} {torque:>14.3f} {lim:>10.2f} {util*100:>7.1f}% {ok:>6s}")

    print("-" * 84)
    print(f"Worst arm-weight utilization: {worst*100:.1f}%")
    print()

    # Hand-calc for torso support
    torso_per_arm = model.body_mass[torso_id] * 9.81 / 4
    moment_arm = 0.15  # m, conservative shoulder-to-torso-COM horizontal distance
    torso_torque = torso_per_arm * moment_arm
    shoulder_limit = limits[names.index("fl_link1_to_link2")] if "fl_link1_to_link2" in names else 5.886
    shoulder_arm_torque = abs(joint_torques[names.index("fl_link1_to_link2")]) if "fl_link1_to_link2" in names else 0
    total_shoulder = shoulder_arm_torque + torso_torque
    combined_util = total_shoulder / shoulder_limit

    print(f"Combined shoulder load estimate:")
    print(f"  Arm weight torque:   {shoulder_arm_torque:.3f} Nm")
    print(f"  Torso support torque: {torso_torque:.3f} Nm  (torso/4 × {moment_arm}m)")
    print(f"  Total shoulder need: {total_shoulder:.3f} Nm  vs limit {shoulder_limit:.3f} Nm")
    print(f"  → Shoulder utilization: {combined_util*100:.1f}%")
    print()

    if combined_util <= 1.0 and worst <= 1.0:
        print(f"FEASIBLE: robot can statically hold torso at z={args.target_z:.2f}m.")
        print("RL has a chance — if training fails, the problem is reward shaping or exploration.")
    elif combined_util <= 1.0:
        print(f"MARGINAL: shoulder OK, but other joints show high utilization.")
        print("Check elbow angle — a more vertical arm reduces elbow torque.")
    else:
        print(f"INFEASIBLE: shoulder cannot support robot at z={args.target_z:.2f}m.")
        print("Options: increase actuator forcerange, reduce mass, or lower target_z.")


if __name__ == "__main__":
    main()
