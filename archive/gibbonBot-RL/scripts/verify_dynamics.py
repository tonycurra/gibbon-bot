#!/usr/bin/env python3
"""Audit MuJoCo scene dynamics vs. expected real-world values.

Cross-references data/motors.md, real robot materials, and physics constants.
Run after any scene.xml or motor change to confirm nothing has drifted.
"""
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "scene.xml"

# Expected values from data/motors.md — fail loud if scene disagrees.
EXPECTED_MOTOR_SPEC = {
    # joint suffix → (motor_kind, no_load_radps, stall_torque_Nm, damping)
    "shoulder_yaw":   ("ST3235",    4.71, 2.94, 0.62),
    "shoulder_pitch": ("ST3235",    4.71, 2.94, 0.62),
    "elbow":          ("ST3235",    4.71, 2.94, 0.62),
    "knee":           ("ST3235",    4.71, 2.94, 0.62),
    "hip_yaw":        ("ST3235",    4.71, 2.94, 0.62),
    "hip_pitch":      ("ST3235",    4.71, 2.94, 0.62),
    "forearm_roll":   ("ST3235",    4.71, 2.94, 0.62),
    "shin_roll":      ("ST3235",    4.71, 2.94, 0.62),
    "wrist":          ("ST3215",    4.71, 2.94, 0.62),
    "ankle":          ("ST3215",    4.71, 2.94, 0.62),
    "gripper":        ("ST3215_HS", 11.10, 1.96, 0.18),
    "foot":           ("ST3215_HS", 11.10, 1.96, 0.18),
}

EXPECTED_PHYSICS = {
    "timestep_s": 0.002,
    "gravity_z": -9.81,
}


def joint_suffix(name: str) -> str:
    """Strip limb prefix (fl_, fr_, bl_, br_) from joint name."""
    parts = name.split("_", 1)
    return parts[1] if len(parts) > 1 else name


def main():
    print(f"Loading {SCENE}\n")
    model = mujoco.MjModel.from_xml_path(str(SCENE))

    fail = 0
    warn = 0

    # ── Physics ────────────────────────────────────────────────────────────
    print("=== PHYSICS ===")
    dt = float(model.opt.timestep)
    g = model.opt.gravity
    print(f"timestep:   {dt:.4f} s   (expected {EXPECTED_PHYSICS['timestep_s']})")
    print(f"gravity:    ({g[0]:+.2f}, {g[1]:+.2f}, {g[2]:+.2f})   (expected z={EXPECTED_PHYSICS['gravity_z']})")
    print(f"integrator: {['Euler', 'RK4', 'implicit', 'implicitfast'][model.opt.integrator]}")
    print(f"iterations: {model.opt.iterations}")
    print(f"solver:     {['PGS', 'CG', 'Newton'][model.opt.solver]}")
    if abs(dt - EXPECTED_PHYSICS["timestep_s"]) > 1e-6:
        print(f"  ⚠ timestep differs from expected"); warn += 1
    if abs(g[2] - EXPECTED_PHYSICS["gravity_z"]) > 1e-3:
        print(f"  ⚠ gravity z differs"); warn += 1
    print()

    # ── Robot torso mass & inertia ─────────────────────────────────────────
    print("=== ROBOT MASSES ===")
    total_mass = 0.0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body{i}>"
        if name in ("world",): continue
        mass = model.body_mass[i]
        total_mass += mass
        if mass > 0.001:
            print(f"  {name:<30} {mass*1000:>7.1f} g")
    print(f"  TOTAL: {total_mass:.3f} kg = {total_mass*9.81:.2f} N weight")
    if total_mass < 0.5 or total_mass > 10.0:
        print(f"  ⚠ total mass outside expected 0.5-10 kg range"); warn += 1
    print()

    # ── Joints ─────────────────────────────────────────────────────────────
    print("=== JOINTS (24 actuated + torso freejoint) ===")
    print(f"  {'name':<24} {'type':<12} {'damping':<10} {'range (rad)':<24}")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<j{i}>"
        jtype = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        damp = model.dof_damping[model.jnt_dofadr[i]]
        if model.jnt_limited[i]:
            r = model.jnt_range[i]
            r_str = f"[{r[0]:+.2f}, {r[1]:+.2f}]"
        else:
            r_str = "(unlimited)"
        print(f"  {name:<24} {jtype:<12} {damp:<10.3f} {r_str}")

        suffix = joint_suffix(name)
        if suffix in EXPECTED_MOTOR_SPEC:
            _, _, _, exp_damp = EXPECTED_MOTOR_SPEC[suffix]
            if abs(damp - exp_damp) > 0.01:
                print(f"      ⚠ damping {damp} != expected {exp_damp} (motors.md)"); fail += 1
    print()

    # ── Actuators ──────────────────────────────────────────────────────────
    print("=== ACTUATORS (24) ===")
    print(f"  {'name':<28} {'kp':<6} {'forcerange':<22} {'ctrlrange':<22}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"<u{i}>"
        kp = model.actuator_gainprm[i, 0]
        if model.actuator_forcelimited[i]:
            fr = model.actuator_forcerange[i]
            fr_str = f"[{fr[0]:+.2f}, {fr[1]:+.2f}]"
        else:
            fr_str = "(unlimited)"
        cr = model.actuator_ctrlrange[i]
        cr_str = f"[{cr[0]:+.2f}, {cr[1]:+.2f}]"
        print(f"  {name:<28} {kp:<6.1f} {fr_str:<22} {cr_str:<22}")

        # cross-check forcerange vs motor stall torque
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        suffix = joint_suffix(joint_name)
        if suffix in EXPECTED_MOTOR_SPEC:
            _, _, exp_torque, _ = EXPECTED_MOTOR_SPEC[suffix]
            if model.actuator_forcelimited[i]:
                if abs(fr[1] - exp_torque) > 0.01:
                    print(f"      ⚠ stall torque {fr[1]} != expected {exp_torque} N·m (motors.md)"); fail += 1
    print()

    # ── Floor contact dynamics ─────────────────────────────────────────────
    print("=== FLOOR / CONTACT ===")
    floor_id = None
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name == "floor":
            floor_id = i
            break
    if floor_id is not None:
        print(f"  geom 'floor':")
        print(f"    type:      {['plane','heightfield','sphere','capsule','ellipsoid','cylinder','box','mesh','sdf'][model.geom_type[floor_id]]}")
        print(f"    friction:  ({model.geom_friction[floor_id][0]:.3f}, {model.geom_friction[floor_id][1]:.4f}, {model.geom_friction[floor_id][2]:.4f})")
        print(f"               (sliding, torsional, rolling — agricultural soft ~ 1.5/0.07/0.05)")
        print(f"    solref:    {tuple(model.geom_solref[floor_id])}  (time-const, damping ratio; soft ~ 0.035 1.0)")
        print(f"    solimp:    {tuple(model.geom_solimp[floor_id])}  (d_min, d_max, width; soft ~ 0.85 0.92 0.02)")
        if model.geom_solref[floor_id][0] < 0.025:
            print(f"      ⚠ solref looks rigid (< 0.025) — set 0.035+ for soft ground"); warn += 1
    else:
        print("  ⚠ no geom named 'floor' found"); fail += 1
    print()

    # ── Robot self-collision ───────────────────────────────────────────────
    n_robot_geoms = sum(1 for i in range(model.ngeom)
                        if model.geom_contype[i] != 0 and "tree" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""))
    print(f"=== COLLISION ===")
    print(f"  total geoms:       {model.ngeom}")
    print(f"  collidable (contype≠0): {sum(1 for i in range(model.ngeom) if model.geom_contype[i] != 0)}")
    print(f"  mesh geoms (visual only, contype=0): {sum(1 for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[i] == 0)}")
    print()

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 60)
    if fail == 0 and warn == 0:
        print(f"  ✅ all dynamics match expected values from motors.md")
    elif fail == 0:
        print(f"  ⚠ {warn} warning(s) — see above")
    else:
        print(f"  ❌ {fail} failure(s), {warn} warning(s) — see above")
    sys.exit(fail)


if __name__ == "__main__":
    main()
