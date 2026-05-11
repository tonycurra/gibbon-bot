#!/usr/bin/env python3
"""
Real-time MuJoCo viewer — no training, no checkpoints.
Just the robot in the physics sim so you can see if things look right.

Policy modes (set POLICY at the top):
  "zero"    — all joints at rest, see how the robot falls/settles
  "random"  — random actions each step
  "latest"  — load the most recent training checkpoint if one exists

Usage:
    python view.py
"""

import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
POLICY = "latest"  # "zero" | "random" | "latest" | "demo"
SCENE = pathlib.Path("data/scene.xml")
PHYSICS_STEPS = (
    16  # physics sub-steps per action (matches PHYSICS_STEPS_PER_ACTION in params)
)
# MOTOR speed/accel and ACTION_ALPHA are read per-episode from envs/params_walk_mjx.py.
# Edit the file → next episode reset (~few seconds) picks up the new values.
# No need to restart the viewer or the trainer for motor-feel tweaks.
# ─────────────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--since",
    type=float,
    default=0.0,
    help="ignore checkpoints saved before this unix timestamp",
)
_parser.add_argument(
    "--env",
    default="walk",
    help="which task was trained (walk, stand, walk_forward, turn_left, turn_right). "
    "Selects runs/ape_<env>_mjx/ as the checkpoint dir.",
)
_parser.add_argument(
    "--best",
    action="store_true",
    help="load the best-mean_rew checkpoint (per mlflow) instead of newest, no hot-swap",
)
_args = _parser.parse_args()
SINCE = _args.since
ENV = _args.env
BEST = _args.best
RUNS = pathlib.Path(f"runs/ape_{ENV}_mjx")
# All current tasks share the 88-dim obs contract (envs/_shared_obs.py).
OBS_DIM = 88
GAIT_PERIOD_STEPS = 25


def _is_compatible(path):
    """Check if .npz first param shape matches OBS_DIM."""
    try:
        d = np.load(str(path))
        # Find the (?, 256) shape with first dim == OBS_DIM (Dense_0/kernel)
        return any(d[k].shape == (OBS_DIM, 256) for k in d.files)
    except Exception:
        return False


def latest_checkpoint():
    if BEST:
        try:
            from utils.best import find_best

            results = find_best(task=ENV, top=1)
            if results:
                path, rew, step = results[0]
                if _is_compatible(path):
                    print(
                        f"[view] best: {path.name}"
                        + (f" mean_rew={rew:.3f}" if rew else "")
                    )
                    return path
        except Exception as e:
            print(f"[view] --best failed ({e}), falling back to latest")
    # Sort by MTIME (write time), not filename. Otherwise old runs with higher
    # step numbers shadow new fresh runs starting from step 0.
    ckpts = [
        c
        for c in sorted(RUNS.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
        if c.stat().st_mtime >= SINCE
    ]
    # Walk backwards (newest first); return newest *compatible* checkpoint
    for c in reversed(ckpts):
        if _is_compatible(c):
            return c
    if ckpts:
        print(
            f"[view] WARNING: {len(ckpts)} checkpoints found but none match current OBS_DIM={OBS_DIM}"
        )
    return None


def _build_net():
    """Construct the actor-critic network. Returns (net, OBS_DIM, ACT_DIM, jnp)."""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    ACT_DIM = 24

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(128)(x))
            mean = nn.Dense(ACT_DIM)(x)
            log_std = self.param("log_std", nn.initializers.zeros, (ACT_DIM,))
            value = nn.Dense(1)(x).squeeze(-1)
            return mean, log_std, value

    return Net(), OBS_DIM, ACT_DIM, jax, jnp


def load_predict(path, cached=[None]):
    """Load (or hot-swap) policy from .npz checkpoint. Reuses the JIT'd apply_fn."""
    if cached[0] is None:
        net, OBS_DIM, ACT_DIM, jax, jnp = _build_net()
        init_params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
        cached[0] = (net, init_params, jax, jnp)
    net, init_params, jax, jnp = cached[0]

    data = np.load(str(path))
    # data.files is sorted ALPHABETICALLY (arr_0, arr_1, arr_10, arr_2, ...) — broken
    # for nets with >10 params. Sort numerically by suffix.
    keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
    flat = [jnp.asarray(data[k]) for k in keys]
    init_flat, treedef = jax.tree_util.tree_flatten(init_params)
    # Shape-check: skip incompatible checkpoints (e.g. older OBS_DIM).
    if any(a.shape != b.shape for a, b in zip(flat, init_flat)):
        raise ValueError(
            f"shape mismatch in {path.name} (probably trained on different OBS_DIM)"
        )
    params = jax.tree_util.tree_unflatten(treedef, flat)

    apply_jit = jax.jit(lambda obs, p=params: net.apply(p, obs)[0])

    def predict(obs):
        return np.array(jnp.tanh(apply_jit(jnp.asarray(obs, dtype=jnp.float32))))

    return predict


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # Mesh-mesh self-collision is the O(n²) cost that makes the CPU viewer crawl.
    # Move mesh geoms to group 2 so they only collide with the floor (group 1, conaffinity=31).
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_contype[i] = 2
            model.geom_conaffinity[i] = 1

    data = mujoco.MjData(model)
    ACT_DIM = model.nu
    DT = float(model.opt.timestep)
    # Skip joint 0 (the torso freejoint, no range) and stop after the 24
    # actuated robot joints — branch ball joints don't enter the action
    # space.
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, 1 + ACT_DIM)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, 1 + ACT_DIM)])

    # Per-actuator motor speed / accel limits — re-read each episode from
    # envs/params_walk_mjx.py so motor-feel tweaks land within seconds.
    PER_LIMB_MAX_RAD_PER_SEC = [4.71, 4.71, 4.71, 4.71, 4.71, 11.10]
    motor_max_radps = np.array(PER_LIMB_MAX_RAD_PER_SEC * 4, dtype=np.float32)
    action_dt = DT * PHYSICS_STEPS
    # Standing rest pose (action=0 → these joint angles, in radians).
    # Per limb: yaw, pitch, elbow/knee, roll, wrist/ankle, gripper/foot.
    DEG = np.pi / 180.0
    rest_pose = np.tile(np.array([0.0, 0.0, 30.0 * DEG, 0.0, 0.0, 0.0]), 4).astype(
        np.float32
    )
    half_range = np.maximum(np.abs(jnt_hi - rest_pose), np.abs(rest_pose - jnt_lo))

    _params_mod = [None]  # cached module + last mtime
    _params_mtime = [0.0]
    _params_path = pathlib.Path(__file__).parent / "envs" / "params_walk_mjx.py"

    def reload_motor_params():
        """Returns (motor_max_speed_arr, motor_max_accel_arr, action_alpha).
        Re-imports envs.params_walk_mjx if its file has changed on disk."""
        try:
            import envs.params_walk_mjx as _p

            mtime = _params_path.stat().st_mtime
            if _params_mod[0] is None or mtime != _params_mtime[0]:
                import importlib

                importlib.reload(_p)
                _params_mod[0] = _p
                _params_mtime[0] = mtime
                speed_frac = float(getattr(_p, "MOTOR_SPEED_FRAC", 1.0))
                accel_ratio = float(getattr(_p, "MOTOR_ACCEL_RATIO", 5.0))
                alpha = float(getattr(_p, "ACTION_SMOOTH_ALPHA", 1.0))
                print(
                    f"[view] reloaded motor params: SPEED_FRAC={speed_frac} "
                    f"ACCEL_RATIO={accel_ratio} ACTION_ALPHA={alpha}",
                    flush=True,
                )
            else:
                _p = _params_mod[0]
                speed_frac = float(getattr(_p, "MOTOR_SPEED_FRAC", 1.0))
                accel_ratio = float(getattr(_p, "MOTOR_ACCEL_RATIO", 5.0))
                alpha = float(getattr(_p, "ACTION_SMOOTH_ALPHA", 1.0))
        except Exception as e:
            print(f"[view] params reload failed, using defaults: {e}", flush=True)
            speed_frac, accel_ratio, alpha = 1.0, 5.0, 0.45
        speed_arr = speed_frac * motor_max_radps * action_dt / half_range
        accel_arr = speed_arr * accel_ratio * action_dt
        return speed_arr, accel_arr, alpha

    motor_max_speed_arr, motor_max_accel_arr, ACTION_ALPHA = reload_motor_params()

    tree_xy = None
    tree_top_z = 2.5
    for name in ("t0_trunk", "tree_0_trunk_col"):
        try:
            gid = model.geom(name).id
            p1 = model.geom_pos[gid].copy()
            # capsule fromto isn't directly stored; estimate top from size+pos.
            tree_xy = np.array(p1[:2], dtype=np.float32)
            tree_top_z = float(
                p1[2] + model.geom_size[gid][1] + model.geom_size[gid][0]
            )
            print(
                f"[view] tree '{name}' at xy={tree_xy.tolist()} top_z≈{tree_top_z:.2f}"
            )
            break
        except (KeyError, ValueError):
            continue

    predict = None
    last_ckpt = None
    last_mtime = 0.0
    if POLICY == "latest":
        ckpt = latest_checkpoint()
        if ckpt is not None:
            predict = load_predict(ckpt)
            last_ckpt = ckpt
            last_mtime = ckpt.stat().st_mtime
            print(f"[view] loaded {ckpt.name}")
        else:
            print(
                "[view] no checkpoint yet — using random policy, will hot-swap when one appears"
            )
    print(
        f"[view] PHYSICS_STEPS={PHYSICS_STEPS}, action={PHYSICS_STEPS * DT * 1000:.0f}ms sim time, policy={POLICY}"
    )

    viewer = mujoco.viewer.launch_passive(model, data)
    rng = np.random.default_rng(0)
    ep = 0

    def do_reset():
        mujoco.mj_resetData(model, data)
        # Match training reset: spawn at origin facing the tree (±0.5 rad noise).
        # If we randomize yaw across 360°, the policy sees orientations it was
        # never trained on and runs off in arbitrary directions.
        if ENV == "walk" and tree_xy is not None:
            yaw_target = float(np.arctan2(tree_xy[1], tree_xy[0]))
            yaw = yaw_target + rng.uniform(-0.5, 0.5)
        else:
            yaw = rng.uniform(-np.pi, np.pi)
        data.qpos[0] = rng.uniform(-0.1, 0.1)
        data.qpos[1] = rng.uniform(-0.1, 0.1)
        # spawn at the patched torso height (set by gen_terrain.py based on
        # terrain). MJX training uses a low spawn (0.10) as a curriculum
        # — view.py is for inspection so we use the proper standing height.
        data.qpos[2] = float(model.body("torso").pos[2])
        data.qpos[3] = np.cos(yaw / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = np.sin(yaw / 2)
        # Robot joints only — slice [7:7+ACT_DIM] so we don't write into
        # grove DOFs (branch ball joints, future olive freejoints).
        data.qpos[7 : 7 + ACT_DIM] = rng.uniform(-0.15, 0.15, size=ACT_DIM)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)

    print(f"{'ep':>4}  {'status':<8}  {'torso_z':>8}  {'steps':>6}  {'upright':>8}")

    try:
        while viewer.is_running():
            # Hot-swap to latest checkpoint between episodes (cheap: just file stat).
            if POLICY == "latest":
                ckpt = latest_checkpoint()
                if ckpt is not None:
                    mtime = ckpt.stat().st_mtime
                    if mtime != last_mtime:
                        try:
                            predict = load_predict(ckpt)
                            last_ckpt = ckpt
                            last_mtime = mtime
                            tag = "FIRST" if predict is None else "NEW"
                            print(f"[view] >>> {tag} POLICY: {ckpt.name} <<<")
                        except Exception as e:
                            print(f"[view] skip {ckpt.name}: {e}")
                            last_mtime = mtime

            do_reset()
            motor_pos = np.zeros(ACT_DIM, dtype=np.float32)
            motor_vel = np.zeros(ACT_DIM, dtype=np.float32)
            filt_action = np.zeros(ACT_DIM, dtype=np.float32)
            # Hot-reload motor params from envs/params_walk_mjx.py — picks up edits
            # within ~one episode (no viewer/trainer restart needed).
            motor_max_speed_arr, motor_max_accel_arr, ACTION_ALPHA = (
                reload_motor_params()
            )
            substep_speed = motor_max_speed_arr / PHYSICS_STEPS
            substep_accel = motor_max_accel_arr / PHYSICS_STEPS
            step_i = 0
            MAX_EP = 250  # ~8s per episode — enough to see motion, short enough to keep cycling

            # Settle on the ground before the policy takes over.
            # Drive ctrl to rest_pose so the robot stands rather than starfishes.
            settle_seconds = 2.0
            settle_ctrl = rest_pose.copy()
            settle_steps = int(settle_seconds / DT)
            t_settle = time.time()
            for k in range(settle_steps):
                data.ctrl[:] = settle_ctrl
                mujoco.mj_step(model, data)
                if k % PHYSICS_STEPS == 0:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    lag = (k + PHYSICS_STEPS) * DT - (time.time() - t_settle)
                    if lag > 0:
                        time.sleep(lag)
            if not viewer.is_running():
                break

            # ── scripted demo: mocap-style approach + climb ──
            if POLICY == "demo":
                # phases (in viewer-DT seconds, DT=0.002, sleep=DT each substep)
                FPS = 1.0 / DT
                T_APPROACH = 4.0
                T_CLIMB = 4.0
                T_HOLD = 1.5
                T_TOTAL = T_APPROACH + T_CLIMB + T_HOLD
                start_xy = np.array([0.0, 0.0], dtype=np.float32)
                tx, ty = tree_xy if tree_xy is not None else np.array([2.5, 0.0])
                # stop short of trunk so robot is "at" the tree, not inside it
                approach_xy = np.array([tx - 0.45, ty], dtype=np.float32)
                ground_z = float(model.body("torso").pos[2])
                top_z = max(tree_top_z - 0.3, ground_z + 1.5)
                t_demo = 0.0

                while t_demo < T_TOTAL and viewer.is_running():
                    # face tree
                    yaw = float(np.arctan2(ty - data.qpos[1], tx - data.qpos[0]))
                    if t_demo < T_APPROACH:
                        u = t_demo / T_APPROACH
                        u_smooth = 0.5 - 0.5 * np.cos(np.pi * u)  # ease in/out
                        data.qpos[0:2] = start_xy + (approach_xy - start_xy) * u_smooth
                        data.qpos[2] = ground_z
                        # trot gait
                        ph = t_demo * 6.0
                        s, c = np.sin(ph), np.cos(ph)
                        joints = np.zeros(ACT_DIM, dtype=np.float32)
                        joints[1] = 0.6 * s
                        joints[7] = -0.6 * s
                        joints[13] = -0.6 * s
                        joints[19] = 0.6 * s
                        joints[2] = 0.5 + 0.4 * c
                        joints[8] = 0.5 - 0.4 * c
                        joints[14] = 0.5 - 0.4 * c
                        joints[20] = 0.5 + 0.4 * c
                    elif t_demo < T_APPROACH + T_CLIMB:
                        u = (t_demo - T_APPROACH) / T_CLIMB
                        u_smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                        data.qpos[0:2] = approach_xy
                        data.qpos[2] = ground_z + (top_z - ground_z) * u_smooth
                        # climbing: arms reach overhead, alternating; legs grip
                        ph = (t_demo - T_APPROACH) * 5.0
                        s, c = np.sin(ph), np.cos(ph)
                        joints = np.zeros(ACT_DIM, dtype=np.float32)
                        # front limbs reach up + alternate
                        joints[1] = -0.85 + 0.25 * s  # FL shoulder pitch (up)
                        joints[7] = -0.85 - 0.25 * s  # FR
                        joints[2] = 0.6 + 0.3 * c  # FL elbow
                        joints[8] = 0.6 - 0.3 * c
                        # back legs hug trunk
                        joints[13] = 0.6 + 0.2 * s
                        joints[19] = 0.6 - 0.2 * s
                        joints[14] = 0.9
                        joints[20] = 0.9
                        # torso pitched up against trunk
                        pitch = -0.6 * u_smooth
                    else:
                        data.qpos[0:2] = approach_xy
                        data.qpos[2] = top_z
                        joints = np.zeros(ACT_DIM, dtype=np.float32)
                        joints[1] = -0.9
                        joints[7] = -0.9
                        joints[2] = 0.7
                        joints[8] = 0.7
                        joints[13] = 0.7
                        joints[19] = 0.7
                        joints[14] = 0.9
                        joints[20] = 0.9

                    # set torso orientation: yaw toward tree, pitch up while climbing
                    pitch = (
                        -0.6 * np.clip((t_demo - T_APPROACH) / T_CLIMB, 0, 1)
                        if t_demo > T_APPROACH
                        else 0.0
                    )
                    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
                    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
                    data.qpos[3] = cy * cp
                    data.qpos[4] = cy * sp  # pitch
                    data.qpos[5] = 0.0
                    data.qpos[6] = sy * cp

                    # joints: clip to [-1,1] then map to joint range
                    joints = np.clip(joints, -1.0, 1.0)
                    data.qpos[7 : 7 + ACT_DIM] = (
                        0.5 * (joints + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
                    )
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(DT * PHYSICS_STEPS)
                    t_demo += DT * PHYSICS_STEPS
                    step_i += 1

                ep += 1
                print(
                    f"{ep:>4}  {'CLIMB':<8}  {top_z:>8.3f}m  {step_i:>6}  {1.0:>8.3f}"
                )
                continue

            while step_i < MAX_EP and viewer.is_running():
                if POLICY == "zero":
                    action = np.zeros(ACT_DIM, dtype=np.float32)
                elif POLICY == "random":
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
                elif POLICY == "demo":
                    # Trot-style gait: diagonal pairs (FL+BR vs FR+BL) swing
                    # opposite phase. Indices match scene.xml actuator order.
                    t = step_i * 0.06
                    a = np.zeros(ACT_DIM, dtype=np.float32)
                    s = np.sin(t)
                    c = np.cos(t)
                    # shoulder/hip pitch — alternating swing
                    a[1] = 0.55 * s  # FL shoulder pitch
                    a[7] = -0.55 * s  # FR
                    a[13] = -0.55 * s  # BL hip pitch
                    a[19] = 0.55 * s  # BR
                    # elbow/knee — fold during swing-back
                    a[2] = 0.40 + 0.35 * c  # FL elbow
                    a[8] = 0.40 - 0.35 * c  # FR
                    a[14] = 0.40 - 0.35 * c  # BL knee
                    a[20] = 0.40 + 0.35 * c  # BR
                    action = a
                elif predict is not None:
                    # robot joints only — bound slices to ACT_DIM so grove
                    # DOFs (branches/olives) don't poison the observation.
                    # Match the 88-dim contract from envs/_shared_obs.py:
                    #   [0]     torso_z (1)
                    #   [1:5]   quat (4)
                    #   [5:8]   linvel (3)
                    #   [8:11]  angvel (3)
                    #   [11:35] joint positions (24)
                    #   [35:59] joint velocities (24)
                    #   [59:83] joint torques (24)
                    #   [83:86] target dx, dy, distance (3) - zeros for stand/turn
                    #   [86:88] gait phase sin, cos (2)
                    # For walk, use tree_xy as target; for stand/turn, use zeros
                    target = (
                        tree_xy
                        if tree_xy is not None
                        else np.zeros(2, dtype=np.float32)
                    )
                    rel = target.astype(np.float32) - data.qpos[0:2].astype(np.float32)
                    dist = np.linalg.norm(rel)
                    proprio = np.concatenate(
                        [
                            data.qpos[2:3],  # 1   torso z
                            data.qpos[3:7],  # 4   quat
                            data.qvel[0:3],  # 3   linvel
                            data.qvel[3:6],  # 3   angvel
                            data.qpos[7 : 7 + ACT_DIM],  # 24  joint positions
                            data.qvel[6 : 6 + ACT_DIM],  # 24  joint velocities
                            data.qfrc_actuator[6 : 6 + ACT_DIM],  # 24  joint torques
                        ],
                        dtype=np.float32,
                    )
                    target_slot = np.concatenate([rel, [dist]], dtype=np.float32)  # 3
                    ph = (2.0 * np.pi / GAIT_PERIOD_STEPS) * step_i
                    gait_clock = np.array(
                        [np.sin(ph), np.cos(ph)], dtype=np.float32
                    )  # 2
                    obs = np.concatenate(
                        [proprio, target_slot, gait_clock], dtype=np.float32
                    )
                    action = predict(obs)
                else:
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)

                # EMA lowpass on policy output — kills micro-oscillations.
                filt_action = ACTION_ALPHA * action + (1.0 - ACTION_ALPHA) * filt_action

                # CPG calibrated trot (mirrors envs/walk_env_mjx.py).
                try:
                    import envs.params_walk_mjx as _p_view

                    w_cpg_pitch = float(getattr(_p_view, "W_CPG_PITCH", 0.0))
                    w_cpg_elbow = float(getattr(_p_view, "W_CPG_ELBOW", 0.0))
                    GAIT_PERIOD = 40.0
                except Exception:
                    w_cpg_pitch = w_cpg_elbow = 0.0
                    GAIT_PERIOD = 40.0
                LIMB_PHASE = np.array(
                    [0.0, np.pi, 0.0, np.pi]
                )  # pace: left in phase, right in phase
                phase = 2.0 * np.pi * step_i / GAIT_PERIOD + LIMB_PHASE
                sin_p = np.sin(phase)
                cpg = np.zeros(ACT_DIM, dtype=np.float32)
                cpg[0] = -w_cpg_pitch * sin_p[0]  # FL_yaw
                cpg[1] = -w_cpg_elbow * max(sin_p[0], 0.0)  # FL_pitch (lift)
                cpg[6] = +w_cpg_pitch * sin_p[1]  # FR_yaw
                cpg[7] = -w_cpg_elbow * max(sin_p[1], 0.0)  # FR_pitch (lift)
                cpg[13] = +w_cpg_pitch * sin_p[2]  # BL_pitch
                cpg[14] = +w_cpg_elbow * max(sin_p[2], 0.0)  # BL_knee (lift)
                cpg[19] = +w_cpg_pitch * sin_p[3]  # BR_pitch
                cpg[20] = +w_cpg_elbow * max(sin_p[3], 0.0)  # BR_knee (lift)
                filt_action_with_cpg = filt_action + cpg

                # One sync per action step, then sleep the remainder of the
                # action's wall-clock budget so playback tracks real time even
                # when mj_step is heavy (e.g. branched tree contacts).
                t0 = time.time()
                for _ in range(PHYSICS_STEPS):
                    err = filt_action_with_cpg - motor_pos
                    desired_vel = np.clip(err, -substep_speed, substep_speed)
                    motor_vel = np.clip(
                        desired_vel,
                        motor_vel - substep_accel,
                        motor_vel + substep_accel,
                    )
                    motor_vel = np.clip(motor_vel, -substep_speed, substep_speed)
                    motor_pos += motor_vel
                    data.ctrl[:] = np.where(
                        motor_pos >= 0,
                        rest_pose + motor_pos * (jnt_hi - rest_pose),
                        rest_pose + motor_pos * (rest_pose - jnt_lo),
                    )
                    mujoco.mj_step(model, data)
                viewer.sync()
                lag = DT * PHYSICS_STEPS - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)

                step_i += 1
                if data.qpos[3] < 0:  # flipped
                    break

            if not viewer.is_running():
                break

            ep += 1
            torso_z = float(data.qpos[2])
            upright = float(data.qpos[3])
            status = (
                "FLIPPED"
                if data.qpos[3] < 0
                else ("TIMEOUT" if step_i >= MAX_EP else "DONE")
            )
            print(
                f"{ep:>4}  {status:<8}  {torso_z:>8.3f}m  {step_i:>6}  {upright:>8.3f}"
            )

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("[view] closed.")


if __name__ == "__main__":
    main()
