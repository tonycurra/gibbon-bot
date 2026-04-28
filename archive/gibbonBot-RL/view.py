#!/usr/bin/env python3
"""
Real-time MuJoCo viewer — no training, no checkpoints.
Just the robot in the physics sim so you can see if things look right.

Policy modes (pick one at the top of this file):
  "zero"    — all joints at rest, see how the robot falls/settles
  "random"  — random actions each step, chaotic but shows range of motion
  "latest"  — load the most recent training checkpoint if one exists

Usage:
    python view.py
"""

import pathlib
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── config ────────────────────────────────────────────────────────────────────
POLICY        = "latest"   # "zero" | "random" | "latest"
SCENE         = pathlib.Path("data/scene.xml")
RUNS          = pathlib.Path("runs/ape_stand_mjx")
PHYSICS_STEPS = 200        # physics steps per action (must match training)
MOTOR_SPEED   = 0.10       # max joint travel per action (must match training)
# ─────────────────────────────────────────────────────────────────────────────


def load_policy_net():
    """Load latest .npz checkpoint. Returns (net, params) or None."""
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    if not ckpts:
        return None
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    ACT_DIM = 24
    OBS_DIM = 83

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(128)(x))
            mean    = nn.Dense(ACT_DIM)(x)
            log_std = self.param("log_std", nn.initializers.zeros, (ACT_DIM,))
            value   = nn.Dense(1)(x).squeeze(-1)
            return mean, log_std, value

    net = Net()
    dummy = jnp.zeros((1, OBS_DIM))
    init_params = net.init(jax.random.PRNGKey(0), dummy)
    data = np.load(str(ckpts[-1]))
    flat = [jnp.asarray(data[k]) for k in data.files]
    _, treedef = jax.tree_util.tree_flatten(init_params)
    params = jax.tree_util.tree_unflatten(treedef, flat)
    print(f"[view] loaded {ckpts[-1].name}")

    def predict(obs):
        import jax.numpy as jnp
        mean, _, _ = net.apply(params, jnp.array(obs, dtype=jnp.float32))
        import jax
        return np.array(jax.nn.tanh(mean))

    return predict


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # Disable all robot self-collision for CPU viewer speed.
    # Keep floor (plane geom) at contype=1 so the robot still lands on it.
    # Robot geoms → contype=2, conaffinity=1: they collide with the floor
    # (floor conaffinity=31 covers bit 2) but never with each other.
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_contype[i] = 2
            model.geom_conaffinity[i] = 1

    data  = mujoco.MjData(model)
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, model.njnt)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, model.njnt)])
    ACT_DIM = model.nu

    predict = None
    if POLICY == "latest":
        result = load_policy_net()
        if result is None:
            print("[view] no checkpoint found — falling back to random policy")
            predict = None
        else:
            predict = result
    elif POLICY == "zero":
        print("[view] zero policy — joints at rest")
    elif POLICY == "random":
        print("[view] random policy")

    viewer = mujoco.viewer.launch_passive(model, data)
    rng    = np.random.default_rng(0)
    ep     = 0

    def do_reset():
        mujoco.mj_resetData(model, data)
        yaw = rng.uniform(-np.pi, np.pi)
        data.qpos[0] = rng.uniform(-0.1, 0.1)
        data.qpos[1] = rng.uniform(-0.1, 0.1)
        data.qpos[2] = 0.45
        data.qpos[3] = np.cos(yaw / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = np.sin(yaw / 2)
        data.qpos[7:] = rng.uniform(-0.15, 0.15, size=model.nq - 7)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)

    print("[view] press Ctrl+C or close the window to quit")
    print(f"{'ep':>4}  {'status':<8}  {'torso_z':>8}  {'steps':>6}  {'upright':>8}")

    try:
        while viewer.is_running():
            do_reset()
            motor_pos   = np.zeros(ACT_DIM, dtype=np.float32)
            prev_action = np.zeros(ACT_DIM, dtype=np.float32)
            step_i = 0
            MAX_EP = 2000
            SYNC_HZ = 1.0 / 30.0
            action_dt = PHYSICS_STEPS * float(model.opt.timestep)

            while step_i < MAX_EP and viewer.is_running():
                t0 = time.time()

                # Get action
                if POLICY == "zero":
                    action = np.zeros(ACT_DIM, dtype=np.float32)
                elif POLICY == "random":
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
                elif predict is not None:
                    obs = np.concatenate([
                        data.qpos[2:3], data.qpos[3:7],
                        data.qvel[0:3], data.qvel[3:6],
                        data.qpos[7:],  data.qvel[6:],
                        data.qfrc_actuator[6:30],
                    ]).astype(np.float32)
                    action = predict(obs)
                else:
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)

                substep_speed = MOTOR_SPEED / PHYSICS_STEPS
                last_sync = time.time()
                for _ in range(PHYSICS_STEPS):
                    delta = np.clip(action - motor_pos, -substep_speed, substep_speed)
                    motor_pos += delta
                    ctrl = 0.5 * (motor_pos + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
                    data.ctrl[:] = ctrl
                    mujoco.mj_step(model, data)
                    if time.time() - last_sync >= SYNC_HZ:
                        viewer.sync()
                        last_sync = time.time()
                viewer.sync()

                # Real-time pacing
                behind = action_dt - (time.time() - t0)
                if behind > 0:
                    time.sleep(behind)

                prev_action = action.copy()
                step_i += 1

                if data.qpos[3] < 0:  # flipped
                    break

            ep += 1
            torso_z = float(data.qpos[2])
            upright = float(data.qpos[3])
            status  = "FLIPPED" if data.qpos[3] < 0 else ("TIMEOUT" if step_i >= MAX_EP else "DONE")
            print(f"{ep:>4}  {status:<8}  {torso_z:>8.3f}m  {step_i:>6}  {upright:>8.3f}")

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("[view] closed.")


if __name__ == "__main__":
    main()
