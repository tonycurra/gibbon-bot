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
import numpy as np
import mujoco
import mujoco.viewer

# ── config ────────────────────────────────────────────────────────────────────
POLICY        = "latest"   # "zero" | "random" | "latest"
SCENE         = pathlib.Path("data/scene.xml")
RUNS          = pathlib.Path("runs/ape_stand_mjx")
PHYSICS_STEPS = 32         # physics sub-steps per action (matches training)
MOTOR_SPEED   = 0.10       # max joint travel per action (matches training)
# ─────────────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser()
_parser.add_argument("--since", type=float, default=0.0,
                     help="ignore checkpoints saved before this unix timestamp")
_args = _parser.parse_args()
SINCE = _args.since


def latest_checkpoint():
    ckpts = [c for c in sorted(RUNS.glob("mjx_*.npz")) if c.stat().st_mtime >= SINCE]
    return ckpts[-1] if ckpts else None


def _build_net():
    """Construct the actor-critic network. Returns (net, OBS_DIM, ACT_DIM, jnp)."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    ACT_DIM, OBS_DIM = 24, 83

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

    return Net(), OBS_DIM, ACT_DIM, jax, jnp


def load_predict(path, cached=[None]):
    """Load (or hot-swap) policy from .npz checkpoint. Reuses the JIT'd apply_fn."""
    if cached[0] is None:
        net, OBS_DIM, ACT_DIM, jax, jnp = _build_net()
        init_params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
        cached[0] = (net, init_params, jax, jnp)
    net, init_params, jax, jnp = cached[0]

    data = np.load(str(path))
    flat = [jnp.asarray(data[k]) for k in data.files]
    _, treedef = jax.tree_util.tree_flatten(init_params)
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
            model.geom_contype[i]     = 2
            model.geom_conaffinity[i] = 1

    data    = mujoco.MjData(model)
    jnt_lo  = np.array([model.jnt_range[i][0] for i in range(1, model.njnt)])
    jnt_hi  = np.array([model.jnt_range[i][1] for i in range(1, model.njnt)])
    ACT_DIM = model.nu
    DT      = float(model.opt.timestep)

    predict     = None
    last_ckpt   = None
    last_mtime  = 0.0
    if POLICY == "latest":
        ckpt = latest_checkpoint()
        if ckpt is not None:
            predict    = load_predict(ckpt)
            last_ckpt  = ckpt
            last_mtime = ckpt.stat().st_mtime
            print(f"[view] loaded {ckpt.name}")
        else:
            print("[view] no checkpoint yet — using random policy, will hot-swap when one appears")
    print(f"[view] PHYSICS_STEPS={PHYSICS_STEPS}, action={PHYSICS_STEPS*DT*1000:.0f}ms sim time, policy={POLICY}")

    viewer = mujoco.viewer.launch_passive(model, data)
    rng    = np.random.default_rng(0)
    ep     = 0

    def do_reset():
        mujoco.mj_resetData(model, data)
        yaw = rng.uniform(-np.pi, np.pi)
        data.qpos[0]  = rng.uniform(-0.1, 0.1)
        data.qpos[1]  = rng.uniform(-0.1, 0.1)
        data.qpos[2]  = 0.10
        data.qpos[3]  = np.cos(yaw / 2)
        data.qpos[4]  = 0.0
        data.qpos[5]  = 0.0
        data.qpos[6]  = np.sin(yaw / 2)
        data.qpos[7:] = rng.uniform(-0.15, 0.15, size=model.nq - 7)
        data.qvel[:]  = 0.0
        data.ctrl[:]  = 0
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
                            predict   = load_predict(ckpt)
                            last_ckpt = ckpt
                            last_mtime = mtime
                            tag = "FIRST" if predict is None else "NEW"
                            print(f"[view] >>> {tag} POLICY: {ckpt.name} <<<")
                        except Exception as e:
                            print(f"[view] skip {ckpt.name}: {e}")
                            last_mtime = mtime

            do_reset()
            motor_pos     = np.zeros(ACT_DIM, dtype=np.float32)
            substep_speed = MOTOR_SPEED / PHYSICS_STEPS
            step_i        = 0
            MAX_EP        = 2000

            while step_i < MAX_EP and viewer.is_running():
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

                # Per-step sync + per-step sleep — the pattern that worked in
                # every viewer from 2ea623f through dfae092. With DT=0.002s
                # this gives 500 viewer frames per second and real-time playback.
                for _ in range(PHYSICS_STEPS):
                    delta = np.clip(action - motor_pos, -substep_speed, substep_speed)
                    motor_pos += delta
                    data.ctrl[:] = 0.5 * (motor_pos + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(DT)

                step_i += 1
                if data.qpos[3] < 0:  # flipped
                    break

            if not viewer.is_running():
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
