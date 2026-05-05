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
PHYSICS_STEPS = 32         # physics sub-steps per action (matches training)
MOTOR_SPEED   = 0.10       # max joint travel per action (matches training)
# ─────────────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser()
_parser.add_argument("--since", type=float, default=0.0,
                     help="ignore checkpoints saved before this unix timestamp")
_parser.add_argument("--env", choices=["stand", "walk"], default="stand",
                     help="which env was trained (selects runs dir + obs format)")
_args = _parser.parse_args()
SINCE = _args.since
ENV   = _args.env
RUNS  = pathlib.Path("runs/ape_walk_mjx" if ENV == "walk" else "runs/ape_stand_mjx")
OBS_DIM = 86 if ENV == "walk" else 83


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

    ACT_DIM = 24

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
    ACT_DIM = model.nu
    DT      = float(model.opt.timestep)
    # Skip joint 0 (the torso freejoint, no range) and stop after the 24
    # actuated robot joints — branch ball joints don't enter the action
    # space.
    jnt_lo  = np.array([model.jnt_range[i][0] for i in range(1, 1 + ACT_DIM)])
    jnt_hi  = np.array([model.jnt_range[i][1] for i in range(1, 1 + ACT_DIM)])

    tree_xy = None
    if ENV == "walk":
        try:
            gid = model.geom("t0_trunk").id
            tree_xy = np.array(model.geom_pos[gid][:2], dtype=np.float32)
            print(f"[view] walk env, tree at xy={tree_xy.tolist()}")
        except (KeyError, ValueError):
            print("[view] WARNING: walk env but no t0_trunk geom — run gen_trees.py --inject")

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
        # spawn at the patched torso height (set by gen_terrain.py based on
        # terrain). MJX training uses a low spawn (0.10) as a curriculum
        # — view.py is for inspection so we use the proper standing height.
        data.qpos[2]  = float(model.body("torso").pos[2])
        data.qpos[3]  = np.cos(yaw / 2)
        data.qpos[4]  = 0.0
        data.qpos[5]  = 0.0
        data.qpos[6]  = np.sin(yaw / 2)
        # Robot joints only — slice [7:7+ACT_DIM] so we don't write into
        # grove DOFs (branch ball joints, future olive freejoints).
        data.qpos[7:7 + ACT_DIM] = rng.uniform(-0.15, 0.15, size=ACT_DIM)
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
                    # robot joints only — bound slices to ACT_DIM so grove
                    # DOFs (branches/olives) don't poison the observation.
                    obs_parts = [
                        data.qpos[2:3], data.qpos[3:7],
                        data.qvel[0:3], data.qvel[3:6],
                        data.qpos[7:7 + ACT_DIM], data.qvel[6:6 + ACT_DIM],
                        data.qfrc_actuator[6:6 + ACT_DIM],
                    ]
                    if ENV == "walk" and tree_xy is not None:
                        rel = tree_xy - data.qpos[0:2].astype(np.float32)
                        obs_parts.append(rel)
                        obs_parts.append(np.array([np.linalg.norm(rel)], dtype=np.float32))
                    obs = np.concatenate(obs_parts).astype(np.float32)
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
