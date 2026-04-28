"""
Live viewer for MJX training.

Loads .npz Flax checkpoints and runs them on CPU MuJoCo.
Hot-swaps to the latest checkpoint whenever training saves a new one.

Usage:
    python utils/watch_mjx.py
    python utils/watch_mjx.py --since <unix_ts>   # ignore old checkpoints
"""

import argparse
import os
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
import mujoco.viewer

from envs.stand_env_mjx import OBS_DIM, ACT_DIM, MAX_STEPS, PHYSICS_STEPS_PER_ACTION, SCENE
import envs.params_mjx as P

RUNS = pathlib.Path("runs/ape_stand_mjx")

_parser = argparse.ArgumentParser()
_parser.add_argument("--since", type=float, default=0.0,
                     help="ignore checkpoints saved before this unix timestamp")
_args = _parser.parse_args()
SINCE = _args.since


class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(256)(obs))
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(128)(x))
        mean    = nn.Dense(ACT_DIM)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (ACT_DIM,))
        value   = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value


def load_policy(path):
    net = ActorCritic()
    dummy = jnp.zeros((1, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), dummy)
    data = np.load(str(path))
    flat = [jnp.asarray(data[k]) for k in data.files]
    _, treedef = jax.tree_util.tree_flatten(params)
    loaded = jax.tree_util.tree_unflatten(treedef, flat)
    first_kernel = next((x for x in flat if x.ndim == 2), None)
    if first_kernel is None or first_kernel.shape[0] != OBS_DIM:
        got = first_kernel.shape[0] if first_kernel is not None else "?"
        raise ValueError(
            f"Checkpoint obs_dim={got} != current OBS_DIM={OBS_DIM}. "
            "Delete old checkpoints and restart training."
        )
    return net, loaded


def predict(net, params, obs):
    mean, _, _ = net.apply(params, jnp.array(obs, dtype=jnp.float32))
    return np.array(jnp.tanh(mean))


def latest_checkpoint():
    ckpts = [c for c in sorted(RUNS.glob("mjx_*.npz")) if c.stat().st_mtime >= SINCE]
    return ckpts[-1] if ckpts else None


def get_obs(data):
    return np.concatenate([
        data.qpos[2:3],
        data.qpos[3:7],
        data.qvel[0:3],
        data.qvel[3:6],
        data.qpos[7:],
        data.qvel[6:],
    ]).astype(np.float32)


def env_reset(model, data, rng):
    mujoco.mj_resetData(model, data)
    yaw = rng.uniform(-np.pi, np.pi)
    data.qpos[0] = rng.uniform(-0.1, 0.1)
    data.qpos[1] = rng.uniform(-0.1, 0.1)
    data.qpos[2] = 0.45
    data.qpos[3] = np.cos(yaw / 2)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = np.sin(yaw / 2)
    data.qpos[7:] = rng.uniform(-0.15, 0.15, size=24)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)


def main():
    print("[watch_mjx] loading model...")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # Disable all robot self-collision for CPU viewer speed.
    # Floor (plane) stays at contype=1; robot geoms → contype=2 so they
    # collide with the floor (conaffinity=31) but never with each other.
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_contype[i] = 2
            model.geom_conaffinity[i] = 1
    data = mujoco.MjData(model)
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, model.njnt)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, model.njnt)])

    viewer = mujoco.viewer.launch_passive(model, data)
    rng = np.random.default_rng(0)

    ckpt = latest_checkpoint()
    if ckpt is None:
        print("[watch_mjx] no checkpoint yet — using random policy, waiting...")
        net, params = None, None
        last_mtime = 0.0
    else:
        print(f"[watch_mjx] loaded {ckpt.name}")
        net, params = load_policy(ckpt)
        last_mtime = ckpt.stat().st_mtime

    ep = 0
    try:
        while viewer.is_running():
            # Hot-swap to latest checkpoint
            new_ckpt = latest_checkpoint()
            if new_ckpt is not None:
                mtime = new_ckpt.stat().st_mtime
                if mtime != last_mtime:
                    try:
                        was_none = net is None
                        net, params = load_policy(new_ckpt)
                        try:
                            steps = int(new_ckpt.stem.split("_")[1])
                            step_str = f"{steps:,} steps"
                        except (IndexError, ValueError):
                            step_str = "?"
                        tag = "FIRST" if was_none else "NEW"
                        print(f"[watch_mjx] >>> {tag} POLICY: {new_ckpt.name}  ({step_str}) <<<")
                        last_mtime = mtime
                    except ValueError as e:
                        print(f"[watch_mjx] skipping {new_ckpt.name}: {e}")
                        last_mtime = mtime

            env_reset(model, data, rng)
            motor_pos   = np.zeros(ACT_DIM, dtype=np.float32)
            prev_action = np.zeros(ACT_DIM, dtype=np.float32)
            step_i      = 0
            flipped     = False
            ep_reward   = 0.0

            while step_i < MAX_STEPS and viewer.is_running():
                action_sim_dt  = PHYSICS_STEPS_PER_ACTION * float(model.opt.timestep)
                action_wall_t0 = time.time()

                obs = get_obs(data)
                if net is None:
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
                else:
                    action = predict(net, params, obs)

                alpha = float(getattr(P, "ACTION_SMOOTH_ALPHA", 1.0))
                filt_action = (alpha * action + (1.0 - alpha) * prev_action).astype(np.float32)
                prev_action = filt_action

                motor_max_speed = float(getattr(P, "MOTOR_MAX_SPEED", 0.2))
                substep_speed   = motor_max_speed / PHYSICS_STEPS_PER_ACTION
                SYNC_INTERVAL   = 1.0 / 30.0
                last_sync_t     = time.time()

                for _ in range(PHYSICS_STEPS_PER_ACTION):
                    delta = np.clip(filt_action - motor_pos, -substep_speed, substep_speed)
                    motor_pos += delta
                    ctrl = 0.5 * (motor_pos + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
                    data.ctrl[:] = ctrl
                    mujoco.mj_step(model, data)
                    if time.time() - last_sync_t >= SYNC_INTERVAL:
                        viewer.sync()
                        last_sync_t = time.time()
                viewer.sync()

                behind = action_sim_dt - (time.time() - action_wall_t0)
                if behind > 0:
                    time.sleep(behind)

                # Track a rough per-step reward for episode stats
                torso_z = float(data.qpos[2])
                w       = float(data.qpos[3])
                tgt_z   = float(getattr(P, "TARGET_Z", 0.3))
                tgt_tol = float(getattr(P, "TARGET_TOL", 0.1))
                h_err   = (torso_z - tgt_z) / max(tgt_tol, 1e-6)
                ep_reward += (
                    float(getattr(P, "W_HEIGHT", 10.0)) * float(np.exp(-0.5 * h_err * h_err))
                    + (float(getattr(P, "W_UPRIGHT", 5.0)) * w * w if w > 0 else 0.0)
                    + (float(getattr(P, "W_ALIVE", 0.05)) if w > 0 else 0.0)
                )

                if data.qpos[3] < 0:
                    flipped = True
                    break
                step_i += 1

            if not viewer.is_running():
                break

            ep += 1
            torso_z    = float(data.qpos[2])
            avg_reward = ep_reward / max(step_i, 1)
            status     = "FLIPPED" if flipped else ("TIMEOUT" if step_i >= MAX_STEPS else "DONE")
            ckpt_name  = new_ckpt.name if new_ckpt else "random"
            print(f"Ep {ep:3d} {status:<7}  torso={torso_z:.3f}m  len={step_i:>4}  "
                  f"avg_rew={avg_reward:.3f}  [{ckpt_name}]")

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("[watch_mjx] closed.")


if __name__ == "__main__":
    main()
