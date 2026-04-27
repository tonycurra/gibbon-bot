"""
Live viewer for MJX training — loads .npz Flax checkpoints and runs them
on CPU MuJoCo with the same step semantics as MJX (direct ctrl, 16 physics steps).

Hot-swaps policy when a newer checkpoint appears.

Usage:
    python utils/watch_mjx.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
import mujoco.viewer

from envs.common import SCENE, MAX_STEPS, ACT_DIM
from envs.stand_env_mjx import OBS_DIM, PHYSICS_STEPS_PER_ACTION
import envs.params as params_mod
import importlib

RUNS = pathlib.Path("runs/ape_stand_mjx")
PARAMS_FILE = pathlib.Path(__file__).parent.parent / "envs" / "params.py"


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
    return net, jax.tree_util.tree_unflatten(treedef, flat)


def predict(net, params, obs):
    obs_j = jnp.array(obs, dtype=jnp.float32)
    mean, _, _ = net.apply(params, obs_j)
    return np.array(jnp.tanh(mean))


def latest_checkpoint():
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
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


def reset(model, data, rng):
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
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, model.njnt)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, model.njnt)])

    viewer = mujoco.viewer.launch_passive(model, data)
    rng = np.random.default_rng(0)

    ckpt = latest_checkpoint()
    if ckpt is None:
        print(f"[watch_mjx] no checkpoint yet in {RUNS} — random policy, waiting...")
        net, params = None, None
        last_mtime = 0.0
    else:
        print(f"[watch_mjx] loading {ckpt.name} ...")
        net, params = load_policy(ckpt)
        last_mtime = ckpt.stat().st_mtime

    params_mtime = 0.0

    ep = 0
    try:
        while viewer.is_running():
            new_ckpt = latest_checkpoint()
            if new_ckpt is not None:
                mtime = new_ckpt.stat().st_mtime
                if mtime != last_mtime:
                    print(f"[watch_mjx] {'first' if net is None else 'new'} checkpoint: {new_ckpt.name}")
                    net, params = load_policy(new_ckpt)
                    last_mtime = mtime

            reset(model, data, rng)
            step = 0
            flipped = False

            while step < MAX_STEPS and viewer.is_running():
                # hot-reload params
                try:
                    pm = PARAMS_FILE.stat().st_mtime
                    if pm != params_mtime:
                        importlib.reload(params_mod)
                        params_mtime = pm
                except Exception:
                    pass
                speed = max(getattr(params_mod, "REALTIME_FACTOR", 1.0), 0.01)
                sleep_dt = float(model.opt.timestep) / speed

                obs = get_obs(data)
                if net is None:
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
                else:
                    action = predict(net, params, obs)

                ctrl = 0.5 * (action + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
                data.ctrl[:] = ctrl

                for _ in range(PHYSICS_STEPS_PER_ACTION):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)

                if data.qpos[3] < 0:
                    flipped = True
                    break
                step += 1

            if not viewer.is_running():
                break

            ep += 1
            ckpt_name = new_ckpt.name if new_ckpt else "?"
            status = "FLIPPED" if flipped else ("TIMEOUT" if step >= MAX_STEPS else "DONE")
            torso_z = float(data.qpos[2])
            print(f"Ep {ep:3d}: {status:<8}  torso={torso_z:.3f}m  steps={step}  [{ckpt_name}]")

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("\n[watch_mjx] closed.")


if __name__ == "__main__":
    main()
