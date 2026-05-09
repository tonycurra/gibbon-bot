"""
Live viewer for MJX training — mirrors utils/watch.py but loads .npz Flax checkpoints.

Runs the best checkpoint so far on CPU MuJoCo with the viewer open.
Hot-swaps policy when a newer checkpoint appears.

Usage:
    python utils/watch_mjx.py              # watches runs/ape_stand_mjx/
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

from envs.stand_env import StandEnv
from envs.stand_env_mjx import OBS_DIM, ACT_DIM

RUNS = pathlib.Path("runs/ape_stand_mjx")


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
    mean, log_std, _ = net.apply(params, obs_j)
    return np.array(jnp.tanh(mean))


def latest_checkpoint():
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    return ckpts[-1] if ckpts else None


def main():
    env = StandEnv(render_mode="human")

    ckpt = latest_checkpoint()
    if ckpt is None:
        print(f"[watch_mjx] no checkpoint yet in {RUNS} — showing random policy, waiting...")
        net, params = None, None
        last_mtime = 0.0
    else:
        print(f"[watch_mjx] loading {ckpt.name} ...")
        net, params = load_policy(ckpt)
        last_mtime = ckpt.stat().st_mtime

    ep = 0

    def viewer_alive():
        v = getattr(env, "_viewer", None)
        return v is None or v.is_running()

    try:
        while viewer_alive():
            new_ckpt = latest_checkpoint()
            if new_ckpt is not None:
                mtime = new_ckpt.stat().st_mtime
                if mtime != last_mtime:
                    print(f"[watch_mjx] {'first' if net is None else 'new'} checkpoint: {new_ckpt.name}")
                    net, params = load_policy(new_ckpt)
                    last_mtime = mtime

            obs, _ = env.reset(seed=ep, options={"no_settle": True})
            done = False
            total_reward = 0.0
            step = 0

            while not done and viewer_alive():
                if net is None:
                    action = env.action_space.sample()
                else:
                    action = predict(net, params, obs[:OBS_DIM])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated

            if not viewer_alive():
                break

            ep += 1
            ckpt_name = new_ckpt.name if new_ckpt else "?"
            status = "FLIPPED" if info.get("upside_down") else ("TIMEOUT" if step >= 2000 else "DONE")
            print(f"Ep {ep:3d}: {status:<8}  torso={info['torso_z']:.3f}m  "
                  f"reward={total_reward:.1f}  steps={step}  [{ckpt_name}]")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\n[watch_mjx] closed.")


if __name__ == "__main__":
    main()
