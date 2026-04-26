"""
Live viewer for MJX training.

Polls runs/ape_stand_mjx/ for the newest checkpoint (.npz),
runs the policy on CPU via the normal StandEnv (so you get the viewer),
hot-swaps to the latest checkpoint when training saves a new one.

Usage:
    python tools/watch_mjx.py
"""

import pathlib
import sys

# Add the src directory to Python path
src_path = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0, src_path)

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from gibbonbot.config.train_mjx import ACT_DIM, OBS_DIM, ActorCritic
from gibbonbot.envs.stand_env import StandEnv

RUNS = pathlib.Path("runs/ape_stand_mjx")
POLL_INTERVAL = 5  # seconds between checkpoint checks


def latest_checkpoint():
    ckpts = sorted(RUNS.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_params(path):
    """Restore Flax params from the flat .npz saved by train_mjx._save()."""
    data = np.load(str(path))
    flat = [
        jnp.array(data[k])
        for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))
    ]
    # round-trip through init to get the pytree structure, then fill
    net = ActorCritic()
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, flat)


def main():
    print(f"[watch_mjx] waiting for first checkpoint in {RUNS} ...")
    while True:
        ckpt = latest_checkpoint()
        if ckpt:
            break
        time.sleep(POLL_INTERVAL)

    print(f"[watch_mjx] loading {ckpt.name} ...")
    params = load_params(ckpt)
    last_mtime = ckpt.stat().st_mtime
    net = ActorCritic()

    # JIT the forward pass (CPU is fine — network is tiny)
    @jax.jit
    def act(params, obs):
        mean, log_std, _ = net.apply(params, obs)
        return jnp.tanh(mean)  # deterministic: use mean, not sample

    env = StandEnv(render_mode="human")
    ep = 0

    def alive():
        v = getattr(env, "_viewer", None)
        return v is None or v.is_running()

    try:
        while alive():
            new_ckpt = latest_checkpoint()
            if new_ckpt is not None and new_ckpt.stat().st_mtime != last_mtime:
                print(f"\n[watch_mjx] new checkpoint: {new_ckpt.name} — swapping")
                params = load_params(new_ckpt)
                last_mtime = new_ckpt.stat().st_mtime

            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0.0
            step = 0
            # CPU env has 83-dim obs; MJX checkpoint expects 86 (includes 3D tree pos).
            # Pad with a fixed dummy tree position — policy ignores it during stand task.
            _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)
            while not done and alive():
                obs_padded = np.concatenate([obs, _TREE_PAD])
                action = np.array(act(params, jnp.array(obs_padded)))
                obs, reward, terminated, truncated, info = env.step(action)
                # obs is 83-dim; padding applied at top of loop before next act()
                total_reward += reward
                step += 1
                done = terminated or truncated

            if not alive():
                break
            ep += 1
            status = "FALLEN" if info["fallen"] else "ALIVE"
            print(
                f"Ep {ep:3d}: {status:<6}  height={info['above_ground']:.3f}m  "
                f"h_rew={info['height_reward']:.3f}  reward={total_reward:.1f}  steps={step}  "
                f"[{new_ckpt.name if new_ckpt else ckpt.name}]"
            )

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\n[watch_mjx] closed.")


if __name__ == "__main__":
    main()
