#!/usr/bin/env python3
"""Headless single-episode CPU playback of the latest trained policy.

Used by the dashboard's "Run Trial" button so you see one robot trying
once, at real-time pace, without the trainer's resets and parallel envs.

Writes the live qpos to runs/.trial_qpos.npy at every action step.
Dashboard's viewer reads that file. Exits when the episode ends
(robot flips, reaches the tree, or hits max steps).

Usage:
    python scripts/play_trial.py --task walk [--max-steps 600] [--realtime 1.0]
"""

# CPU-only JAX so we don't fight the trainer for the GPU.
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import pathlib
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "scene.xml"
OBS_DIM = 88
ACT_DIM = 24
GAIT_PERIOD_STEPS = 25
PHYSICS_STEPS_PER_ACTION = 16


def latest_checkpoint(task: str) -> pathlib.Path | None:
    """Newest .npz under runs/ape_<task>_mjx/ (keepers or rolling)."""
    runs = ROOT / "runs" / f"ape_{task}_mjx"
    if not runs.exists():
        return None
    candidates = list(runs.glob("mjx_*.npz")) + list((runs / "keepers").glob("*.npz")
        if (runs / "keepers").exists() else [])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_predict(npz_path: pathlib.Path):
    """Recreate the actor network and load weights. Returns predict(obs) → action."""
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    class Net(nn.Module):
        @nn.compact
        def __call__(self, obs):
            x = nn.tanh(nn.Dense(256)(obs))
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(128)(x))
            mean = nn.Dense(ACT_DIM)(x)
            log_std = self.param("log_std", nn.initializers.zeros, (ACT_DIM,))
            value = nn.Dense(1)(x).squeeze(-1)
            return mean, log_std, value

    net = Net()
    init_params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))

    data = np.load(str(npz_path))
    keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
    flat = [jnp.asarray(data[k]) for k in keys]
    init_flat, treedef = jax.tree_util.tree_flatten(init_params)
    if any(a.shape != b.shape for a, b in zip(flat, init_flat)):
        raise ValueError(f"shape mismatch in {npz_path.name}")
    params = jax.tree_util.tree_unflatten(treedef, flat)
    apply_jit = jax.jit(lambda obs, p=params: net.apply(p, obs)[0])

    def predict(obs):
        return np.array(jnp.tanh(apply_jit(jnp.asarray(obs, dtype=jnp.float32))))
    return predict


def build_obs(model, data, tree_xy, step_i):
    """88-dim obs matching envs/_shared_obs.py — copied from view.py."""
    rel = tree_xy - data.qpos[0:2].astype(np.float32)
    dist = float(np.linalg.norm(rel))
    proprio = np.concatenate([
        data.qpos[2:3],
        data.qpos[3:7],
        data.qvel[0:3],
        data.qvel[3:6],
        data.qpos[7:7 + ACT_DIM],
        data.qvel[6:6 + ACT_DIM],
        data.qfrc_actuator[6:6 + ACT_DIM],
    ], dtype=np.float32)
    target_slot = np.concatenate([rel, [dist]], dtype=np.float32)
    ph = (2.0 * np.pi / GAIT_PERIOD_STEPS) * step_i
    gait_clock = np.array([np.sin(ph), np.cos(ph)], dtype=np.float32)
    return np.concatenate([proprio, target_slot, gait_clock], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="walk")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="action steps before forced exit (default 600 = ~20s sim)")
    parser.add_argument("--realtime", type=float, default=1.0,
                        help="real-time multiplier; 1.0 = real time, 0.5 = half-speed")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # Reset to model rest pose + small noise so it's not perfectly symmetric.
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.30  # standing height matching trainer's reset
    rng = np.random.default_rng(int(time.time()))
    data.qpos[7:7 + ACT_DIM] += rng.normal(0, 0.02, ACT_DIM)
    mujoco.mj_forward(model, data)

    try:
        gid = model.geom("t0_trunk").id
        tree_xy = np.array(model.geom_pos[gid][:2], dtype=np.float32)
    except KeyError:
        tree_xy = np.array([3.0, 0.0], dtype=np.float32)

    ckpt = latest_checkpoint(args.task)
    if ckpt is None:
        print(f"[trial] no checkpoint for task={args.task}; using random policy", flush=True)
        predict = None
    else:
        print(f"[trial] loading {ckpt.name}", flush=True)
        try:
            predict = build_predict(ckpt)
        except Exception as e:
            print(f"[trial] policy load failed: {e}; falling back to random", flush=True)
            predict = None

    qpos_path = ROOT / "runs" / ".trial_qpos.npy"
    qpos_path.parent.mkdir(parents=True, exist_ok=True)
    action_dt = PHYSICS_STEPS_PER_ACTION * float(model.opt.timestep)  # ~32ms
    sleep_per_action = action_dt / max(0.05, args.realtime)

    print(f"[trial] starting episode (tree at {tree_xy.tolist()}, max {args.max_steps} steps)", flush=True)
    t_start = time.time()
    for step_i in range(args.max_steps):
        if predict is not None:
            obs = build_obs(model, data, tree_xy, step_i)
            action = predict(obs)
        else:
            action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)

        # Map [-1, 1] to actuator ctrlrange.
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = lo + 0.5 * (action + 1.0) * (hi - lo)

        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(model, data)

        np.save(str(qpos_path), np.asarray(data.qpos))

        # Termination: torso flipped (quat w < 0.2 → past 90°) or reached tree.
        if data.qpos[3] < 0.2:
            print(f"[trial] flip at step {step_i}", flush=True)
            break
        dist_to_tree = float(np.linalg.norm(tree_xy - data.qpos[0:2]))
        if dist_to_tree < 0.6:
            print(f"[trial] SUCCESS at step {step_i} — reached tree", flush=True)
            break

        # Pace to real time.
        elapsed = time.time() - t_start
        target_elapsed = (step_i + 1) * sleep_per_action
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)

    # Mark end of trial by removing the qpos file (or leaving the final pose).
    print(f"[trial] done in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
