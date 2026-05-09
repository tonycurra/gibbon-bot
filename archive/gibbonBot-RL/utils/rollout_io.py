"""Load saved rollout dumps from runs/<task>/rollouts/.

Each rollout file is one PPO update's collected experience for a sub-sample
of envs. Shapes are (N_STEPS, n_envs_keep, ...) for obs/act, (N_STEPS, n_envs_keep)
for scalars. Use these for offline analysis, behavior cloning, or replay.
"""
import pathlib
import numpy as np


def list_rollouts(runs_dir):
    return sorted(pathlib.Path(runs_dir).glob("rollouts/rollout_*.npz"))


def load_rollout(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def iter_transitions(path):
    """Yield (obs_t, act_t, rew_t, done_t, obs_tp1) tuples flattened across envs."""
    r = load_rollout(path)
    obs, act, rew, done = r["obs"], r["act"], r["rew"], r["done"]
    T, E = obs.shape[:2]
    for t in range(T - 1):
        yield obs[t], act[t], rew[t], done[t], obs[t + 1]


def stack_all(runs_dir, key="obs"):
    """Concat one field across all rollout files. Use for quick stats."""
    files = list_rollouts(runs_dir)
    return np.concatenate([load_rollout(f)[key] for f in files])
