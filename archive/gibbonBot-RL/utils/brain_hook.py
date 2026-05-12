"""
BrainHook — bridge between the JIT'd PPO trainer and the ApeBrain modules.

Architecture
------------
The PPO trainer in `train_mjx.py` runs fully JIT'd rollouts; it cannot call
into stateful Python brain code from inside the jitted hot path. This hook
sits *beside* the trainer: it samples a small subset of each rollout, copies
those transitions to host memory, and feeds them to the hippocampus
(persistent episodic store) and basal ganglia (consolidation replay).

Think of it as: PPO does procedural learning (slow weight updates from
millions of transitions); ApeBrain does episodic learning on top
(remembers individual events, replays during "sleep"). They share no
gradients — the trainer drives, the brain observes and consolidates.

Usage
-----
    hook = BrainHook(persist_dir="data/brain", obs_dim=88, act_dim=24,
                    samples_per_rollout=32, sleep_every_updates=20)
    # inside training loop, after rollout collection:
    hook.record_rollout(obs_t, act_t, rew_t, next_obs_t)
    if hook.maybe_sleep(update_count):
        ...  # consolidation happened
    metrics = hook.metrics()  # log via mlflow
    hook.close()
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import numpy as np


class BrainHook:
    def __init__(
        self,
        persist_dir: str,
        obs_dim: int,
        act_dim: int,
        samples_per_rollout: int = 32,
        sleep_every_updates: int = 20,
        sleep_replays: int = 64,
        sleep_lr_scale: float = 2.0,
        threat_threshold: float = -1.0,
    ):
        from brain.ape_brain import ApeBrain, ApeBrainConfig
        from brain.bus import EVENT_REPLAY_TICK, EVENT_THREAT

        self.brain = ApeBrain(ApeBrainConfig(
            obs_dim=obs_dim, act_dim=act_dim, training=True, persist_dir=persist_dir,
        ))
        self.samples_per_rollout = samples_per_rollout
        self.sleep_every_updates = sleep_every_updates
        self.sleep_replays = sleep_replays
        self.sleep_lr_scale = sleep_lr_scale
        self.threat_threshold = threat_threshold
        self._rng = np.random.default_rng(0)
        self._last_sleep_replays = 0
        self._total_episodes_stored = 0
        self._total_replays = 0
        self._threat_events = 0
        self._novelty_events = 0

        self.brain.bus.subscribe(
            EVENT_THREAT, lambda e, p: self._on_threat(p)
        )

    def _on_threat(self, payload: Dict) -> None:
        self._threat_events += 1

    def record_rollout(
        self,
        obs_t,         # (T, N_ENVS, obs_dim) jax array
        act_t,         # (T, N_ENVS, act_dim)
        rew_t,         # (T, N_ENVS)
        next_obs_t,    # (T, N_ENVS, obs_dim) — typically obs_t shifted by 1, or rebuilt
    ) -> int:
        """Sample `samples_per_rollout` random (t, env) transitions and store them."""
        T = int(obs_t.shape[0])
        N = int(obs_t.shape[1])
        if T == 0 or N == 0:
            return 0
        k = min(self.samples_per_rollout, T * N)

        # Sample on host, then index host arrays.
        ts = self._rng.integers(0, T, size=k)
        es = self._rng.integers(0, N, size=k)

        # Single device->host transfer of the rollout. Cheap relative to the
        # rollout itself, and only the sampled slices are persisted.
        obs_np = np.asarray(jax.device_get(obs_t))
        act_np = np.asarray(jax.device_get(act_t))
        rew_np = np.asarray(jax.device_get(rew_t))
        nxt_np = np.asarray(jax.device_get(next_obs_t))

        hippo = self.brain.modules["hippocampus"]
        stored = 0
        for t, e in zip(ts, es):
            obs = obs_np[t, e]
            act = act_np[t, e]
            nxt = nxt_np[t, e]
            r = float(rew_np[t, e])
            hippo.store_memory(obs, act, nxt, reward=r, punishment=max(-r, 0.0))
            if r <= self.threat_threshold:
                self.brain.emit_threat(r, threshold=self.threat_threshold)
            stored += 1

        self._total_episodes_stored += stored
        return stored

    def maybe_sleep(self, update_count: int) -> bool:
        """If due, run a consolidation pass. Returns True if sleep ran."""
        if self.sleep_every_updates <= 0:
            return False
        if update_count == 0 or update_count % self.sleep_every_updates != 0:
            return False
        n = self.brain.sleep(
            n_replays=self.sleep_replays, lr_scale=self.sleep_lr_scale,
        )
        self._last_sleep_replays = n
        self._total_replays += n
        return n > 0

    def metrics(self) -> Dict[str, float]:
        hippo = self.brain.modules["hippocampus"]
        store_size = len(hippo.store) if hippo.store is not None else 0
        return {
            "brain_episodes_stored": float(self._total_episodes_stored),
            "brain_store_size": float(store_size),
            "brain_last_replays": float(self._last_sleep_replays),
            "brain_total_replays": float(self._total_replays),
            "brain_threat_events": float(self._threat_events),
        }

    def close(self) -> None:
        hippo = self.brain.modules["hippocampus"]
        if hippo.store is not None:
            hippo.store.close()
        if self.brain.semantic_map is not None:
            self.brain.semantic_map.close()


def test_brain_hook() -> None:
    import tempfile
    import jax.numpy as jnp
    import jax.random as jr

    tmp = tempfile.mkdtemp()
    hook = BrainHook(
        persist_dir=tmp, obs_dim=8, act_dim=4,
        samples_per_rollout=10, sleep_every_updates=2, sleep_replays=5,
    )
    T, N = 4, 6
    key = jr.PRNGKey(0)
    k1, k2, k3, k4 = jr.split(key, 4)
    obs = jr.normal(k1, (T, N, 8))
    act = jr.normal(k2, (T, N, 4))
    rew = jr.normal(k3, (T, N))
    nxt = jr.normal(k4, (T, N, 8))

    stored = hook.record_rollout(obs, act, rew, nxt)
    assert stored == 10, f"stored={stored}"
    assert not hook.maybe_sleep(1)   # not due
    slept = hook.maybe_sleep(2)
    assert slept
    m = hook.metrics()
    assert m["brain_episodes_stored"] == 10
    assert m["brain_store_size"] == 10
    assert m["brain_last_replays"] > 0
    hook.close()
    print("✅ BrainHook tests passed")


if __name__ == "__main__":
    test_brain_hook()
