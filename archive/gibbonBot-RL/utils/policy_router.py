import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class PolicyCandidate:
    checkpoint_path: str
    task: str
    step_count: int
    mean_reward: float
    stability_score: float
    evaluation_reward: Optional[float] = None
    evaluation_steps: Optional[int] = None
    compatibility_score: float = 1.0
    context_similarity: float = 0.0

    @property
    def score(self) -> float:
        reward_score = self.mean_reward or 0.0
        eval_score = self.evaluation_reward or 0.0
        stability = self.stability_score
        compat = self.compatibility_score
        similarity = self.context_similarity
        if self.evaluation_reward is not None:
            return eval_score * 0.6 + reward_score * 0.3 + stability * 0.1
        else:
            return reward_score * 0.7 + stability * 0.2 + similarity * 0.1


@dataclass
class SelectedPolicy:
    checkpoint_path: str
    params: Any
    task: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyRouter:
    def __init__(self, db_path="runs/ape_policy_db.sqlite", obs_dim=88, act_dim=24):
        self.db_path = pathlib.Path(db_path)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._db = None
        self._net = None
        self._init_params = None

    @property
    def db(self):
        if self._db is None:
            from brain.policy_database import PolicyDatabase

            self._db = PolicyDatabase(self.db_path)
        return self._db

    @property
    def net(self):
        if self._net is None:
            self._init_network()
        return self._net

    @property
    def init_params(self):
        if self._init_params is None:
            self._init_network()
        return self._init_params

    def _init_network(self):
        import flax.linen as nn

        act_dim = self.act_dim
        obs_dim = self.obs_dim

        class ActorCritic(nn.Module):
            @nn.compact
            def __call__(self, obs):
                x = nn.tanh(nn.Dense(256)(obs))
                x = nn.tanh(nn.Dense(256)(x))
                x = nn.tanh(nn.Dense(128)(x))
                mean = nn.Dense(act_dim)(x)
                log_std = self.param("log_std", nn.initializers.zeros, (act_dim,))
                value = nn.Dense(1)(x).squeeze(-1)
                return mean, log_std, value

        self._net = ActorCritic()
        self._init_params = self._net.init(
            jax.random.PRNGKey(0), jnp.zeros((1, obs_dim))
        )

    def load_policy_params(self, checkpoint_path):
        checkpoint_path = pathlib.Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        data = np.load(str(checkpoint_path))
        keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
        flat = [jnp.asarray(data[k]) for k in keys]
        init_flat, treedef = jax.tree_util.tree_flatten(self.init_params)
        if any(a.shape != b.shape for a, b in zip(flat, init_flat)):
            raise ValueError(f"Shape mismatch in {checkpoint_path.name}")
        return jax.tree_util.tree_unflatten(treedef, flat)

    def get_candidate_policies(
        self, task=None, scene_path=None, limit=10, min_reward=None
    ):
        candidates = []
        if scene_path:
            scene_hash = self.db.compute_scene_hash(scene_path)
            for record in self.db.query_by_scene(scene_hash=scene_hash, limit=limit):
                if task and record.task != task:
                    continue
                candidates.append(
                    PolicyCandidate(
                        checkpoint_path=record.checkpoint_path,
                        task=record.task,
                        step_count=record.step_count,
                        mean_reward=record.mean_reward or 0.0,
                        stability_score=record.stability_score or 0.0,
                    )
                )
        if len(candidates) < limit and task:
            for record in self.db.query_by_task(task, limit=limit * 2):
                existing = {c.checkpoint_path for c in candidates}
                if record.checkpoint_path not in existing:
                    candidates.append(
                        PolicyCandidate(
                            checkpoint_path=record.checkpoint_path,
                            task=record.task,
                            step_count=record.step_count,
                            mean_reward=record.mean_reward or 0.0,
                            stability_score=record.stability_score or 0.0,
                        )
                    )
                    if len(candidates) >= limit:
                        break
        if len(candidates) < limit and scene_path:
            scene_hash = self.db.compute_scene_hash(scene_path)
            for record in self.db.query_similar_scenes(
                scene_hash=scene_hash, limit=limit * 2
            ):
                existing = {c.checkpoint_path for c in candidates}
                if record.checkpoint_path not in existing:
                    candidates.append(
                        PolicyCandidate(
                            checkpoint_path=record.checkpoint_path,
                            task=record.task,
                            step_count=record.step_count,
                            mean_reward=record.mean_reward or 0.0,
                            stability_score=record.stability_score or 0.0,
                            context_similarity=0.5,
                        )
                    )
                    if len(candidates) >= limit:
                        break
        if min_reward is not None:
            candidates = [c for c in candidates if c.mean_reward >= min_reward]
        candidates.sort(key=lambda c: -c.score)
        return candidates[:limit]

    def evaluate_policy(
        self, checkpoint_path, scene_path="data/scene.xml", num_steps=200
    ):
        import mujoco

        try:
            params = self.load_policy_params(checkpoint_path)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            torso_idx = next(
                (i for i in range(model.nbody) if model.body(i).name == "torso"), None
            )
            mujoco.mj_resetData(model, data)
            total_reward = 0.0
            for step in range(num_steps):
                obs = np.concatenate([data.qpos, data.qvel])
                if len(obs) < self.obs_dim:
                    obs = np.pad(obs, (0, self.obs_dim - len(obs)))
                elif len(obs) > self.obs_dim:
                    obs = obs[: self.obs_dim]
                mean, _, _ = self.net.apply(params, jnp.array(obs[np.newaxis, :]))
                action = np.array(jnp.tanh(mean).squeeze(0))
                data.ctrl[:] = action
                mujoco.mj_step(model, data)
                if torso_idx is not None:
                    torso_z = data.body(torso_idx).xpos[2]
                    reward = max(0, torso_z - 0.3)
                    if torso_z < 0.1:
                        total_reward += reward
                        return total_reward, step
                else:
                    reward = 0.1
                total_reward += reward
            return total_reward, num_steps
        except Exception as e:
            return -float("inf"), 0

    def select_best_policy(
        self, task, scene_path="data/scene.xml", evaluate=False, num_steps=200, limit=5
    ):
        candidates = self.get_candidate_policies(
            task=task, scene_path=scene_path, limit=limit
        )
        if not candidates:
            return None
        if evaluate:
            for c in candidates:
                eval_reward, eval_steps_done = self.evaluate_policy(
                    c.checkpoint_path, scene_path, num_steps
                )
                c.evaluation_reward = eval_reward
                c.evaluation_steps = eval_steps_done
                c.compatibility_score = (
                    eval_steps_done / num_steps if eval_steps_done > 0 else 0.0
                )
            candidates.sort(key=lambda c: -c.score)
        best = candidates[0]
        try:
            params = self.load_policy_params(best.checkpoint_path)
        except Exception as e:
            print(f"Failed to load {best.checkpoint_path}: {e}")
            return None
        return SelectedPolicy(
            checkpoint_path=best.checkpoint_path,
            params=params,
            task=best.task,
            score=best.score,
            metadata={
                "mean_reward": best.mean_reward,
                "step_count": best.step_count,
                "stability": best.stability_score,
                "evaluation_reward": best.evaluation_reward,
            },
        )

    def import_from_runs(self, runs_dir="runs", task=None):
        from brain.policy_database import import_checkpoints_from_runs

        return import_checkpoints_from_runs(self.db_path, runs_dir, task)

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
