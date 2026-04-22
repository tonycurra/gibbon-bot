"""
Gymnasium environment: teach the ape robot to reach the tree.

Observation (58 dims):
  [0]      torso height (z)
  [1:5]    torso orientation quaternion (w x y z)
  [5:8]    torso linear velocity
  [8:11]   torso angular velocity
  [11:35]  joint positions  (24)
  [35:59]  joint velocities (24)
  [59:62]  unit vector from torso to tree (world frame)
  [62]     distance torso → tree (metres)
  Total: 63

Action (24 dims):
  Target joint positions, normalised to [-1, 1].
  Mapped back to each joint's [lo, hi] range before writing to data.ctrl.

Reward per step:
  + progress   reduction in torso-to-tree distance vs previous step
  + alive      0.1 per step (encourages not dying early)
  - energy     1e-4 * sum(ctrl²)  (discourages flailing)
  - fall       -10 on termination if torso too low
  + success    +50 on reaching tree (torso within 1.5 m)
"""

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import pathlib
import terrain as terrain_mod

SCENE       = pathlib.Path(__file__).parent / "scene.xml"
TREE_POS    = np.array([5.0, 0.0, 0.0])
SUCCESS_DIST = 1.5    # metres — torso this close = reached tree
MIN_HEIGHT   = 0.15   # metres — torso below this = fallen
MAX_STEPS    = 2000
PHYSICS_STEPS_PER_ACTION = 5   # simulate 5 MuJoCo steps per RL action


class ApeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        terrain_mod.apply(self.model)
        self.data  = mujoco.MjData(self.model)

        # joint position ranges for action rescaling
        # skip joint 0 (freejoint); joints 1-24 are arm joints
        self._jnt_lo = np.array([self.model.jnt_range[i][0]
                                  for i in range(1, self.model.njnt)])
        self._jnt_hi = np.array([self.model.jnt_range[i][1]
                                  for i in range(1, self.model.njnt)])

        n_obs = 63
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        self._prev_dist = None
        self._step_count = 0

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer = None

    # ── core API ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # small random perturbation so the agent doesn't overfit to one start
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.data.qpos[7:] += rng.uniform(-0.05, 0.05, size=24)   # arm joints
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist  = self._torso_tree_dist()
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        # action ∈ [-1,1]  →  ctrl ∈ [jnt_lo, jnt_hi]
        ctrl = 0.5 * (action + 1.0) * (self._jnt_hi - self._jnt_lo) + self._jnt_lo
        self.data.ctrl[:] = ctrl

        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1
        obs = self._get_obs()

        dist       = self._torso_tree_dist()
        progress   = self._prev_dist - dist          # positive = moved closer
        self._prev_dist = dist

        torso_z    = float(self.data.qpos[2])
        fallen     = torso_z < MIN_HEIGHT
        success    = dist < SUCCESS_DIST

        jvel       = self.data.qvel[6:]
        reward = (
              progress * 2.0
            + 0.1                                    # alive bonus
            - 5e-4 * float(np.sum(ctrl ** 2))        # energy penalty
            - 1e-3 * float(np.sum(jvel ** 2))        # velocity penalty (discourages flailing)
        )
        if fallen:
            reward -= 10.0
        if success:
            reward += 50.0

        terminated = fallen or success
        truncated  = self._step_count >= MAX_STEPS

        info = {"dist_to_tree": dist, "torso_z": torso_z,
                "success": success, "fallen": fallen}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        torso_z    = qpos[2:3]                   # (1,)
        torso_quat = qpos[3:7]                   # (4,) w x y z
        torso_lvel = qvel[0:3]                   # linear vel
        torso_avel = qvel[3:6]                   # angular vel
        jpos       = qpos[7:]                    # (24,)
        jvel       = qvel[6:]                    # (24,)

        torso_xyz  = self.data.qpos[0:3]
        to_tree    = TREE_POS - torso_xyz
        dist       = float(np.linalg.norm(to_tree))
        direction  = to_tree / (dist + 1e-8)     # unit vector (3,)

        obs = np.concatenate([
            torso_z, torso_quat, torso_lvel, torso_avel,
            jpos, jvel,
            direction, [dist],
        ]).astype(np.float32)
        return obs

    def _torso_tree_dist(self):
        return float(np.linalg.norm(TREE_POS - self.data.qpos[0:3]))
