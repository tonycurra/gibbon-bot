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

Reward per step (optimized for fast learning):
  + progress       3.0 * reduction in distance (50% more weight)
  + proximity      2.0 * exp(-dist/2.0) (exponential shaping)
  + alignment      1.5 * dot(torso_forward, tree_direction) (face the tree)
  + alive          0.1 per step
  - energy         3e-4 * sum(ctrl²) (reduced penalty)
  - velocity       5e-4 * sum(jvel²) (reduced penalty)
  + success        50.0 * (1 - dist/SUCCESS_DIST) (early success bonus)
  - fall           -10.0 on termination

Curriculum Learning:
  Stage 0 (0-2M steps):   tree at 2.0m, success radius 1.0m
  Stage 1 (2M-5M steps): tree at 3.5m, success radius 0.8m
  Stage 2 (5M+ steps):  tree at 5.0m, success radius 0.5m
"""

import pathlib

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

SCENE = pathlib.Path(__file__).parent.parent / "scene.xml"
BASE_TREE_POS = np.array([5.0, 0.0, 0.0])  # Base tree position
BASE_SUCCESS_DIST = 1.5  # metres — torso this close = reached tree
MIN_HEIGHT = 0.10  # metres — torso below this = fallen (lower to prevent false falls)
MAX_STEPS = 2000
PHYSICS_STEPS_PER_ACTION = (
    15  # simulate 15 MuJoCo steps per RL action (even slower, smoother)
)

# Curriculum learning parameters
CURRICULUM_STAGES = [
    {
        "tree_dist": 2.0,
        "success_dist": 1.0,
        "min_steps": 0,
    },  # Easy: close tree, large success radius
    {
        "tree_dist": 3.5,
        "success_dist": 0.8,
        "min_steps": 2_000_000,
    },  # Medium: farther tree, smaller radius
    {
        "tree_dist": 5.0,
        "success_dist": 0.5,
        "min_steps": 5_000_000,
    },  # Hard: full distance, tight radius
]


class ApeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)

        # joint position ranges for action rescaling
        # skip joint 0 (freejoint); joints 1-24 are arm joints
        self._jnt_lo = np.array(
            [self.model.jnt_range[i][0] for i in range(1, self.model.njnt)]
        )
        self._jnt_hi = np.array(
            [self.model.jnt_range[i][1] for i in range(1, self.model.njnt)]
        )

        n_obs = 63
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        self._prev_dist = None
        self._step_count = 0
        self._curriculum_stage = 0
        self._prev_action = None  # For smoothness calculation
        self._update_curriculum_params()

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
        self.data.qpos[7:] += rng.uniform(-0.05, 0.05, size=24)  # arm joints
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._torso_tree_dist()
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        # action ∈ [-1,1]  →  ctrl ∈ [jnt_lo, jnt_hi]
        ctrl = 0.5 * (action + 1.0) * (self._jnt_hi - self._jnt_lo) + self._jnt_lo

        # Apply adaptive stiffness based on joint positions
        ctrl_with_adaptive_physics = self._apply_adaptive_physics(ctrl)
        self.data.ctrl[:] = ctrl_with_adaptive_physics

        # Track action for smoothness calculation
        self._prev_action = action.copy()

        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1

        # Update curriculum stage based on total steps
        if self._curriculum_stage < len(CURRICULUM_STAGES) - 1:
            next_stage = CURRICULUM_STAGES[self._curriculum_stage + 1]
            if self._step_count >= next_stage["min_steps"]:
                self._curriculum_stage += 1
                self._update_curriculum_params()
                print(
                    f"Curriculum advanced to stage {self._curriculum_stage}: "
                    f"tree_dist={self.TREE_POS[0]:.1f}m, success_dist={self.SUCCESS_DIST:.1f}m"
                )

        obs = self._get_obs()

        dist = self._torso_tree_dist()
        progress = self._prev_dist - dist  # positive = moved closer
        self._prev_dist = dist

        torso_z = float(self.data.qpos[2])
        fallen = torso_z < MIN_HEIGHT
        success = dist < self.SUCCESS_DIST

        jvel = self.data.qvel[6:]

        # Get torso forward direction (x-axis in torso frame)
        torso_quat = self.data.qpos[3:7]
        torso_forward = self._get_torso_forward(torso_quat)
        tree_direction = (self.TREE_POS - self.data.qpos[0:3]) / (dist + 1e-8)

        # Calculate smoothness penalty (reduce jerkiness)
        smoothness_penalty = 0.0
        if self._prev_action is not None:
            action_diff = action - self._prev_action
            smoothness_penalty = -0.001 * float(np.sum(action_diff**2))

        # Calculate alignment between torso forward and tree direction
        alignment = float(np.dot(torso_forward, tree_direction))

        # Optimized reward system with dense shaping (slower, smoother movements)
        proximity_reward = np.exp(-dist / 2.0)  # Exponential decay: stronger near tree
        progress_reward = (
            progress * 1.0
        )  # Reduced weight on progress for smoother movement
        alignment_reward = max(0, alignment) * 1.0  # Reduced weight on alignment

        # Penalize fast vertical movement to encourage stability
        torso_z_vel = abs(self.data.qvel[2])  # Vertical velocity of torso
        stability_penalty = -0.01 * torso_z_vel  # Penalize up-down oscillation

        reward = (
            progress_reward
            + proximity_reward * 1.0  # Get closer at natural pace
            + alignment_reward  # Face the right direction
            + smoothness_penalty  # Smooth movement
            + stability_penalty  # Encourage stable height
            + 0.1  # alive bonus
            - 3e-4 * float(np.sum(ctrl**2))  # Reduced energy penalty
            - 5e-4 * float(np.sum(jvel**2))  # Reduced velocity penalty
        )

        # Success bonus - make it proportional to how early we succeed
        if success:
            reward += 50.0 * (1.0 - min(dist / SUCCESS_DIST, 1.0))

        if fallen:
            reward -= 10.0

        terminated = fallen or success
        truncated = self._step_count >= MAX_STEPS

        info = {
            "dist_to_tree": dist,
            "torso_z": torso_z,
            "success": success,
            "fallen": fallen,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_adaptive_physics(self, ctrl):
        """Apply adaptive joint stiffness based on position for more natural motion."""
        # Normalize control values to [0,1] range for each joint
        normalized_ctrl = (ctrl - self._jnt_lo) / (self._jnt_hi - self._jnt_lo + 1e-8)

        # Adaptive stiffness: higher near center, lower at extremes
        # This creates more stable motion near neutral positions
        # and allows more flexibility at joint limits
        adaptive_kp = []
        for norm_pos in normalized_ctrl:
            # Calculate stiffness factor (range: 0.6-1.2)
            # Center positions (norm_pos ~ 0.5) get highest stiffness
            stiffness_factor = 0.6 + 0.6 * np.exp(-4 * (norm_pos - 0.5) ** 2)
            adaptive_kp.append(stiffness_factor)

        # Apply adaptive stiffness by scaling the control signal
        # This simulates variable joint impedance
        return ctrl * np.array(adaptive_kp)

    def _update_curriculum_params(self):
        """Update tree position and success distance based on curriculum stage."""
        stage = self._curriculum_stage
        if stage >= len(CURRICULUM_STAGES):
            stage = len(CURRICULUM_STAGES) - 1

        curriculum = CURRICULUM_STAGES[stage]
        # Calculate tree position based on distance along x-axis
        self.TREE_POS = BASE_TREE_POS * (curriculum["tree_dist"] / BASE_TREE_POS[0])
        self.SUCCESS_DIST = curriculum["success_dist"]

    def _get_torso_forward(self, quat):
        """Get torso forward direction (x-axis) from quaternion."""
        # Quaternions are [w, x, y, z]
        w, x, y, z = quat
        # Forward vector in torso frame is (1, 0, 0)
        # Rotate by quaternion: v' = q * v * q*
        forward_x = 1 - 2 * y * y - 2 * z * z
        forward_y = 2 * x * y + 2 * w * z
        forward_z = 2 * x * z - 2 * w * y
        forward = np.array([forward_x, forward_y, forward_z])
        return forward / (np.linalg.norm(forward) + 1e-8)

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        torso_z = qpos[2:3]  # (1,)
        torso_quat = qpos[3:7]  # (4,) w x y z
        torso_lvel = qvel[0:3]  # linear vel
        torso_avel = qvel[3:6]  # angular vel
        jpos = qpos[7:]  # (24,)
        jvel = qvel[6:]  # (24,)

        torso_xyz = self.data.qpos[0:3]
        to_tree = self.TREE_POS - torso_xyz
        dist = float(np.linalg.norm(to_tree))
        direction = to_tree / (dist + 1e-8)  # unit vector (3,)

        obs = np.concatenate(
            [
                torso_z,
                torso_quat,
                torso_lvel,
                torso_avel,
                jpos,
                jvel,
                direction,
                [dist],
            ]
        ).astype(np.float32)
        return obs

    def _torso_tree_dist(self):
        return float(np.linalg.norm(self.TREE_POS - self.data.qpos[0:3]))
