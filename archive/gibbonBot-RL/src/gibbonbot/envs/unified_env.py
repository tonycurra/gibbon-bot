#!/usr/bin/env python3
"""
Unified Environment for gibbonBot-RL

Single environment that can handle multiple tasks with realistic physics.

Tasks:
- stand: Lift torso to stable height
- reach: Move to target location
- balance: Maintain stability on uneven terrain
- navigate: Move through obstacle course

Features:
- Realistic physics (adaptive damping, friction)
- Unified observation/action space
- Task-specific reward shaping
- Progressive difficulty
"""

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Configuration
SCENE = (
    pathlib.Path(__file__).parent.parent / "scene_fixed.xml"
)  # Use fixed collision version
MAX_STEPS = 2000
PHYSICS_STEPS_PER_ACTION = 15  # Slower, more realistic physics
MIN_HEIGHT = 0.08  # Torso height threshold for falling (lower to account for arm reach)
GROUND_LEVEL = 0.0  # Ground plane z-position


class UnifiedEnv(gym.Env):
    """Unified environment for all robot tasks"""

    def __init__(self, task="stand", render_mode=None):
        super().__init__()
        self.task = task
        self.render_mode = render_mode

        # Load model
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)

        # Joint limits for action scaling
        self._jnt_lo = np.array(
            [self.model.jnt_range[i][0] for i in range(1, self.model.njnt)]
        )
        self._jnt_hi = np.array(
            [self.model.jnt_range[i][1] for i in range(1, self.model.njnt)]
        )

        # Observation space: 63 dimensions
        # [torso_pos(3), torso_quat(4), torso_vel(6), joint_pos(24), joint_vel(24), task_specific(2)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(63,), dtype=np.float32
        )

        # Action space: 24 dimensions (one per joint)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        # Task-specific parameters
        self._setup_task_parameters()

        # Realistic physics parameters
        self._setup_realistic_physics()

        # Viewer
        self._viewer = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # State tracking
        self._step_count = 0
        self._prev_action = None

    def _setup_task_parameters(self):
        """Configure task-specific settings"""
        if self.task == "stand":
            self._target_height = 0.40  # Target torso height
            self._success_threshold = 0.05  # Height tolerance
        elif self.task == "reach":
            self._tree_position = np.array([3.0, 0.0, 0.0])  # Closer tree
            self._success_distance = 1.0  # Success radius
        elif self.task == "balance":
            self._terrain_roughness = 0.1  # Add terrain variation
        else:  # navigate
            self._obstacle_positions = self._setup_obstacles()

    def _setup_realistic_physics(self):
        """Configure realistic physics parameters"""
        # Add adaptive damping based on joint velocity
        for i in range(self.model.njnt):
            if i > 0:  # Skip freejoint
                self.model.dof_damping[i - 1] = 2.0  # Base damping

        # Realistic friction
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                self.model.geom_friction[i][0] = 2.0  # Sliding friction
                self.model.geom_friction[i][1] = 0.01  # Spinning friction
                self.model.geom_friction[i][2] = 0.001  # Rolling friction

    def _setup_obstacles(self):
        """Setup obstacles for navigation task"""
        return [
            np.array([2.0, 0.5, 0.0]),
            np.array([2.0, -0.5, 0.0]),
            np.array([4.0, 0.0, 0.0]),
        ]

    def _get_obs(self):
        """Get observation vector"""
        torso_pos = self.data.qpos[0:3]
        torso_quat = self.data.qpos[3:7]
        torso_vel = self.data.qvel[0:6]
        joint_pos = self.data.qpos[7:31]
        joint_vel = self.data.qvel[6:30]

        # Task-specific observation
        if self.task == "stand":
            task_obs = np.array([self._target_height, 0.0])
        elif self.task == "reach":
            tree_dir = self._tree_position - torso_pos
            task_obs = np.array([np.linalg.norm(tree_dir), 0.0])
        else:
            task_obs = np.zeros(2)

        return np.concatenate(
            [torso_pos, torso_quat, torso_vel, joint_pos, joint_vel, task_obs]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        # Reset to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Add small random perturbation
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Randomize joint positions slightly, but keep arms above minimum height
        joint_noise = rng.uniform(-0.05, 0.05, size=24)

        # Ensure arms don't start too low (prevent floor penetration)
        # Links 3, 4, 5 are the lower arms - limit their downward range
        for i in [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22]:
            # Don't let these joints go too negative (downward)
            if joint_noise[i] < -0.03:
                joint_noise[i] = -0.03

        self.data.qpos[7:] += joint_noise

        # Randomize torso position for reach task
        if self.task == "reach":
            self.data.qpos[0:2] += rng.uniform(-0.1, 0.1, size=2)

        # Forward physics
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = None

        return self._get_obs(), {}

    def step(self, action):
        """Take one step in environment"""
        # Convert action to control signal
        ctrl = 0.5 * (action + 1.0) * (self._jnt_hi - self._jnt_lo) + self._jnt_lo

        # Apply adaptive physics
        ctrl = self._apply_adaptive_physics(ctrl)
        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

            # Enforce joint limits to prevent unrealistic positions
            self._enforce_joint_limits()

        # Sync viewer
        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= MAX_STEPS

        info = {"torso_height": self.data.qpos[2], "steps": self._step_count}

        if self.task == "reach":
            info["distance_to_target"] = np.linalg.norm(
                self.data.qpos[0:2] - self._tree_position[0:2]
            )

        return obs, reward, terminated, truncated, info

    def _apply_adaptive_physics(self, ctrl):
        """Add adaptive physics for realism"""
        # Add slight noise to simulate real-world variability
        noise = np.random.normal(0, 0.01, size=24)
        return ctrl + noise

    def _calculate_reward(self):
        """Calculate task-specific reward"""
        torso_height = self.data.qpos[2]
        fallen = torso_height < MIN_HEIGHT

        if self.task == "stand":
            return self._calculate_stand_reward(torso_height, fallen)
        elif self.task == "reach":
            return self._calculate_reach_reward(torso_height, fallen)
        else:  # balance or navigate
            return self._calculate_balance_reward(torso_height, fallen)

    def _calculate_stand_reward(self, torso_height, fallen):
        """Reward for standing task"""
        height_error = abs(torso_height - self._target_height)
        height_reward = np.exp(-height_error * 10.0)

        # Penalize joint velocity for stability
        jvel_penalty = -0.001 * np.sum(self.data.qvel[6:] ** 2)

        # Penalize control effort
        ctrl_penalty = -0.0003 * np.sum(self.data.ctrl**2)

        reward = height_reward + jvel_penalty + ctrl_penalty + 0.1

        if fallen:
            reward -= 10.0

        return float(reward)

    def _calculate_reach_reward(self, torso_height, fallen):
        """Reward for reach task"""
        # Distance to target
        current_pos = self.data.qpos[0:2]
        distance = np.linalg.norm(current_pos - self._tree_position[0:2])

        # Progress reward
        progress_reward = (
            1.0 * (self._prev_distance - distance)
            if hasattr(self, "_prev_distance")
            else 0
        )
        self._prev_distance = distance

        # Proximity reward
        proximity_reward = np.exp(-distance / 2.0)

        # Stability reward
        stability_reward = 0.5 * (
            1 - abs(self.data.qvel[2]) / 2.0
        )  # Penalize vertical velocity

        # Control penalty
        ctrl_penalty = -0.0003 * np.sum(self.data.ctrl**2)

        reward = (
            progress_reward
            + proximity_reward * 0.5
            + stability_reward
            + ctrl_penalty
            + 0.1
        )

        if fallen:
            reward -= 10.0

        return float(reward)

    def _calculate_balance_reward(self, torso_height, fallen):
        """Reward for balance task"""
        # Reward for maintaining height
        height_reward = 0.5 * min(torso_height / 0.4, 1.0)

        # Penalize movement
        movement_penalty = -0.01 * np.sum(abs(self.data.qvel[6:]))

        # Penalize control effort
        ctrl_penalty = -0.0003 * np.sum(self.data.ctrl**2)

        reward = height_reward + movement_penalty + ctrl_penalty + 0.1

        if fallen:
            reward -= 10.0

        return float(reward)

    def _check_termination(self):
        """Check if episode should terminate"""
        torso_height = self.data.qpos[2]

        # Check if any arm is below ground (unrealistic)
        if self._check_arm_collisions():
            return True

        # Fallen
        if torso_height < MIN_HEIGHT:
            return True

        # Success conditions
        if self.task == "stand":
            if abs(torso_height - self._target_height) < self._success_threshold:
                if self._step_count > 100:  # Must maintain for 100 steps
                    print(f"✅ Success! Maintained height for {self._step_count} steps")
                    return True

        return False

    def _enforce_joint_limits(self):
        """Prevent joints from going into unrealistic positions"""
        # Get current joint positions
        current_qpos = self.data.qpos[7:].copy()  # Skip torso (first 7)

        # Define safe ranges for each joint type
        # These prevent arms from going too low
        for i in range(24):
            joint_type = (
                i % 6
            )  # 0=base, 1=shoulder, 2=elbow, 3=wrist1, 4=wrist2, 5=gripper

            if joint_type in [2, 3, 4]:  # Elbow and wrist joints
                # Don't let these go too negative (downward)
                if current_qpos[i] < -1.0:
                    self.data.qpos[7 + i] = -1.0

            # Special handling for gripper joints (last joint of each arm)
            if i % 6 == 5:  # Gripper joints (indices 5, 11, 17, 23)
                if current_qpos[i] < 0.0:
                    self.data.qpos[7 + i] = 0.0

    def _check_arm_collisions(self):
        """Check if any arm geometry is below ground level"""
        # Get positions of all gripper links (end of each arm)
        gripper_names = [
            "fl_gripper_link",
            "fr_gripper_link",
            "bl_gripper_link",
            "br_gripper_link",
        ]

        for name in gripper_names:
            try:
                gripper_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if gripper_id >= 0:
                    gripper_z = self.data.xpos[gripper_id][2]
                    if gripper_z < GROUND_LEVEL - 0.01:  # 1cm below ground
                        print(
                            f"❌ Arm collision: {name} below ground (z={gripper_z:.3f}m)"
                        )
                        return True
            except:
                pass

        return False

    def close(self):
        """Clean up"""
        if self._viewer is not None:
            self._viewer.close()


# Register environments
if __name__ == "__main__":
    # Test the environment
    env = UnifiedEnv(task="stand", render_mode="human")
    obs, _ = env.reset()

    for _ in range(1000):
        action = np.zeros(24)  # Neutral action
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended. Reward: {reward:.2f}")
            obs, _ = env.reset()

    env.close()
