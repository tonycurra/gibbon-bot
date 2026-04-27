"""
Gymnasium environment: teach the ape robot to lift itself and stand.

Observation (83 dims):
  [0]      torso height (z)
  [1:5]    torso orientation quaternion (w x y z)
  [5:8]    torso linear velocity
  [8:11]   torso angular velocity
  [11:35]  joint positions  (24)
  [35:59]  joint velocities (24)
  [59:83]  joint torques via qfrc_actuator (24) — load proxy matching real servo feedback

Action (24 dims):
  Target joint positions, normalised to [-1, 1].
  Mapped back to each joint's [lo, hi] range before writing to data.ctrl.

Reward per step:
  + height    up to 1.0 — Gaussian centred on TARGET_Z (0.30 m)
  + upright   up to 0.3 — penalises torso tilt (quat w^2)
  + alive     0.05 per step
  - energy    5e-4 * sum(ctrl^2)
  - jvel      1e-3 * sum(jvel^2)   (discourages flailing)
"""

import importlib
import pathlib

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from . import hints as hints_mod

ROOT = pathlib.Path(__file__).parent.parent.parent.parent  # Project root
SCENE = ROOT / "data" / "scene.xml"
HINTS_FILE = pathlib.Path(__file__).parent / "hints.py"
TARGET_Z = 0.30   # reward peak: torso this high = full stand
LIFTED_Z = 0.18   # "got up": torso clearly off ground (resting flat = ~0.05m)
MIN_HEIGHT = 0.03  # below this → fallen
SPAWN_Z = 0.35     # drop from here, arms clear the floor
SETTLE_VEL = 0.05  # max joint velocity considered "still"
SETTLE_STEPS = 2000  # max physics steps to wait for settle
MAX_STEPS = 2000
PHYSICS_STEPS_PER_ACTION = 50   # 50 × 0.002s = 0.1s per action — slow, deliberate
MOTOR_MAX_SPEED = 0.015         # max change per step — full sweep takes ~8s real-time
MAX_ACTIVE_JOINTS = 3           # penalize moving more than this many joints at once

FLOOR_Z = 0.04          # body below this = in contact with floor
# body IDs that should plant on floor, not drag: gripper tips (7,13,19,25) + link3s (4,10,16,22)
_CONTACT_BODIES = [4, 7, 10, 13, 16, 19, 22, 25]

# Arms splayed down — a reasonable starting pose for learning to stand.
# Each leg: base(link1) rotated outward, link2 angled down, rest relaxed.
# Indices per leg: [base, l1l2, l2l3, l3l4, l4l5, gripper]  × 4 legs
_STAND_INIT = np.array(
    [
        # fl (front-left)
        0.0,
        1.0,
        1.5,
        0.0,
        0.0,
        0.0,
        # fr (front-right)
        0.0,
        1.0,
        1.5,
        0.0,
        0.0,
        0.0,
        # bl (back-left)
        0.0,
        1.0,
        1.5,
        0.0,
        0.0,
        0.0,
        # br (back-right)
        0.0,
        1.0,
        1.5,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


class StandEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)

        self._jnt_lo = np.array(
            [self.model.jnt_range[i][0] for i in range(1, self.model.njnt)]
        )
        self._jnt_hi = np.array(
            [self.model.jnt_range[i][1] for i in range(1, self.model.njnt)]
        )

        n_obs = 83  # 59 + 24 joint torques
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_action = np.zeros(24, dtype=np.float32)
        self._motor_pos = np.zeros(24, dtype=np.float32)  # current motor position in [-1,1]
        self._hints_mtime = 0.0
        self._hints = dict(hints_mod.HINTS)

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer = None

    # ── core API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = SPAWN_Z
        self.data.qpos[3] = 1.0  # quat w (upright)
        self.data.qpos[4:7] = 0.0

        # arms in splayed-down pose + small noise
        self.data.qpos[7:] = _STAND_INIT + rng.uniform(-0.05, 0.05, size=24)

        # let robot fall and settle before the episode starts
        self.data.ctrl[:] = 0
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
            if np.max(np.abs(self.data.qvel)) < SETTLE_VEL:
                break

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        # initialize motor positions from actual settled joint positions
        settled_pos = self.data.qpos[7:].astype(np.float32)
        self._motor_pos = np.clip(
            2.0 * (settled_pos - self._jnt_lo) / (self._jnt_hi - self._jnt_lo) - 1.0,
            -1.0, 1.0,
        )
        self._prev_action = self._motor_pos.copy()
        self._reload_hints()

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # motor model: each joint moves at most MOTOR_MAX_SPEED per step toward target
        delta = np.clip(action - self._motor_pos, -MOTOR_MAX_SPEED, MOTOR_MAX_SPEED)
        self._motor_pos = self._motor_pos + delta
        ctrl = 0.5 * (self._motor_pos + 1.0) * (self._jnt_hi - self._jnt_lo) + self._jnt_lo
        self.data.ctrl[:] = ctrl

        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1
        obs = self._get_obs()

        torso_z = float(self.data.qpos[2])
        above_ground = torso_z  # ground is z=0
        fallen = above_ground < MIN_HEIGHT
        got_up = above_ground > LIFTED_Z

        # height reward: Gaussian around TARGET_Z above terrain, max 1.0
        height_err = (above_ground - TARGET_Z) / 0.08
        height_reward = float(np.exp(-0.5 * height_err**2))

        # upright reward: penalise tilt — quat[0] is w
        w = float(self.data.qpos[3])
        upright_reward = 0.3 * w * w if w > 0 else 0.0
        upside_down = w < 0  # w < 0 means torso rotated more than 90° — on its back

        jvel = self.data.qvel[6:]
        motor_delta = float(np.sum(delta ** 2))  # actual motor movement this step
        self._prev_action = action.copy()

        # drag penalty: bodies near floor moving horizontally = dragging
        drag_penalty = 0.0
        grippers_planted = 0
        for bid in _CONTACT_BODIES:
            bz = float(self.data.xpos[bid][2])
            if bz < FLOOR_Z:
                horiz_vel = float(np.sqrt(self.data.cvel[bid, 3]**2 + self.data.cvel[bid, 4]**2))
                drag_penalty += horiz_vel
                grippers_planted += 1
        # plant-and-lift: grippers on floor while torso is rising = good push-off
        torso_vz = float(self.data.cvel[1, 5])  # body 1 = torso, z velocity
        plant_and_lift = float(grippers_planted) * max(0.0, torso_vz)

        # allow only a few joints to move at once — encourages sequential animal-like motion
        joints_moving = int(np.sum(np.abs(delta) > 1e-4))
        too_many_joints_penalty = -0.3 * max(0, joints_moving - MAX_ACTIVE_JOINTS)

        reward = (
            height_reward
            + upright_reward
            + (0.5 if got_up else 0.0)   # bonus for lifting torso above LIFTED_Z
            + 0.3 * plant_and_lift        # grippers planted + torso rising = good
            + 0.05                        # alive bonus
            - 5e-4 * float(np.sum(ctrl**2))   # energy
            - 1e-3 * float(np.sum(jvel**2))   # joint velocity / flailing
            - 5e-2 * motor_delta               # penalize actual motor movement (shaking)
            - 0.5 * drag_penalty               # dragging bodies along floor = bad
            + too_many_joints_penalty
        )

        reward += self._hints_reward()

        if upside_down:
            reward -= 10.0  # big penalty for flipping onto back

        terminated = upside_down
        truncated = self._step_count >= MAX_STEPS

        info = {
            "torso_z": torso_z,
            "above_ground": above_ground,
            "fallen": fallen,
            "upside_down": upside_down,
            "got_up": got_up,
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "active_hints": [k for k, v in self._hints.items() if v != 0.0],
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _reload_hints(self):
        try:
            mtime = HINTS_FILE.stat().st_mtime
            if mtime != self._hints_mtime:
                importlib.reload(hints_mod)
                self._hints = dict(hints_mod.HINTS)
                self._hints_mtime = mtime
                active = {k: v for k, v in self._hints.items() if v != 0.0}
                print(f"[hints] reloaded — active: {active or 'none'}")
        except Exception as e:
            print(f"[hints] reload error: {e}")

    def _hints_reward(self) -> float:
        h = self._hints
        r = 0.0
        qpos = self.data.qpos
        qvel = self.data.qvel

        if h.get("upright", 0.0) != 0.0:
            w = float(qpos[3])
            # 1 - w^2: 0 when upright, up to 1 when tipped — negate for reward
            r += h["upright"] * (
                w * w - 1.0 + 1.0
            )  # = w^2, same as base but user-weighted

        if h.get("spread_legs", 0.0) != 0.0:
            # base joints are indices 0,6,12,18 in the arm joint array (qpos[7:])
            base_joints = qpos[7:][::6]
            r += h["spread_legs"] * float(np.mean(np.abs(base_joints)))

        if h.get("front_arms_down", 0.0) != 0.0:
            # fl gripper = body "fl_gripper_link", fr gripper = "fr_gripper_link"
            fl_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "fl_gripper_link"
            )
            fr_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "fr_gripper_link"
            )
            if fl_id >= 0 and fr_id >= 0:
                fl_z = float(self.data.xpos[fl_id][2])
                fr_z = float(self.data.xpos[fr_id][2])
                r -= h["front_arms_down"] * (fl_z + fr_z)  # lower = better

        if h.get("back_arms_down", 0.0) != 0.0:
            bl_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "bl_gripper_link"
            )
            br_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "br_gripper_link"
            )
            if bl_id >= 0 and br_id >= 0:
                bl_z = float(self.data.xpos[bl_id][2])
                br_z = float(self.data.xpos[br_id][2])
                r -= h["back_arms_down"] * (bl_z + br_z)

        if h.get("symmetry", 0.0) != 0.0:
            jpos = qpos[7:]
            # compare fl vs fr (indices 0:6 vs 6:12) and bl vs br (12:18 vs 18:24)
            lr_diff = float(np.mean(np.abs(jpos[0:6] - jpos[6:12]))) + float(
                np.mean(np.abs(jpos[12:18] - jpos[18:24]))
            )
            r -= h["symmetry"] * lr_diff

        if h.get("stillness", 0.0) != 0.0:
            r -= h["stillness"] * float(np.sum(qvel[6:] ** 2))

        return r

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        obs = np.concatenate(
            [
                qpos[2:3],  # torso z
                qpos[3:7],  # torso quat (w x y z)
                qvel[0:3],  # linear vel
                qvel[3:6],  # angular vel
                qpos[7:],  # 24 joint positions
                qvel[6:],  # 24 joint velocities
                self.data.qfrc_actuator[6:30],  # 24 joint torques (load proxy)
            ]
        ).astype(np.float32)
        return obs
