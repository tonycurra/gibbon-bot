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
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from . import hints as hints_mod
from . import params as params_mod

ROOT = pathlib.Path(__file__).parent.parent.parent.parent  # Project root
SCENE = ROOT / "data" / "scene_fixed.xml"
HINTS_FILE = pathlib.Path(__file__).parent / "hints.py"
PARAMS_FILE = pathlib.Path(__file__).parent / "params.py"

# Static constants (not hot-reloaded)
SPAWN_Z = 0.15  # Reduced from 0.35 to make robot fall from smaller distance
SETTLE_VEL = 0.05
SETTLE_STEPS = 3000  # enough for a full fall + 150-step stillness confirmation
MAX_STEPS = 2000
FLOOR_Z = 0.0

# Defaults (overridden each step from params_mod)
TARGET_Z = 0.30
LIFTED_Z = 0.18
MIN_HEIGHT = 0.03
PHYSICS_STEPS_PER_ACTION = 250
MOTOR_MAX_SPEED = 0.015
MAX_ACTIVE_JOINTS = 3
ARM_SAFETY_MARGIN = 0.06
# body IDs: FL=2..7, FR=8..13, BL=14..19, BR=20..25
_ARM_GROUPS = [
    [2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25],
]
# shoulder attachment bodies — base of each arm on the torso (one per arm)
_SHOULDER_BODIES = [2, 8, 14, 20]
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
        self._motor_pos = np.zeros(
            24, dtype=np.float32
        )  # current motor position in [-1,1]
        self._hints_mtime = 0.0
        self._hints = dict(hints_mod.HINTS)
        self._params_mtime = 0.0
        self._p = self._load_params()

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )
        else:
            self._viewer = None

    # ── core API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # randomize spawn position — upright, random yaw only, high enough to clear the floor
        self.data.qpos[0] = rng.uniform(-0.1, 0.1)  # x
        self.data.qpos[1] = rng.uniform(-0.1, 0.1)  # y
        self.data.qpos[2] = rng.uniform(
            0.50, 0.70
        )  # z — high enough arms never clip floor

        # random yaw only so torso is always upright at spawn
        yaw = rng.uniform(-np.pi, np.pi)
        self.data.qpos[3] = np.cos(yaw / 2)  # w
        self.data.qpos[4] = 0.0  # x
        self.data.qpos[5] = 0.0  # y
        self.data.qpos[6] = np.sin(yaw / 2)  # z

        # arms in splayed-down pose + noise for variety
        self.data.qpos[7:] = _STAND_INIT + rng.uniform(-0.3, 0.3, size=24)

        # drop with zero control and wait for the robot to fully settle
        self.data.ctrl[:] = 0
        p = self._load_params()
        dt = float(self.model.opt.timestep)
        realtime = getattr(p, "REALTIME_FACTOR", 0.0)
        sleep_dt = dt * realtime if self._viewer is not None else 0.0

        still_steps = 0
        required_still = (
            150  # must stay below SETTLE_VEL for this many consecutive steps
        )
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
            if np.max(np.abs(self.data.qvel)) < SETTLE_VEL:
                still_steps += 1
                if still_steps >= required_still:
                    break
            else:
                still_steps = 0

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        settled_pos = self.data.qpos[7:].astype(np.float32)
        self._motor_pos = np.clip(
            2.0 * (settled_pos - self._jnt_lo) / (self._jnt_hi - self._jnt_lo) - 1.0,
            -1.0,
            1.0,
        )
        self._prev_action = self._motor_pos.copy()
        self._reload_hints()
        self._p = self._load_params()

        return self._get_obs(), {}

    def step(self, action):
        p = self._load_params()
        action = np.asarray(action, dtype=np.float32)

        # Per-substep motor speed: total travel spread evenly across physics steps
        substep_speed = p.MOTOR_MAX_SPEED / p.PHYSICS_STEPS_PER_ACTION

        # Real-time sleep per physics step (0 during headless training)
        dt = float(self.model.opt.timestep)  # 0.002 s
        realtime = getattr(p, "REALTIME_FACTOR", 0.0)
        sleep_dt = dt * realtime if self._viewer is not None else 0.0

        total_delta = np.zeros_like(self._motor_pos)
        for _ in range(p.PHYSICS_STEPS_PER_ACTION):
            # motor moves a tiny bit each substep — continuous servo motion
            sub_delta = np.clip(action - self._motor_pos, -substep_speed, substep_speed)
            self._motor_pos += sub_delta
            total_delta += sub_delta
            ctrl = (
                0.5 * (self._motor_pos + 1.0) * (self._jnt_hi - self._jnt_lo)
                + self._jnt_lo
            )
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

        delta = total_delta  # used for smoothness penalty below

        self._step_count += 1
        obs = self._get_obs()

        torso_z = float(self.data.qpos[2])
        # use average shoulder height as the lift metric — 4 arm-base attachment points
        shoulder_z = float(np.mean([self.data.xpos[b][2] for b in _SHOULDER_BODIES]))
        above_ground = shoulder_z
        fallen = torso_z < p.MIN_HEIGHT
        got_up = above_ground > p.LIFTED_Z

        tol = getattr(p, "TARGET_TOL", 0.02)
        height_err = (above_ground - p.TARGET_Z) / tol
        height_reward = float(np.exp(-0.5 * height_err**2))

        w = float(self.data.qpos[3])
        upright_reward = p.W_UPRIGHT * w * w if w > 0 else 0.0
        upside_down = w < 0

        jvel = self.data.qvel[6:]
        motor_delta = float(np.sum(delta**2))
        self._prev_action = action.copy()

        proximity_penalty = 0.0
        arms_colliding = False
        margin = p.ARM_SAFETY_MARGIN
        touch_dist = getattr(
            p, "ARM_TOUCH_DIST", 0.01
        )  # below this = collision → terminate
        for i, grp_a in enumerate(_ARM_GROUPS):
            pos_a = self.data.xpos[grp_a]  # (6,3)
            for grp_b in _ARM_GROUPS[i + 1 :]:
                pos_b = self.data.xpos[grp_b]  # (6,3)
                dists = np.linalg.norm(pos_a[:, None] - pos_b[None, :], axis=-1)
                if float(dists.min()) < touch_dist:
                    arms_colliding = True
                violations = margin - dists
                proximity_penalty += float(np.sum(np.square(np.maximum(violations, 0))))

        drag_penalty = 0.0
        grippers_planted = 0
        for bid in _CONTACT_BODIES:
            bz = float(self.data.xpos[bid][2])
            if bz < FLOOR_Z:
                horiz_vel = float(
                    np.sqrt(self.data.cvel[bid, 3] ** 2 + self.data.cvel[bid, 4] ** 2)
                )
                drag_penalty += horiz_vel
                grippers_planted += 1
        torso_vz = float(self.data.cvel[1, 5])
        plant_and_lift = float(grippers_planted) * max(0.0, torso_vz)

        joints_moving = int(np.sum(np.abs(delta) > 1e-4))
        too_many_joints_penalty = -p.W_JOINTS * max(
            0, joints_moving - p.MAX_ACTIVE_JOINTS
        )

        # stability bonus: at target height AND low joint velocity
        stable_vel = getattr(p, "STABLE_VEL", 0.05)
        w_stable = getattr(p, "W_STABLE", 0.0)
        at_target = abs(above_ground - p.TARGET_Z) < tol
        is_stable = float(np.max(np.abs(self.data.qvel[6:]))) < stable_vel
        stability_bonus = w_stable if (at_target and is_stable) else 0.0

        reward = (
            p.W_HEIGHT * height_reward
            + upright_reward
            + (p.W_GOT_UP if got_up else 0.0)
            + stability_bonus
            + p.W_PLANT_LIFT * plant_and_lift
            + p.W_ALIVE
            - p.W_ENERGY * float(np.sum(ctrl**2))
            - p.W_JVEL * float(np.sum(jvel**2))
            - p.W_SMOOTH * motor_delta
            - p.W_DRAG * drag_penalty
            - p.W_PROXIMITY * proximity_penalty
            + too_many_joints_penalty
        )

        reward += self._hints_reward()

        if upside_down:
            reward -= p.FLIP_PENALTY

        arm_collision_penalty = getattr(p, "ARM_COLLISION_PENALTY", 50.0)
        if arms_colliding:
            reward -= arm_collision_penalty

        terminated = upside_down or arms_colliding
        truncated = self._step_count >= MAX_STEPS

        info = {
            "torso_z": torso_z,
            "above_ground": above_ground,
            "shoulder_z": shoulder_z,
            "fallen": fallen,
            "upside_down": upside_down,
            "arms_colliding": arms_colliding,
            "got_up": got_up,
            "stable": bool(at_target and is_stable),
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "active_hints": [k for k, v in self._hints.items() if v != 0.0],
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_params(self):
        try:
            mtime = PARAMS_FILE.stat().st_mtime
            if mtime != self._params_mtime:
                importlib.reload(params_mod)
                self._params_mtime = mtime
        except Exception as e:
            print(f"[params] reload error: {e}")
        return params_mod

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
