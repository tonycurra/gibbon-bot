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
  + height    up to W_HEIGHT — Gaussian centred on TARGET_Z
  + upright   up to W_UPRIGHT — torso level (quat w^2)
  - FLIP_PENALTY (one-time) if upside-down → terminate
  - W_PROXIMITY * quadratic penalty when arms enter ARM_SAFETY_MARGIN zone
"""

import importlib
import pathlib
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

import envs.params as _params_mod

_ROOT       = pathlib.Path(__file__).parent.parent
SCENE       = _ROOT / "data" / "scene.xml"
_PARAMS_FILE = pathlib.Path(__file__).parent / "params.py"

MAX_STEPS        = 2000
ACT_DIM          = 24
ARM_CONTACT_DIST = 0.02

_ARM_GROUPS = [
    [2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25],
]

SETTLE_VEL   = 0.05
SETTLE_STEPS = 3000
_STAND_INIT  = np.zeros(ACT_DIM, dtype=np.float64)


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

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(83,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_action = np.zeros(24, dtype=np.float32)
        self._motor_pos = np.zeros(24, dtype=np.float32)
        self._params_mtime = 0.0
        self._p = self._load_params()

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer = None

    # ── core API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        p = self._load_params()
        dt = float(self.model.opt.timestep)
        speed = max(getattr(p, "REALTIME_FACTOR", 1.0), 0.01)
        sleep_dt = (dt / speed) if self._viewer is not None else 0.0

        no_settle = (options or {}).get("no_settle", False)

        for _attempt in range(5):
            self.data.qpos[0] = rng.uniform(-0.1, 0.1)
            self.data.qpos[1] = rng.uniform(-0.1, 0.1)
            self.data.qpos[2] = 0.45

            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[3] = np.cos(yaw / 2)
            self.data.qpos[4] = 0.0
            self.data.qpos[5] = 0.0
            self.data.qpos[6] = np.sin(yaw / 2)

            self.data.qpos[7:] = _STAND_INIT + rng.uniform(-0.15, 0.15, size=24)
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0

            if not no_settle:
                still_steps = 0
                for _ in range(SETTLE_STEPS):
                    mujoco.mj_step(self.model, self.data)
                    if self._viewer is not None:
                        self._viewer.sync()
                        if sleep_dt > 0:
                            time.sleep(sleep_dt)
                    if np.max(np.abs(self.data.qvel)) < SETTLE_VEL:
                        still_steps += 1
                        if still_steps >= 150:
                            break
                    else:
                        still_steps = 0

            if float(self.data.qpos[3]) > 0:
                break

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        settled_pos = self.data.qpos[7:].astype(np.float32)
        self._motor_pos = np.clip(
            2.0 * (settled_pos - self._jnt_lo) / (self._jnt_hi - self._jnt_lo) - 1.0,
            -1.0, 1.0,
        )
        self._prev_action = self._motor_pos.copy()
        self._p = self._load_params()

        return self._get_obs(), {}

    def step(self, action):
        p = self._load_params()
        action = np.asarray(action, dtype=np.float32)

        substep_speed = p.MOTOR_MAX_SPEED / p.PHYSICS_STEPS_PER_ACTION

        dt = float(self.model.opt.timestep)
        speed = max(getattr(p, "REALTIME_FACTOR", 1.0), 0.01)
        sleep_dt = (dt / speed) if self._viewer is not None else 0.0

        for _ in range(p.PHYSICS_STEPS_PER_ACTION):
            sub_delta = np.clip(action - self._motor_pos, -substep_speed, substep_speed)
            self._motor_pos += sub_delta
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

        self._step_count += 1
        obs = self._get_obs()

        torso_z = float(self.data.qpos[2])
        w = float(self.data.qpos[3])
        upside_down = w < 0

        height_err = (torso_z - p.TARGET_Z) / p.TARGET_TOL
        height_reward = p.W_HEIGHT * float(np.exp(-0.5 * height_err**2))
        upright_reward = p.W_UPRIGHT * w * w if w > 0 else 0.0

        arms_colliding = False
        proximity_penalty = 0.0
        for i, grp_a in enumerate(_ARM_GROUPS):
            pos_a = self.data.xpos[grp_a]
            for grp_b in _ARM_GROUPS[i + 1:]:
                pos_b = self.data.xpos[grp_b]
                dists = np.linalg.norm(pos_a[:, None] - pos_b[None, :], axis=-1)
                if float(dists.min()) < ARM_CONTACT_DIST:
                    arms_colliding = True
                violations = p.ARM_SAFETY_MARGIN - dists
                proximity_penalty += float(np.sum(np.square(np.maximum(violations, 0))))

        self._prev_action = action.copy()

        reward = height_reward + upright_reward - p.W_PROXIMITY * proximity_penalty
        if upside_down:
            reward -= p.FLIP_PENALTY

        terminated = upside_down
        truncated = self._step_count >= MAX_STEPS

        info = {
            "torso_z": torso_z,
            "upside_down": upside_down,
            "arms_colliding": arms_colliding,
            "proximity_penalty": proximity_penalty,
            "height_reward": height_reward,
            "upright_reward": upright_reward,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_params(self):
        try:
            mtime = _PARAMS_FILE.stat().st_mtime
            if mtime != self._params_mtime:
                importlib.reload(_params_mod)
                self._params_mtime = mtime
        except Exception as e:
            print(f"[params] reload error: {e}")
        return _params_mod

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([
            qpos[2:3],                       # torso z
            qpos[3:7],                       # torso quat (w x y z)
            qvel[0:3],                       # linear vel
            qvel[3:6],                       # angular vel
            qpos[7:],                        # 24 joint positions
            qvel[6:],                        # 24 joint velocities
            self.data.qfrc_actuator[6:30],   # 24 joint torques (load proxy)
        ]).astype(np.float32)
