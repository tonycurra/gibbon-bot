"""
Gymnasium environment: teach the ape robot to lift itself and stand.

Observation (58 dims):
  [0]      torso height (z)
  [1:5]    torso orientation quaternion (w x y z)
  [5:8]    torso linear velocity
  [8:11]   torso angular velocity
  [11:35]  joint positions  (24)
  [35:59]  joint velocities (24)
  Total: 59

Action (24 dims):
  Target joint positions, normalised to [-1, 1].
  Mapped back to each joint's [lo, hi] range before writing to data.ctrl.

Reward per step:
  + height    up to 1.0 — Gaussian centred on TARGET_Z (0.30 m)
  + upright   up to 0.3 — penalises torso tilt (quat w^2)
  + alive     0.05 per step
  - energy    5e-4 * sum(ctrl^2)
  - jvel      1e-3 * sum(jvel^2)   (discourages flailing)
  - fall      -10 on termination if torso too low
"""

import importlib
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import pathlib
import terrain as terrain_mod
import hints as hints_mod

SCENE      = pathlib.Path(__file__).parent / "scene.xml"
HINTS_FILE = pathlib.Path(__file__).parent / "hints.py"
TARGET_Z   = 0.30   # comfortable standing height above terrain (metres)
MIN_HEIGHT = 0.03   # torso this far below terrain surface → fallen
SPAWN_CLEARANCE = 0.07  # start this far above terrain
MAX_STEPS  = 2000
PHYSICS_STEPS_PER_ACTION = 5

# Arms splayed down — a reasonable starting pose for learning to stand.
# Each leg: base(link1) rotated outward, link2 angled down, rest relaxed.
# Indices per leg: [base, l1l2, l2l3, l3l4, l4l5, gripper]  × 4 legs
_STAND_INIT = np.array([
    # fl (front-left)
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
    # fr (front-right)
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
    # bl (back-left)
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
    # br (back-right)
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
], dtype=np.float64)


class StandEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        terrain_mod.apply(self.model)
        self.data  = mujoco.MjData(self.model)

        self._jnt_lo = np.array([self.model.jnt_range[i][0]
                                  for i in range(1, self.model.njnt)])
        self._jnt_hi = np.array([self.model.jnt_range[i][1]
                                  for i in range(1, self.model.njnt)])

        n_obs = 59
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        self._step_count  = 0
        self._hints_mtime = 0.0
        self._hints       = dict(hints_mod.HINTS)

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer = None

    # ── core API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # start torso just above terrain at spawn XY
        spawn_x, spawn_y = 0.0, 0.0
        ground_z = self._terrain_z(spawn_x, spawn_y)
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = ground_z + SPAWN_CLEARANCE
        self.data.qpos[3] = 1.0   # quat w (upright)
        self.data.qpos[4:7] = 0.0

        # arms in splayed-down pose + small noise
        self.data.qpos[7:] = _STAND_INIT + rng.uniform(-0.1, 0.1, size=24)

        mujoco.mj_forward(self.model, self.data)
        self._ground_z   = ground_z
        self._step_count = 0
        self._reload_hints()

        return self._get_obs(), {}

    def step(self, action):
        ctrl = 0.5 * (action + 1.0) * (self._jnt_hi - self._jnt_lo) + self._jnt_lo
        self.data.ctrl[:] = ctrl

        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1
        obs = self._get_obs()

        torso_z  = float(self.data.qpos[2])
        above_ground = torso_z - self._ground_z
        fallen   = above_ground < MIN_HEIGHT

        # height reward: Gaussian around TARGET_Z above terrain, max 1.0
        height_err    = (above_ground - TARGET_Z) / 0.08
        height_reward = float(np.exp(-0.5 * height_err ** 2))

        # upright reward: penalise tilt — quat[0] is w
        w = float(self.data.qpos[3])
        upright_reward = 0.3 * w * w

        jvel   = self.data.qvel[6:]
        reward = (
              height_reward
            + upright_reward
            + 0.05                                    # alive bonus
            - 5e-4 * float(np.sum(ctrl ** 2))        # energy
            - 1e-3 * float(np.sum(jvel ** 2))        # velocity / flailing
        )
        if fallen:
            reward -= 10.0

        reward += self._hints_reward()

        terminated = fallen
        truncated  = self._step_count >= MAX_STEPS

        info = {"torso_z": torso_z, "above_ground": above_ground,
                "fallen": fallen, "height_reward": height_reward,
                "upright_reward": upright_reward,
                "active_hints": [k for k, v in self._hints.items() if v != 0.0]}
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
            r += h["upright"] * (w * w - 1.0 + 1.0)   # = w^2, same as base but user-weighted

        if h.get("spread_legs", 0.0) != 0.0:
            # base joints are indices 0,6,12,18 in the arm joint array (qpos[7:])
            base_joints = qpos[7:][::6]
            r += h["spread_legs"] * float(np.mean(np.abs(base_joints)))

        if h.get("front_arms_down", 0.0) != 0.0:
            # fl gripper = body "fl_gripper_link", fr gripper = "fr_gripper_link"
            fl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fl_gripper_link")
            fr_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fr_gripper_link")
            if fl_id >= 0 and fr_id >= 0:
                fl_z = float(self.data.xpos[fl_id][2]) - self._ground_z
                fr_z = float(self.data.xpos[fr_id][2]) - self._ground_z
                r -= h["front_arms_down"] * (fl_z + fr_z)   # lower = better

        if h.get("back_arms_down", 0.0) != 0.0:
            bl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bl_gripper_link")
            br_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "br_gripper_link")
            if bl_id >= 0 and br_id >= 0:
                bl_z = float(self.data.xpos[bl_id][2]) - self._ground_z
                br_z = float(self.data.xpos[br_id][2]) - self._ground_z
                r -= h["back_arms_down"] * (bl_z + br_z)

        if h.get("symmetry", 0.0) != 0.0:
            jpos = qpos[7:]
            # compare fl vs fr (indices 0:6 vs 6:12) and bl vs br (12:18 vs 18:24)
            lr_diff = float(np.mean(np.abs(jpos[0:6] - jpos[6:12]))) \
                    + float(np.mean(np.abs(jpos[12:18] - jpos[18:24])))
            r -= h["symmetry"] * lr_diff

        if h.get("stillness", 0.0) != 0.0:
            r -= h["stillness"] * float(np.sum(qvel[6:] ** 2))

        return r

    def _terrain_z(self, x: float, y: float) -> float:
        """World-space terrain height at (x, y). Returns 0.0 if no hfield."""
        hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hid < 0:
            return 0.0
        size  = self.model.hfield_size[hid]   # [x_half, y_half, z_scale, base]
        nrow  = self.model.hfield_nrow[hid]
        ncol  = self.model.hfield_ncol[hid]
        col   = int(np.clip((x + size[0]) / (2 * size[0]) * (ncol - 1), 0, ncol - 1))
        row   = int(np.clip((y + size[1]) / (2 * size[1]) * (nrow - 1), 0, nrow - 1))
        h_norm = float(self.model.hfield_data[row * ncol + col])
        return h_norm * float(size[2])   # z_scale; base already in geom pos

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        obs = np.concatenate([
            qpos[2:3],    # torso z
            qpos[3:7],    # torso quat (w x y z)
            qvel[0:3],    # linear vel
            qvel[3:6],    # angular vel
            qpos[7:],     # 24 joint positions
            qvel[6:],     # 24 joint velocities
        ]).astype(np.float32)
        return obs
