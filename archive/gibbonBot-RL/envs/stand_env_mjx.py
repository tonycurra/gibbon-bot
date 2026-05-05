"""
MJX environment — GPU-accelerated via JAX vmap.

Stateless functions (reset, step) operate on MJX data pytrees.
Call load_model() once, then vmap reset/step over N_ENVS.

Reward + motor params are passed as a plain tuple of JAX scalars (rparams) so
they can be updated between rollouts without recompiling the JIT:
  (target_z, target_tol, w_height, w_upright,
   arm_safety_margin, w_proximity, flip_penalty, motor_max_speed)

PHYSICS_STEPS_PER_ACTION is read once from params_mjx at import — it is the
JAX scan length and cannot be hot-reloaded. Restart training to change it.
"""

import importlib
import pathlib

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

import envs.params_mjx as _p

_ROOT        = pathlib.Path(__file__).parent.parent
SCENE        = _ROOT / "data" / "scene.xml"
_PARAMS_FILE = pathlib.Path(__file__).parent / "params_mjx.py"

MAX_STEPS = 500   # ~32s sim time at 32 substeps × 0.002s — plenty for stand+hold
ACT_DIM   = 24
OBS_DIM   = 83  # z(1)+quat(4)+linvel(3)+angvel(3)+jpos(24)+jvel(24)+torques(24)

# Static at import — baked into the JAX scan. Change in params_mjx.py + restart.
PHYSICS_STEPS_PER_ACTION = int(_p.PHYSICS_STEPS_PER_ACTION)

# Body IDs for arm proximity penalty: FL, FR, BL, BR (6 bodies each)
_ARM_GROUPS_NP = np.array([
    [2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25],
], dtype=np.int32)

# All 6 cross-arm pairs (indices into _ARM_GROUPS_NP)
_PAIR_A = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
_PAIR_B = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)

_params_mtime = 0.0


def read_params():
    """Hot-reload params_mjx.py if changed. Returns tuple of floats for rparams."""
    global _params_mtime
    try:
        mtime = _PARAMS_FILE.stat().st_mtime
        if mtime != _params_mtime:
            importlib.reload(_p)
            _params_mtime = mtime
    except Exception as e:
        print(f"[params_mjx] reload error: {e}")
    return (
        float(_p.TARGET_Z),
        float(_p.TARGET_TOL),
        float(_p.W_HEIGHT),
        float(_p.W_UPRIGHT),
        float(_p.ARM_SAFETY_MARGIN),
        float(_p.W_PROXIMITY),
        float(_p.FLIP_PENALTY),
        float(_p.MOTOR_MAX_SPEED),
        float(_p.W_SMOOTH),
        float(_p.W_JVEL),
        float(_p.W_ENERGY),
        float(_p.W_ALIVE),
        float(getattr(_p, "ACTION_SMOOTH_ALPHA", 1.0)),
    )


def load_model():
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    mj_model.opt.timestep = 0.002

    # Mesh geoms cause combinatorial JIT explosion in MJX — disable their collision.
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0

    # Joint damping is set per-joint in scene.xml via motor-class defaults
    # (st3235 / st3215_hs), derived from datasheet τ_stall / ω_no_load.
    # MOTOR_KP stays a tuning knob (controller stiffness, not motor spec).
    kp = float(getattr(_p, "MOTOR_KP", 8.0))
    mj_model.actuator_gainprm[:, 0] = kp
    mj_model.actuator_biasprm[:, 1] = -kp  # position actuator bias must match -kp

    mx = mjx.put_model(mj_model)
    jnt_lo = jnp.array([mj_model.jnt_range[i][0] for i in range(1, mj_model.njnt)])
    jnt_hi = jnp.array([mj_model.jnt_range[i][1] for i in range(1, mj_model.njnt)])
    return mx, jnt_lo, jnt_hi


def reset(mx, jnt_lo, jnt_hi, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    dx = mjx.make_data(mx)

    xy  = jax.random.uniform(k1, (2,), minval=-0.1, maxval=0.1)
    yaw = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)

    qpos = dx.qpos.at[0].set(xy[0])
    qpos = qpos.at[1].set(xy[1])
    qpos = qpos.at[2].set(0.10)               # spawn near floor — robot must push up
    qpos = qpos.at[3].set(jnp.cos(yaw / 2))   # w — upright orientation
    qpos = qpos.at[4].set(0.0)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(jnp.sin(yaw / 2))

    rng, knoise = jax.random.split(rng)
    noise = jax.random.uniform(knoise, (24,), minval=-0.15, maxval=0.15)
    # robot joints only — slices [7:7+24]; grove DOFs (branches, olives) come after
    qpos = qpos.at[7:7 + ACT_DIM].set(noise)

    dx = dx.replace(qpos=qpos, qvel=jnp.zeros_like(dx.qvel), ctrl=jnp.zeros_like(dx.ctrl))
    dx = mjx.forward(mx, dx)

    motor_pos   = jnp.zeros(ACT_DIM)
    prev_action = jnp.zeros(ACT_DIM)
    obs = _get_obs(dx)
    return dx, obs, motor_pos, prev_action


def step(mx, jnt_lo, jnt_hi, dx, action, motor_pos, prev_action, rparams):
    target_z, target_tol, w_height, w_upright, \
        arm_safety_margin, w_proximity, flip_penalty, motor_max_speed, \
        w_smooth, w_jvel, w_energy, w_alive, action_alpha = rparams

    # Action low-pass filter: smooths raw policy output before it ever hits motors.
    filt_action = action_alpha * action + (1.0 - action_alpha) * prev_action

    substep_speed = motor_max_speed / PHYSICS_STEPS_PER_ACTION

    def substep(carry, _):
        mpos, d = carry
        delta = jnp.clip(filt_action - mpos, -substep_speed, substep_speed)
        mpos  = mpos + delta
        ctrl  = 0.5 * (mpos + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mx, d)
        return (mpos, d), None

    (motor_pos, dx), _ = jax.lax.scan(
        substep, (motor_pos, dx), None, length=PHYSICS_STEPS_PER_ACTION
    )

    obs    = _get_obs(dx)
    torso_z = dx.qpos[2]
    w       = dx.qpos[3]

    height_err    = (torso_z - target_z) / target_tol
    height_reward = w_height * jnp.exp(-0.5 * height_err ** 2)
    upright_reward = jnp.where(w > 0, w_upright * w * w, 0.0)
    prox_penalty   = w_proximity * _proximity_penalty(dx.xpos, arm_safety_margin)
    smooth_penalty = w_smooth * jnp.sum((filt_action - prev_action) ** 2)
    jvel_penalty   = w_jvel * jnp.sum(dx.qvel[6:6 + ACT_DIM] ** 2)
    energy_penalty = w_energy * jnp.sum(dx.ctrl ** 2)

    upside_down = w < 0
    alive_bonus = w_alive * (1.0 - upside_down.astype(jnp.float32))
    reward = (height_reward + upright_reward + alive_bonus
              - prox_penalty - smooth_penalty - jvel_penalty - energy_penalty
              - flip_penalty * upside_down)
    done   = upside_down

    return dx, obs, reward, done, motor_pos, filt_action


def _proximity_penalty(xpos, arm_safety_margin):
    groups = jnp.array(_ARM_GROUPS_NP)
    pair_a = jnp.array(_PAIR_A)
    pair_b = jnp.array(_PAIR_B)

    def pair_penalty(a_idx, b_idx):
        pos_a = xpos[groups[a_idx]]
        pos_b = xpos[groups[b_idx]]
        diffs = pos_a[:, None, :] - pos_b[None, :, :]
        dists = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))
        return jnp.sum(jnp.maximum(arm_safety_margin - dists, 0.0) ** 2)

    return jnp.sum(jax.vmap(pair_penalty)(pair_a, pair_b))


def _get_obs(dx):
    return jnp.concatenate([
        dx.qpos[2:3],            # torso z
        dx.qpos[3:7],            # quat (w x y z)
        dx.qvel[0:3],            # linear vel
        dx.qvel[3:6],            # angular vel
        dx.qpos[7:7 + ACT_DIM],  # 24 robot joint positions (skip grove DOFs)
        dx.qvel[6:6 + ACT_DIM],  # 24 robot joint velocities
        dx.qfrc_actuator[6:30],  # 24 joint torques
    ]).astype(jnp.float32)
