"""MJX environment — stand. Hold body upright at target height, no locomotion.

Reward = upright + height_gaussian + alive − flip_penalty. That's the entire
reward. No CPG (standing isn't a gait). No tree, no progress, no smoothness
soup. Minimal signal so PPO can find the gradient quickly.

88-dim obs contract (envs/_shared_obs.py): proprio(83) + target_slot(3, zeros
for stand) + gait_clock(2). Frozen across all curriculum tasks for clean
weight transfer.
"""
import importlib
import pathlib

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx

import envs.params_stand_mjx as _p
from envs._shared_load_model import base_load_model
from envs._shared_dynamics import apply_action, cpg_pace_offset
from envs._shared_obs import assemble_obs

_ROOT        = pathlib.Path(__file__).parent.parent
SCENE        = _ROOT / "data" / "scene.xml"
_PARAMS_FILE = pathlib.Path(__file__).parent / "params_stand_mjx.py"

MAX_STEPS = 200
ACT_DIM   = 24
OBS_DIM   = 88
GAIT_PERIOD_STEPS = 40
PHYSICS_STEPS_PER_ACTION = int(_p.PHYSICS_STEPS_PER_ACTION)

_params_mtime = 0.0


def read_params():
    global _params_mtime
    try:
        mtime = _PARAMS_FILE.stat().st_mtime
        if mtime != _params_mtime:
            importlib.reload(_p)
            _params_mtime = mtime
    except Exception as e:
        print(f"[params_stand_mjx] reload error: {e}")
    return (
        float(_p.TARGET_Z),
        float(_p.TARGET_TOL),
        float(_p.W_UPRIGHT),
        float(_p.W_HEIGHT),
        float(_p.W_ALIVE),
        float(_p.FLIP_PENALTY),
        float(_p.ACTION_SMOOTH_ALPHA),
        float(_p.W_CPG_PITCH),
        float(_p.W_CPG_ELBOW),
    )


def load_model():
    print("[stand_env] loading scene")
    _, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose = base_load_model(
        SCENE, _p, PHYSICS_STEPS_PER_ACTION
    )
    target_xy = jnp.zeros(2, dtype=jnp.float32)  # no spatial target
    return mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose, target_xy


def reset(mx, jnt_lo, jnt_hi, target_xy, rng):
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    dx = mjx.make_data(mx)

    # Spawn at origin with random yaw and slight joint noise — stand task
    # should be orientation-independent.
    yaw = jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi)
    xy_jitter = jax.random.uniform(k2, (2,), minval=-0.1, maxval=0.1)

    qpos = dx.qpos.at[0].set(xy_jitter[0])
    qpos = qpos.at[1].set(xy_jitter[1])
    qpos = qpos.at[2].set(0.30)
    qpos = qpos.at[3].set(jnp.cos(yaw / 2))
    qpos = qpos.at[4].set(0.0)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(jnp.sin(yaw / 2))

    joint_noise = jax.random.uniform(k3, (24,), minval=-0.20, maxval=0.20)
    qpos = qpos.at[7:7 + ACT_DIM].set(joint_noise)

    dx = dx.replace(qpos=qpos, qvel=jnp.zeros_like(dx.qvel), ctrl=jnp.zeros_like(dx.ctrl))
    dx = mjx.forward(mx, dx)

    motor_pos   = jnp.zeros(ACT_DIM)
    motor_vel   = jnp.zeros(ACT_DIM)
    prev_action = jnp.zeros(ACT_DIM)
    step_idx    = jnp.int32(0)
    obs = assemble_obs(dx, target_xy, step_idx, GAIT_PERIOD_STEPS, ACT_DIM)
    prev_scalar = jnp.float32(0.0)
    return dx, obs, motor_pos, motor_vel, prev_action, prev_scalar, step_idx


def step(mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose, target_xy,
         dx, action, motor_pos, motor_vel, prev_action, prev_scalar, step_idx, rparams):
    (target_z, target_tol, w_upright, w_height, w_alive, flip_penalty,
     action_alpha, w_cpg_pitch, w_cpg_elbow) = rparams

    filt_action = action_alpha * action + (1.0 - action_alpha) * prev_action

    cpg = cpg_pace_offset(step_idx, GAIT_PERIOD_STEPS, w_cpg_pitch, w_cpg_elbow, ACT_DIM)
    target = filt_action + cpg

    dx, motor_pos, motor_vel = apply_action(
        mx, jnt_lo, jnt_hi, rest_pose, motor_max_speed, motor_max_accel,
        PHYSICS_STEPS_PER_ACTION, dx, motor_pos, motor_vel, target,
    )

    step_idx = step_idx + 1
    obs = assemble_obs(dx, target_xy, step_idx, GAIT_PERIOD_STEPS, ACT_DIM)

    torso_z = dx.qpos[2]
    qw      = dx.qpos[3]

    # Reward — three positive terms + one penalty. That's it.
    height_err     = (torso_z - target_z) / target_tol
    height_reward  = w_height * jnp.exp(-0.5 * height_err ** 2)
    upright_reward = jnp.where(qw > 0, w_upright * qw * qw, 0.0)
    upside_down    = qw < 0
    alive_bonus    = w_alive * (1.0 - upside_down.astype(jnp.float32))

    reward = height_reward + upright_reward + alive_bonus - flip_penalty * upside_down.astype(jnp.float32)

    on_ground = torso_z < 0.05
    done = upside_down | on_ground

    # scalar carries qw (upright quat) — useful for early-stop diagnostics.
    return dx, obs, reward, done, motor_pos, motor_vel, filt_action, jnp.float32(qw), step_idx


def is_keeper(metrics):
    upr = metrics.get("mean_upright", 0.0)
    z   = metrics.get("mean_z", 0.0)
    if upr > 0.90 and z > 0.27:
        return True, f"upr{upr:.2f}_z{z:.2f}"
    return False, ""
