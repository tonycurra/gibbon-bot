"""MJX environment — turn left. Body yaw rate positive (counter-clockwise
viewed from above), in place (no translation). CPG disabled — turning is not
a gait.

Reward = yaw_rate − translation − flip + alive. Same minimal structure as
stand/walk_forward.
"""
import importlib
import pathlib

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx

import envs.params_turn_left_mjx as _p
from envs._shared_load_model import base_load_model
from envs._shared_dynamics import apply_action, cpg_pace_offset
from envs._shared_obs import assemble_obs

_ROOT        = pathlib.Path(__file__).parent.parent
SCENE        = _ROOT / "data" / "scene.xml"
_PARAMS_FILE = pathlib.Path(__file__).parent / "params_turn_left_mjx.py"

MAX_STEPS = 200
ACT_DIM   = 24
OBS_DIM   = 88
GAIT_PERIOD_STEPS = 40
PHYSICS_STEPS_PER_ACTION = int(_p.PHYSICS_STEPS_PER_ACTION)

# Sign convention: turn_left = +1 (rewards +angvel_z). turn_right reuses this
# module via a sign flag in params.
TURN_SIGN = +1.0

_params_mtime = 0.0


def read_params():
    global _params_mtime
    try:
        mtime = _PARAMS_FILE.stat().st_mtime
        if mtime != _params_mtime:
            importlib.reload(_p)
            _params_mtime = mtime
    except Exception as e:
        print(f"[params_turn_left_mjx] reload error: {e}")
    return (
        float(_p.W_YAW),
        float(_p.W_TRANSLATE),
        float(_p.W_ALIVE),
        float(_p.FLIP_PENALTY),
        float(_p.ACTION_SMOOTH_ALPHA),
        float(_p.W_CPG_PITCH),
        float(_p.W_CPG_ELBOW),
    )


def load_model():
    print("[turn_left_env] loading scene")
    _, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose = base_load_model(
        SCENE, _p, PHYSICS_STEPS_PER_ACTION
    )
    target_xy = jnp.zeros(2, dtype=jnp.float32)
    return mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose, target_xy


def reset(mx, jnt_lo, jnt_hi, target_xy, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    dx = mjx.make_data(mx)

    yaw = jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi)
    joint_noise = jax.random.uniform(k2, (24,), minval=-0.20, maxval=0.20)

    qpos = dx.qpos.at[2].set(0.30)
    qpos = qpos.at[3].set(jnp.cos(yaw / 2))
    qpos = qpos.at[6].set(jnp.sin(yaw / 2))
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
    (w_yaw, w_translate, w_alive, flip_penalty, action_alpha, w_cpg_pitch, w_cpg_elbow) = rparams

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
    angvel_z = dx.qvel[5]
    linvel_xy = dx.qvel[0:2]
    speed_xy_mag = jnp.linalg.norm(linvel_xy)

    # Reward — TURN_SIGN flips this for turn_right by mirroring.
    yaw_reward = w_yaw * TURN_SIGN * angvel_z
    upside_down = qw < 0
    alive_bonus = w_alive * (1.0 - upside_down.astype(jnp.float32))
    translation_penalty = w_translate * speed_xy_mag

    reward = yaw_reward + alive_bonus - translation_penalty - flip_penalty * upside_down.astype(jnp.float32)

    on_ground = torso_z < 0.05
    done = upside_down | on_ground

    return dx, obs, reward, done, motor_pos, motor_vel, filt_action, jnp.float32(angvel_z), step_idx


def is_keeper(metrics):
    """Promote when robot rotates left fast in place."""
    omega = metrics.get("mean_angvel_z", 0.0)
    speed = metrics.get("mean_speed", 0.0)  # mean_speed = ||linvel_xy||
    if omega > 0.4 and speed < 0.15:
        return True, f"wz{omega:.2f}_v{speed:.2f}"
    return False, ""
