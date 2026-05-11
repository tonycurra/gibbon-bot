"""MJX environment — walk forward. Translate the body in +x direction.
No tree, no spatial goal — just generic forward locomotion. The robot's
heading at spawn defines what "forward" means in world coordinates: we
spawn at origin facing +x, so linvel_x is the propulsion signal.

Reward = fwd_vel + alive − flip − ground_dip. Pace CPG enabled.
"""

import importlib
import pathlib

import jax
import jax.numpy as jnp

import envs.params_walk_forward_mjx as _p
from envs._shared_dynamics import apply_action, cpg_gait_offset, cpg_pace_offset
from envs._shared_load_model import base_load_model
from envs._shared_obs import assemble_obs
from envs._shared_physics_backend import mjx

_ROOT = pathlib.Path(__file__).parent.parent
SCENE = _ROOT / "data" / "scene.xml"
_PARAMS_FILE = pathlib.Path(__file__).parent / "params_walk_forward_mjx.py"

MAX_STEPS = 200
ACT_DIM = 24
OBS_DIM = 88
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
        print(f"[params_walk_forward_mjx] reload error: {e}")
    return (
        float(_p.W_FWD_VEL),
        float(_p.W_ALIVE),
        float(_p.FLIP_PENALTY),
        float(_p.GROUND_CLEARANCE),
        float(_p.W_GROUND),
        float(_p.ACTION_SMOOTH_ALPHA),
        float(_p.W_CPG_PITCH),
        float(_p.W_CPG_ELBOW),
        float(_p.FRACTAL_OFFSET),
    )


def load_model():
    print("[walk_forward_env] loading scene")
    _, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose = (
        base_load_model(SCENE, _p, PHYSICS_STEPS_PER_ACTION)
    )
    # No spatial target. We pass a far +x point so the obs target_dx slot
    # consistently signals "forward is +x" — useful for the policy even though
    # we never reward distance reduction.
    target_xy = jnp.array([10.0, 0.0], dtype=jnp.float32)
    return mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose, target_xy


def reset(mx, jnt_lo, jnt_hi, target_xy, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    dx = mjx.make_data(mx)

    # Spawn at origin, facing roughly +x with a small yaw jitter so the policy
    # learns slight orientation correction. Big yaw randomization defeats the
    # purpose of "forward" — keep ±30°.
    yaw = jax.random.uniform(k1, (), minval=-jnp.pi / 6, maxval=jnp.pi / 6)
    joint_noise = jax.random.uniform(k2, (24,), minval=-0.20, maxval=0.20)

    qpos = dx.qpos.at[2].set(0.30)
    qpos = qpos.at[3].set(jnp.cos(yaw / 2))
    qpos = qpos.at[6].set(jnp.sin(yaw / 2))
    qpos = qpos.at[7 : 7 + ACT_DIM].set(joint_noise)

    dx = dx.replace(
        qpos=qpos, qvel=jnp.zeros_like(dx.qvel), ctrl=jnp.zeros_like(dx.ctrl)
    )
    dx = mjx.forward(mx, dx)

    motor_pos = jnp.zeros(ACT_DIM)
    motor_vel = jnp.zeros(ACT_DIM)
    prev_action = jnp.zeros(ACT_DIM)
    step_idx = jnp.int32(0)
    obs = assemble_obs(dx, target_xy, step_idx, GAIT_PERIOD_STEPS, ACT_DIM)
    prev_scalar = jnp.float32(0.0)
    return dx, obs, motor_pos, motor_vel, prev_action, prev_scalar, step_idx


def step(
    mx,
    jnt_lo,
    jnt_hi,
    motor_max_speed,
    motor_max_accel,
    rest_pose,
    target_xy,
    dx,
    action,
    motor_pos,
    motor_vel,
    prev_action,
    prev_scalar,
    step_idx,
    rparams,
):
    (
        w_fwd_vel,
        w_alive,
        flip_penalty,
        ground_clearance,
        w_ground,
        action_alpha,
        w_cpg_pitch,
        w_cpg_elbow,
        fractal_offset,
    ) = rparams

    filt_action = action_alpha * action + (1.0 - action_alpha) * prev_action
    cpg = cpg_gait_offset(
        step_idx,
        GAIT_PERIOD_STEPS,
        w_cpg_pitch,
        w_cpg_elbow,
        ACT_DIM,
        gait_type=_p.GAIT_TYPE,
        fractal_offset=fractal_offset,
    )
    target = filt_action + cpg

    dx, motor_pos, motor_vel = apply_action(
        mx,
        jnt_lo,
        jnt_hi,
        rest_pose,
        motor_max_speed,
        motor_max_accel,
        PHYSICS_STEPS_PER_ACTION,
        dx,
        motor_pos,
        motor_vel,
        target,
    )

    step_idx = step_idx + 1
    obs = assemble_obs(dx, target_xy, step_idx, GAIT_PERIOD_STEPS, ACT_DIM)

    torso_z = dx.qpos[2]
    qw = dx.qpos[3]
    linvel_x = dx.qvel[0]

    # Reward.
    fwd_vel_reward = w_fwd_vel * jnp.maximum(linvel_x, 0.0)
    upside_down = qw < 0
    alive_bonus = w_alive * (1.0 - upside_down.astype(jnp.float32))
    ground_violation = jnp.maximum(ground_clearance - torso_z, 0.0)
    ground_penalty = w_ground * ground_violation**2

    reward = (
        fwd_vel_reward
        + alive_bonus
        - ground_penalty
        - flip_penalty * upside_down.astype(jnp.float32)
    )

    on_ground = torso_z < (ground_clearance * 0.5)
    done = upside_down | on_ground

    return (
        dx,
        obs,
        reward,
        done,
        motor_pos,
        motor_vel,
        filt_action,
        jnp.float32(linvel_x),
        step_idx,
    )


def is_keeper(metrics):
    """Promote when robot sustains forward speed."""
    sx = metrics.get("mean_speed_x", 0.0)
    if sx > 0.20:
        return True, f"vx{sx:.2f}"
    return False, ""
