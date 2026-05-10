"""Shared action-application dynamics: CPG offset + trapezoidal motor profile +
substep loop. Used by every task env so a fix lands in one place."""
import jax
import jax.numpy as jnp
import mujoco.mjx as mjx


# Pace gait: same-side limb pairs in phase. FL+BL together (left side),
# FR+BR together (right side, π apart from left).
PACE_LIMB_PHASE = jnp.array([0.0, jnp.pi, 0.0, jnp.pi])


def cpg_pace_offset(step_idx, gait_period_steps, w_cpg_pitch, w_cpg_elbow, act_dim=24):
    """Calibrated pace gait CPG offset. Returns 24-dim action delta.

    Per-limb sign correction (from scripts/calibrate_cpg.py):
    - Front limbs propel via yaw with mirrored signs (FL=-, FR=+)
    - Back limbs propel via pitch with same sign (BL=+, BR=+)
    - Lift component on pitch (front) / knee (back), gated to swing phase only.

    Set w_cpg_pitch=0 and w_cpg_elbow=0 to disable (returns zeros).
    """
    phase = 2.0 * jnp.pi * step_idx / gait_period_steps + PACE_LIMB_PHASE
    sin_p = jnp.sin(phase)

    cpg = jnp.zeros(act_dim)
    # FL: yaw forward (negative direction) + pitch lift (negative) during swing
    cpg = cpg.at[0].set(-w_cpg_pitch * sin_p[0])
    cpg = cpg.at[1].set(-w_cpg_elbow * jnp.maximum(sin_p[0], 0.0))
    # FR: yaw forward (positive direction) + pitch lift (negative) during swing
    cpg = cpg.at[6].set(+w_cpg_pitch * sin_p[1])
    cpg = cpg.at[7].set(-w_cpg_elbow * jnp.maximum(sin_p[1], 0.0))
    # BL: pitch forward (positive) + knee lift (positive) during swing
    cpg = cpg.at[13].set(+w_cpg_pitch * sin_p[2])
    cpg = cpg.at[14].set(+w_cpg_elbow * jnp.maximum(sin_p[2], 0.0))
    # BR: pitch forward (positive) + knee lift (positive) during swing
    cpg = cpg.at[19].set(+w_cpg_pitch * sin_p[3])
    cpg = cpg.at[20].set(+w_cpg_elbow * jnp.maximum(sin_p[3], 0.0))
    return cpg


def apply_action(mx, jnt_lo, jnt_hi, rest_pose, motor_max_speed, motor_max_accel,
                 physics_steps_per_action, dx, motor_pos, motor_vel, target_action):
    """Run physics_steps_per_action substeps with trapezoidal motor profile.

    target_action is the (already EMA-filtered + CPG-augmented) policy target.
    Returns: (dx, motor_pos, motor_vel) after the action step.
    """
    substep_speed = motor_max_speed / physics_steps_per_action
    substep_accel = motor_max_accel / physics_steps_per_action

    def substep(carry, _):
        mpos, mvel, d = carry
        # Trapezoidal motion profile: bound velocity AND change in velocity.
        err = target_action - mpos
        desired_vel = jnp.clip(err, -substep_speed, substep_speed)
        new_vel = jnp.clip(desired_vel, mvel - substep_accel, mvel + substep_accel)
        new_vel = jnp.clip(new_vel, -substep_speed, substep_speed)
        mpos = mpos + new_vel
        # Mapping: action=0 → ctrl=rest_pose. action=±1 → jnt_hi / jnt_lo.
        ctrl = jnp.where(
            mpos >= 0,
            rest_pose + mpos * (jnt_hi - rest_pose),
            rest_pose + mpos * (rest_pose - jnt_lo),
        )
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mx, d)
        return (mpos, new_vel, d), None

    (motor_pos, motor_vel, dx), _ = jax.lax.scan(
        substep, (motor_pos, motor_vel, dx), None, length=physics_steps_per_action
    )
    return dx, motor_pos, motor_vel
