"""Shared action-application dynamics: CPG offset + trapezoidal motor profile +
substep loop. Used by every task env so a fix lands in one place."""

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx

# =============================================================================
# CPG GAIT PATTERNS
# =============================================================================

# Limb joint index offsets (each limb has pitch then elbow/knee joint)
# FL: joints 0(pitch), 1(elbow)   - front left
# FR: joints 6(pitch), 7(elbow)   - front right
# BL: joints 13(pitch), 14(knee)  - back left
# BR: joints 19(pitch), 20(knee)  - back right

# Pace gait: same-side limb pairs in phase. FL+BL together (left side),
# FR+BR together (right side, π apart from left).
PACE_LIMB_PHASE = jnp.array([0.0, jnp.pi, 0.0, jnp.pi])

# Trot gait: diagonal limb pairs. FL+BR together, FR+BL together (π apart).
TROT_LIMB_PHASE = jnp.array([0.0, jnp.pi, jnp.pi, 0.0])

# Walk gait: FL, BR, FR, BL sequence (each limb π/2 apart).
WALK_LIMB_PHASE = jnp.array([0.0, jnp.pi * 1.5, jnp.pi, jnp.pi * 0.5])

# Alternating/fractal: FL arm, FR arm, BL leg, BR leg (sequential)
ALT_ARM_LEG_PHASE = jnp.array([0.0, jnp.pi * 0.5, jnp.pi, jnp.pi * 1.5])

# Asymmetric mirror: FL arm, then FR arm (π apart), BL leg, then BR leg (π apart)
MIRROR_ARM_LEG_PHASE = jnp.array([0.0, jnp.pi, 0.0, jnp.pi])


# Per-limb sign correction for forward propulsion
# Front limbs: yaw (joint 0 for FL, joint 6 for FR)
# Back limbs: pitch (joint 13 for BL, joint 19 for BR)
LIMB_SIGN_PITCH = jnp.array([-1.0, +1.0, +1.0, +1.0])  # FL, FR, BL, BR
LIMB_SIGN_ELBOW = jnp.array([-1.0, -1.0, +1.0, +1.0])  # FL, FR, BL, BR


def cpg_gait_offset(
    step_idx,
    gait_period_steps,
    w_cpg_pitch,
    w_cpg_elbow,
    act_dim=24,
    gait_type="pace",
    fractal_offset=0.0,
):
    """Generalized CPG with multiple gait patterns. Returns 24-dim action delta.

    Args:
        step_idx: current step in episode
        gait_period_steps: number of steps per gait cycle
        w_cpg_pitch: weight for pitch (forward/backward) component
        w_cpg_elbow: weight for elbow/knee lift component
        act_dim: action dimension (24)
        gait_type: one of "pace", "trot", "walk", "alt_arm_leg", "mirror_arm_leg"
        fractal_offset: phase offset for fractal patterns (0.0 to 1.0)

    Lift component (elbow/knee) is gated to swing phase only (sin > 0).
    Set w_cpg_pitch=0 and w_cpg_elbow=0 to disable (returns zeros).
    """
    # Select phase offsets based on gait type
    if gait_type == "pace":
        phases = PACE_LIMB_PHASE
    elif gait_type == "trot":
        phases = TROT_LIMB_PHASE
    elif gait_type == "walk":
        phases = WALK_LIMB_PHASE
    elif gait_type == "alt_arm_leg":
        phases = ALT_ARM_LEG_PHASE
    elif gait_type == "mirror_arm_leg":
        phases = MIRROR_ARM_LEG_PHASE
    else:
        phases = PACE_LIMB_PHASE  # default to pace

    # Apply fractal offset (shifts all phases by a fraction of the cycle)
    phases = phases + fractal_offset * 2.0 * jnp.pi

    phase = 2.0 * jnp.pi * step_idx / gait_period_steps + phases
    sin_p = jnp.sin(phase)

    cpg = jnp.zeros(act_dim)
    # FL limb (joints 0=pitch/yaw, 1=elbow)
    cpg = cpg.at[0].set(LIMB_SIGN_PITCH[0] * w_cpg_pitch * sin_p[0])
    cpg = cpg.at[1].set(LIMB_SIGN_ELBOW[0] * w_cpg_elbow * jnp.maximum(sin_p[0], 0.0))
    # FR limb (joints 6=pitch/yaw, 7=elbow)
    cpg = cpg.at[6].set(LIMB_SIGN_PITCH[1] * w_cpg_pitch * sin_p[1])
    cpg = cpg.at[7].set(LIMB_SIGN_ELBOW[1] * w_cpg_elbow * jnp.maximum(sin_p[1], 0.0))
    # BL limb (joints 13=pitch, 14=knee)
    cpg = cpg.at[13].set(LIMB_SIGN_PITCH[2] * w_cpg_pitch * sin_p[2])
    cpg = cpg.at[14].set(LIMB_SIGN_ELBOW[2] * w_cpg_elbow * jnp.maximum(sin_p[2], 0.0))
    # BR limb (joints 19=pitch, 20=knee)
    cpg = cpg.at[19].set(LIMB_SIGN_PITCH[3] * w_cpg_pitch * sin_p[3])
    cpg = cpg.at[20].set(LIMB_SIGN_ELBOW[3] * w_cpg_elbow * jnp.maximum(sin_p[3], 0.0))
    return cpg


# Backwards compatibility alias
cpg_pace_offset = (
    lambda step_idx, gait_period_steps, w_cpg_pitch, w_cpg_elbow, act_dim=24: (
        cpg_gait_offset(
            step_idx,
            gait_period_steps,
            w_cpg_pitch,
            w_cpg_elbow,
            act_dim,
            gait_type="pace",
        )
    )
)


def apply_action(
    mx,
    jnt_lo,
    jnt_hi,
    rest_pose,
    motor_max_speed,
    motor_max_accel,
    physics_steps_per_action,
    dx,
    motor_pos,
    motor_vel,
    target_action,
):
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
