"""Shared model loader for all tasks. Tasks wrap this with task-specific extras
(e.g. tree position lookup for walk_to_tree). All tasks share the same robot
URDF, motor calibration, and rest pose."""

import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

# Per-limb joint order: yaw, pitch, elbow/knee, roll, wrist/ankle, gripper/foot.
# ST3235 / ST3215: 45 RPM = 4.71 rad/s. ST3215-HS (gripper/foot): 106 RPM = 11.10 rad/s.
PER_LIMB_MAX_RAD_PER_SEC = [4.71, 4.71, 4.71, 4.71, 4.71, 11.10]

# Rest pose (action=0 → these joint angles). Knees/elbows pre-bent for
# weight-bearing; everything else neutral.
DEG = jnp.pi / 180.0
PER_LIMB_REST = jnp.array([0.0, 0.0, 30.0 * DEG, 0.0, 0.0, 0.0])


def base_load_model(scene_path, params_module, physics_steps_per_action=16):
    """Common load_model for all tasks.

    Returns: (mj_model, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose)

    Tasks that need extras (e.g. tree position) wrap this and add to the tuple.
    """
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_model.opt.timestep = 0.002

    # Collision filtering: move all robot geoms to group 2 (contype=2, conaffinity=1).
    # This prevents robot self-collisions (legs crossing, arms hitting torso, etc.)
    # while maintaining collisions with the floor and trees (which stay in group 1).
    # Group 2 geoms: contype=2, conaffinity=1
    # Group 1 geoms (floor, trees): contype=1, conaffinity=31
    # Robot vs Floor: (2 & 31) != 0 and (1 & 1) != 0 → collide ✓
    # Robot vs Robot: (2 & 1) = 0 → no collision ✓
    # Robot vs Tree: (2 & 31) != 0 and (1 & 1) != 0 → collide ✓
    for i in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        # Keep floor geom in group 1 (for collision with robot)
        if geom_name and "floor" in geom_name:
            continue
        # Keep tree geoms (t0_*, t1_*, etc.) in group 1 so robot collides with them
        if (
            geom_name
            and geom_name.startswith("t")
            and geom_name[1:].split("_")[0].isdigit()
        ):
            continue
        # Move all other geoms (robot parts, terrain visuals) to group 2
        mj_model.geom_contype[i] = 2
        mj_model.geom_conaffinity[i] = 1

    kp = float(getattr(params_module, "MOTOR_KP", 8.0))
    mj_model.actuator_gainprm[:, 0] = kp
    mj_model.actuator_biasprm[:, 1] = -kp

    mx = mjx.put_model(mj_model)
    jnt_lo = jnp.array([mj_model.jnt_range[i][0] for i in range(1, mj_model.njnt)])
    jnt_hi = jnp.array([mj_model.jnt_range[i][1] for i in range(1, mj_model.njnt)])

    motor_max_radps = jnp.array(PER_LIMB_MAX_RAD_PER_SEC * 4, dtype=jnp.float32)
    rest_pose = jnp.tile(PER_LIMB_REST, 4)

    speed_frac = float(getattr(params_module, "MOTOR_SPEED_FRAC", 1.0))
    accel_ratio = float(getattr(params_module, "MOTOR_ACCEL_RATIO", 5.0))

    action_dt = mj_model.opt.timestep * physics_steps_per_action
    # Asymmetric joint ranges around rest pose: half_range is the larger of
    # (jnt_hi - rest) and (rest - jnt_lo) so action=±1 still reaches a limit.
    half_range = jnp.maximum(jnp.abs(jnt_hi - rest_pose), jnp.abs(rest_pose - jnt_lo))
    motor_max_speed = speed_frac * motor_max_radps * action_dt / half_range
    motor_max_accel = motor_max_speed * accel_ratio * action_dt

    return mj_model, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose
