# ─────────────────────────────────────────────────────────────────────────────
# Training hints — edit this file while training runs; changes are picked up
# automatically at the start of the next episode (no restart needed).
#
# Each entry is a reward term added on top of the base reward.
# Set weight to 0.0 to disable a hint without deleting it.
# ─────────────────────────────────────────────────────────────────────────────

HINTS = {
    # Keep torso level — penalise tilt (roll/pitch)
    # Measured as 1 - quat_w^2; 0 = perfectly upright, 1 = fully tipped
    "upright": 0.0,

    # Reward spreading legs outward — encourages a wider, more stable base
    # Measured as mean absolute value of the 4 base-rotation joints
    "spread_legs": 0.0,

    # Reward keeping front arms planted — front gripper tips close to ground
    # Measured as penalty on front gripper z height
    "front_arms_down": 0.0,

    # Reward keeping back arms planted — same for rear grippers
    "back_arms_down": 0.0,

    # Reward symmetric left-right pose — discourages leaning to one side
    # Measured as mean abs difference between left and right joint angles
    "symmetry": 0.0,

    # Reward low joint velocity — encourages slow, deliberate movement
    # (stacks with the base velocity penalty; raise this to really slow it down)
    "stillness": 0.0,
}
