# ─────────────────────────────────────────────────────────────────────────────
# Training hints — edit this file while training runs; changes are picked up
# automatically at the start of the next episode (no restart needed).
#
# Each entry is a reward term added on top of the base reward.
# Set weight to 0.0 to disable a hint without deleting it.
# ─────────────────────────────────────────────────────────────────────────────

HINTS = {
    # Goal: torso/chassis up high, gripper tips planted on floor —
    # classic quadruped stance, stable and tall.
    # ──────────────────────────────────────────────────────────────

    # Don't tip — strong penalty on torso tilt
    "upright":         0.5,

    # Wider stance = more stable base
    "spread_legs":     0.3,

    # Press gripper tips DOWN toward the floor — they are the "feet"
    "front_arms_down": 0.5,
    "back_arms_down":  0.5,

    # Discourage arm flailing
    "stillness":       0.05,

    # Off — symmetry kicks in naturally once all four grippers are down
    "symmetry":        0.0,
}
