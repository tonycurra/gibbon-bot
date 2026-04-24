# ─────────────────────────────────────────────────────────────────────────────
# Training hints — edit this file while training runs; changes are picked up
# automatically at the start of the next episode (no restart needed).
#
# Each entry is a reward term added on top of the base reward.
# Set weight to 0.0 to disable a hint without deleting it.
# ─────────────────────────────────────────────────────────────────────────────

HINTS = {
    # Goal: high + stable torso, no tipping, no arm flailing
    # ──────────────────────────────────────────────────────

    # Don't tip — strong penalty on torso tilt
    "upright":     0.5,

    # Wider stance = more stable base
    "spread_legs": 0.3,

    # Discourage arm flailing (joint-velocity penalty on top of the base term)
    "stillness":   0.05,

    # Off — would push grippers to floor; we don't want that, we want the
    # torso pushed UP, not the arms pinned DOWN
    "front_arms_down": 0.0,
    "back_arms_down":  0.0,

    # Off — symmetry kicks in once it can stand; turning it on too early
    # discourages exploration of asymmetric push-up moves
    "symmetry": 0.0,
}
