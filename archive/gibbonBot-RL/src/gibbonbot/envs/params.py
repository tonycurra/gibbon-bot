# Hot-reloadable training parameters.
# Edit this file or use the watch.py UI — changes take effect within 1 episode.

# --- motor ---
MOTOR_MAX_SPEED   = 0.015   # max normalized joint movement per step
MAX_ACTIVE_JOINTS = 24      # no joint limit penalty

# --- timing ---
PHYSICS_STEPS_PER_ACTION = 210  # steps per policy action (250 × 0.002s = 0.5s)

# --- goals ---
TARGET_Z   = 0.10   # shoulder height target (10 cm off ground)
TARGET_TOL = 0.02   # Gaussian sigma
LIFTED_Z   = 0.07   # "got up" bonus threshold
MIN_HEIGHT = 0.03   # below this → fallen
STABLE_VEL = 0.05   # max joint velocity to count as "stable at goal"
W_STABLE   = 1.0    # bonus when at target height AND stable

# --- reward weights ---
W_HEIGHT    = 2.0   # Gaussian height reward scale
W_UPRIGHT   = 0.3   # upright orientation reward
W_GOT_UP    = 0.5   # bonus when shoulder > LIFTED_Z
W_PLANT_LIFT = 0.3  # grippers on floor × torso rising
W_ALIVE     = 0.05  # per-step survival bonus
W_ENERGY    = 5e-4  # penalty: sum(ctrl²)
W_JVEL      = 1e-3  # penalty: joint velocity flailing
W_SMOOTH    = 5e-2  # penalty: actual motor movement
W_DRAG      = 0.5   # penalty: horizontal velocity while on floor
W_PROXIMITY = 2.0   # penalty: arms too close to each other
W_JOINTS    = 0.0   # no joint count penalty
ARM_SAFETY_MARGIN = 0.06   # min distance between cross-arm bodies (m)
FLIP_PENALTY = 10.0  # one-time penalty for flipping upside-down
