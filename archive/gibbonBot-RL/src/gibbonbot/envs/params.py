# Hot-reloadable training parameters.
# Edit this file or use the watch.py UI — changes take effect within 1 episode.

# --- motor ---
MOTOR_MAX_SPEED = 0.200000  # max normalized joint movement per step
MAX_ACTIVE_JOINTS = 12  # no joint limit penalty

# --- timing ---
PHYSICS_STEPS_PER_ACTION = 240  # steps per policy action (250 × 0.002s = 0.5s)
REALTIME_FACTOR = 0.100000  # viewer speed: 1.0 = real-time, 0.5 = 2× faster, 2.0 = slow-mo


# --- goals ---
TARGET_Z = 0.3  # shoulder height target (10 cm off ground)
TARGET_TOL = 0.06  # Gaussian sigma
LIFTED_Z = 0.200000  # "got up" bonus threshold
MIN_HEIGHT = 0.03  # below this → fallen
STABLE_VEL = 0.37  # max joint velocity to count as "stable at goal"
W_STABLE = 1.0  # bonus when at target height AND stable

# --- reward weights ---
W_HEIGHT = 2.0  # Gaussian height reward scale
W_UPRIGHT = 0.3  # upright orientation reward
W_GOT_UP = 2.400000  # bonus when shoulder > LIFTED_Z
W_PLANT_LIFT = 0.3  # grippers on floor × torso rising
W_ALIVE = 1.000000  # per-step survival bonus
W_ENERGY = 0.001500  # penalty: sum(ctrl²)
W_JVEL = 1e-3  # penalty: joint velocity flailing
W_SMOOTH = 0.500000  # penalty: actual motor movement
W_DRAG = 0.5  # penalty: horizontal velocity while on floor
W_PROXIMITY = 4.500000  # penalty: arms too close to each other (soft, continuous)
W_JOINTS = 0.800000  # no joint count penalty
ARM_SAFETY_MARGIN = 0.200000  # soft margin — penalty ramps up inside this distance (6 cm)
ARM_TOUCH_DIST = 0.050000  # hard limit — episode ends if any cross-arm pair < 1 cm
ARM_COLLISION_PENALTY = 200.000000  # one-time penalty on arm collision termination
FLIP_PENALTY = 50.000000  # one-time penalty for flipping upside-down
