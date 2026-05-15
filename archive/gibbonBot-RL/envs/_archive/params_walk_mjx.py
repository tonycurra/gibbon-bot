"""Walk-to-tree training parameters. Edit values and restart training."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_MAX_SPEED = (
    0.10  # legacy — unused; per-joint limits derived from data/motors.md at env load
)
MOTOR_MAX_ACCEL = 0.025  # legacy — unused
# Smooth-motor defaults. Slower + gentler + heavier EMA = visible, not jerky.
# ACTION_SMOOTH_ALPHA is the EMA pole on policy output — LOWER = smoother.
# Past experiments converged on these values; anything higher than ~0.6
# alpha or 5.0 accel makes the limbs snap.
MOTOR_SPEED_FRAC = 0.4165
MOTOR_ACCEL_RATIO = 2.285
ACTION_SMOOTH_ALPHA = 0.5

# Central Pattern Generator baseline — built-in trot rhythm added to policy
# output before motor mapping. Lets policy learn corrections to a working gait
# instead of inventing gait from scratch. 0.0 disables.
W_CPG_PITCH = 0.0  # DISABLED - propulsive sweep amplitude
W_CPG_ELBOW = 0.0  # DISABLED - foot lift during swing
GAIT_TYPE = "pace"  # Gait pattern: pace, trot, walk, alt_arm_leg, mirror_arm_leg
FRACTAL_OFFSET = 0.0  # Phase offset (0.0 to 1.0) for additional variation
MOTOR_KP = 8.0

# ─── Goal: walk toward tree without dragging torso ───────────────────────────
TARGET_Z = 0.30  # desired torso height while walking
TARGET_TOL = 0.10  # Gaussian sigma around target_z
W_HEIGHT = 0.5      # standing pays minimal — don't anchor to "do nothing"
W_UPRIGHT = 3.0     # mild upright bonus

W_PROGRESS = 80.0   # main signal: meters closer to tree per step
W_FWD_VEL = 25.0  # 3x bumped — direct reward for moving toward tree dominates standing
W_PROXIMITY_TREE = 1.0  # smooth bonus near goal
W_HEADING = (
    8.0  # body forward axis aligned with tree direction — kills squirm/turn-in-place
)
SUCCESS_DIST = 0.6  # within this radius of tree → episode wins
W_SUCCESS = 200.0  # one-shot reward on success — make it the obvious winning move

# ─── Ground avoidance ────────────────────────────────────────────────────────
GROUND_CLEARANCE = 0.18  # torso must stay above this z
W_GROUND = 25.0  # quadratic penalty for dipping below

# ─── Survival ────────────────────────────────────────────────────────────────
W_ALIVE = 0.2
FLIP_PENALTY = 60.0  # discourages flips without crushing exploration

# ─── Smoothness ──────────────────────────────────────────────────────────────
W_SMOOTH = 0.30  # mild jerk penalty — encourages stable gait
W_JVEL = 0.05  # mild joint thrashing penalty (0.15 was crushing exploration)
W_SYMMETRY = 0.0  # off until policy walks. Then re-enable for polish.
W_ENERGY = 0.02  # dialed back from 0.10 — enough to prefer cheaper actions without preventing exploration

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.05  # low entropy — policy commits to its best strategy instead of randomly probing (was 0.20, too noisy at this stage)
CLIP_EPS = 0.15

# ─── Physics (requires restart) ──────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
