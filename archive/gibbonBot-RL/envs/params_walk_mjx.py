"""Walk-to-tree training parameters. Edit values and restart training."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_MAX_SPEED = 0.60
ACTION_SMOOTH_ALPHA = 1.0
MOTOR_KP = 8.0

# ─── Goal: walk toward tree without dragging torso ───────────────────────────
TARGET_Z = 0.30           # desired torso height while walking
TARGET_TOL = 0.10         # Gaussian sigma around target_z
W_HEIGHT = 4.0            # height-shaping weight (less dominant than progress)

W_UPRIGHT = 3.0           # bonus when w > 0

W_PROGRESS = 80.0         # main signal: reward each meter closer per step (bumped for fast walk)
W_PROXIMITY_TREE = 1.0    # smooth bonus near goal
SUCCESS_DIST = 0.6        # within this radius of tree → episode wins
W_SUCCESS = 50.0          # one-shot reward on success

# ─── Ground avoidance ────────────────────────────────────────────────────────
GROUND_CLEARANCE = 0.18   # torso must stay above this z
W_GROUND = 50.0           # quadratic penalty for dipping below

# ─── Survival ────────────────────────────────────────────────────────────────
W_ALIVE = 0.2
FLIP_PENALTY = 200.0

# ─── Smoothness ──────────────────────────────────────────────────────────────
W_SMOOTH = 0.1
W_JVEL = 0.005      # scaled down ~30× from the old (over-damped) motors
W_ENERGY = 0.005

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.05
CLIP_EPS = 0.15

# ─── Physics (requires restart) ──────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
