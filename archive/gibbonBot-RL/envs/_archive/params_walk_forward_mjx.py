"""Walk-forward task hyperparameters. Task = move body in +x direction
(no goal tree, just generic forward locomotion)."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 0.05
MOTOR_ACCEL_RATIO = 0.5
ACTION_SMOOTH_ALPHA = 0.975

# ─── Reward — minimal: forward velocity is the only positive driver ──────────
W_FWD_VEL = 25.0  # primary signal: linvel_x (m/s)
W_ALIVE = 0.5  # per-step survival
FLIP_PENALTY = 500.0
GROUND_CLEARANCE = 0.18  # torso must stay above this z
W_GROUND = 25.0  # quadratic penalty for dipping below

# ─── CPG — pace gait baseline (forward locomotion needs rhythm) ──────────────
W_CPG_PITCH = 0.18
W_CPG_ELBOW = 0.10
GAIT_TYPE = "pace"  # Gait pattern: pace, trot, walk, alt_arm_leg, mirror_arm_leg
FRACTAL_OFFSET = 0.0  # Phase offset (0.0 to 1.0) for additional variation

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
