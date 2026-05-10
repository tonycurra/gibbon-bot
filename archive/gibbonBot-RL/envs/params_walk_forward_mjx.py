"""Walk-forward task hyperparameters. Task = move body in +x direction
(no goal tree, just generic forward locomotion)."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 1.0
MOTOR_ACCEL_RATIO = 5.0
ACTION_SMOOTH_ALPHA = 0.30   # firmer lowpass than walk-to-tree (encourages smoother steps)

# ─── Reward — minimal: forward velocity is the only positive driver ──────────
W_FWD_VEL = 25.0             # primary signal: linvel_x (m/s)
W_ALIVE = 0.5                # per-step survival
FLIP_PENALTY = 60.0
GROUND_CLEARANCE = 0.18      # torso must stay above this z
W_GROUND = 25.0              # quadratic penalty for dipping below

# ─── CPG — pace gait baseline (forward locomotion needs rhythm) ──────────────
W_CPG_PITCH = 0.18
W_CPG_ELBOW = 0.10

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
