"""Turn-right task hyperparameters. Same as turn_left but mirrored sign.
Implementation reuses turn_left_env_mjx.py with the W_YAW sign flipped via
the env's reading of TURN_SIGN."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 1.0
MOTOR_ACCEL_RATIO = 5.0
ACTION_SMOOTH_ALPHA = 0.30

# ─── Reward — minimal: yaw rate is the signal (negative target now) ──────────
W_YAW = 15.0                # absolute magnitude — sign flipped in env via TURN_SIGN
W_TRANSLATE = 5.0
W_ALIVE = 0.5
FLIP_PENALTY = 60.0

# ─── CPG disabled ────────────────────────────────────────────────────────────
W_CPG_PITCH = 0.0
W_CPG_ELBOW = 0.0

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
