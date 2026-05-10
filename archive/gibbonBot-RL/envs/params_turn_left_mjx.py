"""Turn-left task hyperparameters. Task = body angular velocity around +z axis
(yaw rate positive), in place (no translation)."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 1.0
MOTOR_ACCEL_RATIO = 5.0
ACTION_SMOOTH_ALPHA = 0.30

# ─── Reward — minimal: yaw rate is the signal ────────────────────────────────
W_YAW = 15.0                # primary signal: angvel_z (rad/s, positive=left)
W_TRANSLATE = 5.0           # penalty on linear xy velocity (don't translate, just rotate)
W_ALIVE = 0.5
FLIP_PENALTY = 60.0

# ─── CPG disabled — turning isn't a gait ─────────────────────────────────────
W_CPG_PITCH = 0.0
W_CPG_ELBOW = 0.0

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
