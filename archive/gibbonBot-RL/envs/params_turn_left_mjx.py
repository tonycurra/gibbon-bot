"""Turn-left task hyperparameters. Task = body angular velocity around +z axis
(yaw rate positive), in place (no translation)."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 0.8
MOTOR_ACCEL_RATIO = 2.0
ACTION_SMOOTH_ALPHA = 0.10  # stronger lowpass for smoother movement

# ─── Reward — minimal: yaw rate is the signal ────────────────────────────────
W_YAW = 15.0  # primary signal: angvel_z (rad/s, positive=left)
W_TRANSLATE = 5.0  # penalty on linear xy velocity (don't translate, just rotate)
W_ALIVE = 0.5
FLIP_PENALTY = 60.0

# ─── CPG — can be enabled for rhythmic turning patterns ───────────────────────
W_CPG_PITCH = 0.0
W_CPG_ELBOW = 0.0
GAIT_TYPE = "pace"  # Gait pattern: pace, trot, walk, alt_arm_leg, mirror_arm_leg
FRACTAL_OFFSET = 0.0  # Phase offset (0.0 to 1.0) for additional variation

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
