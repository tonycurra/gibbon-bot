"""Stand task hyperparameters. Task = hold body upright at z≈0.30m."""

# ─── Motor & action shaping (shared with all tasks) ──────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 1.0
MOTOR_ACCEL_RATIO = 5.0
ACTION_SMOOTH_ALPHA = 0.40   # moderate lowpass

# ─── Reward shaping (minimal — one signal: stay upright at target height) ────
TARGET_Z = 0.30
TARGET_TOL = 0.10           # σ on height Gaussian
W_UPRIGHT = 5.0             # primary signal: torso z-axis aligned with world up
W_HEIGHT = 2.0              # secondary: stay near 0.30m
W_ALIVE = 0.5               # per-step survival
FLIP_PENALTY = 60.0         # one-shot at flip

# CPG disabled — standing doesn't need a gait rhythm.
W_CPG_PITCH = 0.0
W_CPG_ELBOW = 0.0

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
