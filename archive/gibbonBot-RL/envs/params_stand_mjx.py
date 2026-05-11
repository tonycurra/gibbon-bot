"""Stand task hyperparameters. Task = hold body upright at z≈0.45m on ankles."""

# ─── Motor & action shaping (shared with all tasks) ──────────────────────────
MOTOR_KP = 8.0
MOTOR_SPEED_FRAC = 0.8
MOTOR_ACCEL_RATIO = 2.0
ACTION_SMOOTH_ALPHA = 0.20  # stronger lowpass for smoother movement

# ─── Reward shaping ────────────────────────────────────────────────────────
TARGET_Z = 0.45
TARGET_TOL = 0.08  # tighter tolerance for higher standing
W_UPRIGHT = 5.0  # primary signal: torso z-axis aligned with world up
W_HEIGHT = 2.0  # secondary: stay near 0.45m
W_ALIVE = 0.5  # per-step survival
FLIP_PENALTY = 60.0  # one-shot at flip
W_STABLE_Z = 1.0  # penalty on |torso vertical velocity| (stops oscillation)
W_ACTION = 0.01  # penalty on action magnitude (minimal motor effort)

# ─── Contact rewards (stand on ankles/feet, NOT knees) ──────────────────────
W_FOOT_CONTACT = 3.0  # reward for foot/ankle ground contact
W_KNEE_PENALTY = 5.0  # penalty for knee ground contact (force standing on ankles)

# ─── Symmetry rewards ──────────────────────────────────────────────────────
W_SYMMETRY = 2.0  # penalty for lateral lean (torso roll) - promotes upright symmetry

# CPG for standing — enables rhythmic limb movement for stability
# Gait types: "pace", "trot", "walk", "alt_arm_leg", "mirror_arm_leg"
# Set GAIT_TYPE in env to change pattern
W_CPG_PITCH = 0.0  # DISABLED - pitch/yaw amplitude for limb oscillation
W_CPG_ELBOW = 0.0  # DISABLED - elbow/knee lift amplitude
GAIT_TYPE = "alt_arm_leg"  # Alternating: FL arm → FR arm → BL leg → BR leg
FRACTAL_OFFSET = 0.0  # Phase offset (0.0 to 1.0) for additional variation

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.10
CLIP_EPS = 0.15

# ─── Physics ─────────────────────────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
