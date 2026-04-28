"""MJX training parameters. Edit values and restart training."""


# ─── Motor & action shaping ──────────────────────────────────────────────────
# Hard cap on how far a joint can travel per action (normalised [-1, 1]).
#   0.05 → barely moves   0.15 → slow   0.30 → moderate   0.60+ → jerky
MOTOR_MAX_SPEED = 0.10

# Low-pass filter on policy output:  filt = α·raw + (1−α)·prev
#   1.0 → no filter   0.5 → light   0.3 → strong   0.1 → very heavy
ACTION_SMOOTH_ALPHA = 1.0

# ─── Goal shaping ────────────────────────────────────────────────────────────
# Target torso height in metres. Reward peaks here.
#   0.20 → crouching   0.30 → half-upright   0.45 → fully standing
TARGET_Z = 0.30

# Width (sigma, metres) of the Gaussian height-reward peak.
#   0.05 → tight   0.10 → forgiving   0.20 → very wide
TARGET_TOL = 0.10

# Scale of height reward — main objective signal.
#   5 → strong   10 → dominant
W_HEIGHT = 10.0

# Reward for keeping torso level (only when w > 0).
#   1 → mild   3 → strong   5 → dominant
W_UPRIGHT = 5.0

# ─── Arm safety ──────────────────────────────────────────────────────────────
# Distance (metres) at which arms incur a penalty for being too close.
ARM_SAFETY_MARGIN = 0.05

# Strength of arm-spacing penalty (quadratic inside the margin).
#   0 → off   10 → strong   20 → very strong
W_PROXIMITY = 20.0

# ─── Survival ────────────────────────────────────────────────────────────────
# Per-step bonus for not flipping.
#   0.05 → light   0.2 → strong (may farm this instead of learning)
W_ALIVE = 0.5

# One-time penalty on flip. Episode ends.
#   10 → mild   100 → strong   200 → avoid risk at all costs
FLIP_PENALTY = 200.0

# ─── Smoothness penalties ────────────────────────────────────────────────────
# Penalty on action change between steps.
#   0 → off   0.05 → light   0.5 → moderate   2.0 → heavy
W_SMOOTH = 0.5

# Penalty on joint velocity magnitude.
#   0 → off   0.001 → light   0.01 → moderate
W_JVEL = 0.5

# Penalty on total motor effort (ctrl²). Encourages energy efficiency.
#   0 → off   5e-4 → light   0.005 → moderate
W_ENERGY = 0.005

# ─── PPO algorithm ───────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF      = 0.05
CLIP_EPS      = 0.15

# ─── Physics (requires restart) ──────────────────────────────────────────────
# Physics steps per policy decision. Each step = 0.002 s.
#   16 → twitchy/fast   32 → fast feedback   64 → balanced   200 → slow/deliberate
# Higher = slower JIT compile and slower rollouts. Drop to 16-32 for quick iteration.
PHYSICS_STEPS_PER_ACTION = 32

# Joint damping (N·m·s/rad).  1.5–8 is realistic for ST32xx motors.
JOINT_DAMPING = 1.5

# Position-controller P-gain.  force = kp·(target − actual)
#   4 → soft   8 → balanced   20 → stiff
MOTOR_KP = 8.0
