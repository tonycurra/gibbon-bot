"""MJX training parameters — edit values below.

Two top-level groups:
  1. RESTART REQUIRED  — baked into JIT/MJX model. Kill train_mjx.py and
                         re-launch for changes to take effect.
  2. HOT-RELOAD        — picked up at the start of the next rollout (a few
                         seconds). No restart needed.

Within each group, params are subdivided by purpose. Comments above each
parameter explain ranges and trade-offs.
"""


# ════════════════════════════════════════════════════════════════════════════
# 2. HOT-RELOAD
#    Changes are picked up at the start of the next rollout — save the file,
#    wait a few seconds, the next rollout uses the new values.
# ════════════════════════════════════════════════════════════════════════════

# ─── Motor & action shaping ─────────────────────────────────────────────────
# Hard cap on how far a joint can travel per action (normalised [-1, 1]).
#   0.05 → barely moves       |  0.30 → moderate
#   0.15 → slow, controlled   |  0.60+ → jerky, robot can fling itself
MOTOR_MAX_SPEED = 0.10

# Low-pass filter applied to policy output before motors:
#     filt = α · raw + (1 − α) · prev_filt
#   1.0 → no filter        |  0.3 → strong smoothing
#   0.5 → light smoothing  |  0.1 → very heavy, slow reactions
# Lower this FIRST if the robot twitches — forces smoothness without retraining.
ACTION_SMOOTH_ALPHA = 0.3

# ─── Goal shaping ───────────────────────────────────────────────────────────
# Target torso height in metres. Reward peaks here.
#   0.10 → nearly lying down (very easy)   |  0.30 → half-upright (moderate)
#   0.20 → crouching low (good early)      |  0.45 → fully standing (hard)
# Curriculum tip: start LOW, raise as the policy succeeds.
TARGET_Z = 0.30

# Width (sigma, metres) of the Gaussian height-reward peak.
#   0.01 → very tight (hard)      |  0.10 → wide and forgiving (easy)
#   0.05 → moderate               |  0.20 → very wide
TARGET_TOL = 0.10

# Scale of the height-reward Gaussian — the main objective signal.
#   0    → height doesn't matter   |  5  → strong (typical)
#   2    → mild pull               |  10 → dominant, overrides other rewards
W_HEIGHT = 10.0

# Reward for keeping torso level: w · w_quat² (only when w > 0).
#   0   → leaning is fine          |  3 → strong
#   1   → mild encouragement       |  5 → dominant, prioritises orientation
W_UPRIGHT = 5.0

# ─── Arm safety ─────────────────────────────────────────────────────────────
# Distance (metres) at which arms incur a penalty for being too close.
#   0.05 → tight (almost touching ok)   |  0.20 → generous spacing
#   0.40 → very generous, forced wide
ARM_SAFETY_MARGIN = 0.05

# Strength of arm-spacing penalty (quadratic inside the margin).
#    0  → off, arms can touch       |  10 → strong, arms repel
#    3  → moderate                  |  20 → very strong, wide spacing always
W_PROXIMITY = 20.0

# ─── Survival ───────────────────────────────────────────────────────────────
# Per-step bonus for not flipping. Counters early give-up behaviour.
#   0    → no bonus, episodes end with no payoff
#   0.05 → light (good default)
#   0.2  → strong (robot may sit there forever to farm reward)
W_ALIVE = 0.5

# One-time penalty applied when torso flips (quat w < 0). Episode ends.
#    10 → mild     |  100 → strong     |  200 → very strong (avoid risky moves)
FLIP_PENALTY = 200.0

# ─── Smoothness penalties ───────────────────────────────────────────────────
# Reward shaping for non-twitchy, low-energy motion. Set any to 0 to disable.

# Penalty on action change between steps:  w · sum((filt − prev_filt)²)
#   0    → policy free to twitch
#   0.05 → light, gentle nudge toward consistency
#   0.5  → moderate
#   2.0  → heavy, controlled but may slow learning
# Pairs with ACTION_SMOOTH_ALPHA: ALPHA prevents twitch in motion *now*,
# W_SMOOTH teaches the policy to not *want* to twitch.
W_SMOOTH = 2.0

# Penalty on joint velocity magnitude:  w · sum(qvel²)
#   0      → no damping, limbs spin freely
#   0.001  → light (good default)
#   0.01   → moderate
#   0.05+  → heavy, joints feel sluggish
W_JVEL = 0.5

# Penalty on total motor effort:  w · sum(ctrl²)
#   0      → no penalty, motors run at any torque
#   5e-4   → light, encourages efficiency
#   0.005  → moderate
#   0.05   → heavy, robot barely actuates
# Useful for sim-to-real (real motors burn out at full power).
W_ENERGY = 0.05

# ─── Learning (PPO algorithm) ───────────────────────────────────────────────
# Rarely touched — only for fixing instability or stuck training.

# Adam optimiser step size — how fast network weights change.
#   1e-5 → very slow, ultra-stable
#   3e-4 → standard PPO default
#   9e-4 → fast (good early training, may oscillate later)
#   1e-3 → very fast, often unstable
LEARNING_RATE = 0.00037

# Entropy bonus — rewards trying varied actions.
#   0.0  → pure exploitation         |  0.02 → moderate (good early)
#   5e-3 → mild (good late training) |  0.05 → heavy exploration
ENT_COEF = 0.05

# PPO clip range — how much the policy can change per update.
#   0.1 → very conservative   |  0.3 → aggressive
#   0.2 → standard            |  0.5 → can collapse training
CLIP_EPS = 0.15


# ════════════════════════════════════════════════════════════════════════════
# 1. RESTART REQUIRED
#    Changes here only take effect when training is re-launched.
# ════════════════════════════════════════════════════════════════════════════

# ─── Timing ─────────────────────────────────────────────────────────────────
# Physics steps per policy decision. Each physics step = 0.002 s.
#    16  → 0.032 s/decision  — twitchy, fast training, less realistic
#    64  → 0.128 s/decision  — balanced
#   200  → 0.400 s/decision  — slow, deliberate, very smooth
# Slow physics ≠ slow training: more steps per action means each rollout covers
# more sim-time, so the policy sees longer-horizon consequences per gradient step.
PHYSICS_STEPS_PER_ACTION = 200

# ─── Hardware (overrides scene.xml) ─────────────────────────────────────────
# Joint friction in N·m·s/rad. Resists motion proportional to velocity.
#   0    → frictionless (oscillates)   |  4   → default
#   1.5  → realistic for ST32xx motors |  10+ → sluggish
# Real ST3215/3235 have significant internal friction from 1:345 reduction;
# 1.5–8 is the realistic range. Higher masks instability but caps useful speed.
JOINT_DAMPING = 4.0

# Position-controller stiffness (P-gain).  force = kp · (target − actual)
#    1 → very compliant     |   8 → default (balanced)
#    4 → soft, smooth       |  20 → stiff, fast, can overshoot
# With torque cap 2.94 N·m, kp=8 reaches full torque at ≈21° error.
MOTOR_KP = 8.0