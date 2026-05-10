"""Walk-to-tree training parameters. Edit values and restart training."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_MAX_SPEED = 0.20      # slower per-step motor travel (still reaches full range over a few steps)
ACTION_SMOOTH_ALPHA = 0.4   # EMA on actions: 1.0=raw (jittery), <1 lowpass-filters commands
MOTOR_KP = 8.0

# ─── Goal: walk toward tree without dragging torso ───────────────────────────
TARGET_Z = 0.30           # desired torso height while walking
TARGET_TOL = 0.10         # Gaussian sigma around target_z
W_HEIGHT = 0.5            # cut hard — standing pays minimal. Was anchoring policy to "do nothing".
W_UPRIGHT = 3.0           # bonus when w > 0

W_PROGRESS = 80.0         # main signal: reward each meter closer per step
W_FWD_VEL = 25.0          # 3x bumped — direct reward for moving toward tree dominates standing
W_PROXIMITY_TREE = 1.0    # smooth bonus near goal
W_HEADING = 8.0           # body forward axis aligned with tree direction — kills squirm/turn-in-place
SUCCESS_DIST = 0.6        # within this radius of tree → episode wins
W_SUCCESS = 200.0         # one-shot reward on success — make it the obvious winning move

# ─── Ground avoidance ────────────────────────────────────────────────────────
GROUND_CLEARANCE = 0.18   # torso must stay above this z
W_GROUND = 25.0           # quadratic penalty for dipping below

# ─── Survival ────────────────────────────────────────────────────────────────
W_ALIVE = 0.2
FLIP_PENALTY = 30.0       # cut — falling is fine, exploration matters more

# ─── Smoothness ──────────────────────────────────────────────────────────────
W_SMOOTH = 0.15     # gentle penalty on jerky commands (heavy values crash reward)
W_JVEL = 0.02       # gentle damping on joint thrashing
W_SYMMETRY = 0.0    # off until policy walks. Then re-enable for polish.
W_ENERGY = 0.005

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.1            # moderate entropy — lets policy adapt to new smoothing
CLIP_EPS = 0.15

# ─── Physics (requires restart) ──────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
