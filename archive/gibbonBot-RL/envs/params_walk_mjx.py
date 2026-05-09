"""Walk-to-tree training parameters. Edit values and restart training."""

# ─── Motor & action shaping ──────────────────────────────────────────────────
MOTOR_MAX_SPEED = 0.60
ACTION_SMOOTH_ALPHA = 1.0
MOTOR_KP = 8.0

# ─── Goal: walk toward tree without dragging torso ───────────────────────────
TARGET_Z = 0.30           # desired torso height while walking
TARGET_TOL = 0.10         # Gaussian sigma around target_z
W_HEIGHT = 0.5            # cut hard — standing pays minimal. Was anchoring policy to "do nothing".
W_UPRIGHT = 3.0           # bonus when w > 0

W_PROGRESS = 80.0         # main signal: reward each meter closer per step
W_FWD_VEL = 25.0          # 3x bumped — direct reward for moving toward tree dominates standing
W_PROXIMITY_TREE = 1.0    # smooth bonus near goal
SUCCESS_DIST = 0.6        # within this radius of tree → episode wins
W_SUCCESS = 200.0         # one-shot reward on success — make it the obvious winning move

# ─── Ground avoidance ────────────────────────────────────────────────────────
GROUND_CLEARANCE = 0.18   # torso must stay above this z
W_GROUND = 25.0           # quadratic penalty for dipping below

# ─── Survival ────────────────────────────────────────────────────────────────
W_ALIVE = 0.2
FLIP_PENALTY = 30.0       # cut — falling is fine, exploration matters more

# ─── Smoothness ──────────────────────────────────────────────────────────────
W_SMOOTH = 0.05     # cut hard — polish, not bootstrap. Restore once walk emerges.
W_JVEL = 0.003      # cut — random exploration was being clobbered
W_SYMMETRY = 0.0    # off until policy walks. Then re-enable for polish.
W_ENERGY = 0.005

# ─── PPO ─────────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.00037
ENT_COEF = 0.2            # more entropy = more random exploration to escape standstill
CLIP_EPS = 0.15

# ─── Physics (requires restart) ──────────────────────────────────────────────
PHYSICS_STEPS_PER_ACTION = 16
