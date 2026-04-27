# Parameter History — All Branches & Commits

All params found across every branch and commit in this repo.
Values shown as: `default (range seen across history)`

---

## TIMING

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main/simple-reward | **current mjx-gpu** |
|-------|-------------|---------|-----------------|---------------|--------------------|---------------------|
| `PHYSICS_STEPS_PER_ACTION` | Physics steps per policy action. Higher = smoother, slower to train. **This is the #1 smoothness lever.** Baked into JIT — requires restart. | 210 | 220 | 240 | 160 | **16** |
| `VIEWER_SPEED` / `REALTIME_FACTOR` | Display speed multiplier. Does NOT affect physics or training. | — | 0.1 | 0.1 | 100 | 4 |

---

## MOTOR

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main | **current** |
|-------|-------------|---------|-----------------|---------------|------|-------------|
| `MOTOR_MAX_SPEED` | Max normalised joint travel per action. Low = slow/deliberate. High = fast/jerky. | 0.015 | 0.071 | 0.20 | 0.17 | **0.50** |
| `MAX_ACTIVE_JOINTS` | Penalty fires if more than N joints move in one step. Encourages sequential, animal-like motion. 24 = disabled. | 24 | 10 | 12 | — | **not implemented** |
| `ACTION_SMOOTH_ALPHA` | Low-pass filter on policy output: `filt = α·raw + (1-α)·prev`. 1.0 = no filter. | — | — | — | — | 1.0 |

---

## HEIGHT GOAL

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main | **current** |
|-------|-------------|---------|-----------------|---------------|------|-------------|
| `TARGET_Z` | Target torso height in metres. | 0.10 | 0.20 | 0.30 | 0.20 | 0.60 |
| `TARGET_TOL` | Gaussian sigma — how precisely height must be hit. Small = hard, large = forgiving. | 0.02 | 0.06 | 0.06 | 0.10 | 0.20 |
| `LIFTED_Z` | One-time "got up" bonus fires when torso first clears this height. | 0.07 | 0.07 | 0.20 | — | **not implemented** |
| `MIN_HEIGHT` | Below this = fallen (no termination, just stops rewarding height). | 0.03 | 0.03 | 0.03 | — | **not implemented** |

---

## REWARD WEIGHTS — POSITIVE

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main | **current** |
|-------|-------------|---------|-----------------|---------------|------|-------------|
| `W_HEIGHT` | Scale of the Gaussian height reward (main objective). | 2.0 | 2.0 | 2.0 | 5.0 | 10.0 |
| `W_UPRIGHT` | Reward for keeping torso level (proportional to quat w²). | 0.3 | 0.3 | 0.3 | 1.0 | 5.0 |
| `W_GOT_UP` | One-time bonus when torso first clears `LIFTED_Z`. Rewards initial lift effort. | 0.5 | 0.5 | 2.4 | — | **not implemented** |
| `W_STABLE` | Per-step bonus when at target height AND joint velocity below `STABLE_VEL`. Rewards holding position. | 1.0 | 1.0 | 1.0 | — | **not implemented** |
| `W_ALIVE` | Small per-step survival bonus. Prevents robot from learning to fall immediately. | 0.05 | 0.05 | 1.0 | — | 0.05 |
| `W_PLANT_LIFT` | Reward for grippers on floor while torso moves up. Teaches push-off mechanics. | 0.3 | 0.3 | 0.3 | — | **not implemented** |

---

## REWARD WEIGHTS — PENALTIES

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main | **current** |
|-------|-------------|---------|-----------------|---------------|------|-------------|
| `W_SMOOTH` | Penalty for motor/action movement each step. Reduces twitching. | 0.05 | 0.05 | 0.50 | — | 2.0 |
| `W_JVEL` | Penalty for fast joint velocities. Reduces flailing. | 0.001 | 0.001 | 0.001 | — | 0.05 |
| `W_ENERGY` | Penalty for motor effort `sum(ctrl²)`. Rewards efficiency. | 5e-4 | 5e-4 | 0.0015 | — | 0.01 |
| `W_DRAG` | Penalty for dragging limbs along floor. Encourages lifting over sliding. | 0.5 | 0.5 | 0.5 | — | **not implemented** |
| `W_JOINTS` | Penalty per joint beyond `MAX_ACTIVE_JOINTS` that moves in a step. 0 = disabled. | 0.0 | 0.0 | 0.8 | — | **not implemented** |

---

## ARM SPACING

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main | **current** |
|-------|-------------|---------|-----------------|---------------|------|-------------|
| `ARM_SAFETY_MARGIN` | Soft penalty ramps up quadratically when arms are closer than this (metres). | 0.06 | 0.10 | 0.20 | 0.20 | 0.05 |
| `W_PROXIMITY` | Scale of the soft arm-spacing penalty. | 2.0 | 50.0 | 4.5 | 10.0 | 3.0 |
| `ARM_TOUCH_DIST` | Hard termination if any cross-arm pair comes within this distance (metres). | — | 0.01 | 0.05 | — | **not implemented** |
| `ARM_COLLISION_PENALTY` | One-time penalty when `ARM_TOUCH_DIST` is violated. | — | 50.0 | 200.0 | — | **not implemented** |

---

## TERMINATION

| Param | What it does | 2a0bfe4 | clean/train-loop | lean-training | main | **current** |
|-------|-------------|---------|-----------------|---------------|------|-------------|
| `FLIP_PENALTY` | One-time penalty when torso flips upside-down. Episode terminates. | 10.0 | 50.0 | 50.0 | 100.0 | 200.0 |
| `STABLE_VEL` | Max joint velocity (rad/s) counting as "still" for `W_STABLE`. | 0.05 | 0.37 | 0.37 | — | **not implemented** |

---

## LEARNING (mjx-gpu only)

| Param | What it does | **current** |
|-------|-------------|-------------|
| `LEARNING_RATE` | Adam optimiser step size. Hot-reloadable (rebuilds optimiser). | 0.00033 |
| `ENT_COEF` | Entropy bonus — higher = more exploration. Hot-reloadable. | 0.05 |
| `CLIP_EPS` | PPO clip range — limits policy change per update. Hot-reloadable. | 0.32 |

---

## HINTS (clean-no-capsules, tree-navigation branches only)

Baked into JIT at `load_model()` time from `hints.py`. Hot-reloadable between episodes (not between steps). Setting to 0.0 disables without deleting.

| Hint | What it does | clean-no-capsules value |
|------|-------------|------------------------|
| `upright` | Extra reward for torso staying level (adds to base W_UPRIGHT). | 0.5 |
| `spread_legs` | Reward for base joints splayed outward — wider stance = more stable. | 0.3 |
| `front_arms_down` | Reward for front gripper tips pointing toward the floor. | 0.5 |
| `back_arms_down` | Reward for back gripper tips pointing toward the floor. | 0.5 |
| `stillness` | Penalty for fast joint velocity (adds to base W_JVEL). | 0.05 |
| `symmetry` | Reward for left-right arm symmetry. | 0.0 (off) |
| `tree_seeking` | (tree-navigation only) Reward for moving toward a tree obstacle. | 1.0 |

---

## SUMMARY: What made the smoothest behaviour

The smoothest runs used:
- `PHYSICS_STEPS_PER_ACTION` = **210–240** (0.42–0.48s per decision) — biggest factor
- `MOTOR_MAX_SPEED` = **0.015–0.071** — slow, deliberate joints
- `MAX_ACTIVE_JOINTS` = **10–12** — sequential limb motion
- `W_SMOOTH` = **0.05** (not 0.5–2.0)
- `W_JVEL` = **0.001** (not 0.05)
- `W_ENERGY` = **5e-4** (not 0.01)
- `FLIP_PENALTY` = **10–50** (not 200)
- `W_GOT_UP` + `W_PLANT_LIFT` + `W_STABLE` active — richer reward shaping

Everything in the current mjx-gpu branch that is **not in the MJX env yet**:
`MAX_ACTIVE_JOINTS`, `W_GOT_UP`, `W_STABLE`, `W_PLANT_LIFT`, `W_DRAG`, `ARM_TOUCH_DIST`, `ARM_COLLISION_PENALTY`, `hints.py` system.
