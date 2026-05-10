# Training Lessons

Cumulative insights from training runs. Append entries newest-first. Each entry: date, run label or MLflow run_id, what worked / what didn't, and the param diff that caused it.

Use MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) for the metrics/curves; this file is for the *prose* — context, hypotheses, gotchas.

---

## 2026-05-10 — calibrated CPG + pace gait

**Run label**: `PACE gait (left-pair right-pair), calibrated CPG, crawl-friendly`

**Goal**: get visible forward locomotion. User feedback: "robot moves at the start then enters a weird loop."

**What we found via `scripts/calibrate_cpg.py`** (probes each joint with ±0.5 to see how the limb tip moves):
- The robot's URDF has **mirrored quaternions per limb** (FL vs FR, BL vs BR), so the same joint angle on FL vs FR moves the limb in opposite world directions.
- **Front limbs propel via `yaw` (joint 0)**, not `pitch` (joint 1). Pitch on FL/FR moves the hand laterally across the body, which is useless for forward locomotion.
- **Back limbs propel via `pitch` (joint 1)**, with strong vertical lift on the positive direction.
- Joint range asymmetry means `+amplitude` and `-amplitude` produce wildly different motions for elbow/knee.

**CPG rewrite**:
- Asymmetric per-limb signal: front uses yaw with mirrored signs (FL `-`, FR `+`), back uses pitch with same sign (BL/BR `+`).
- Lift signal (`-pitch` for front, `+knee` for back) gated by `max(sin, 0)` so legs only lift during swing, never push down.
- Phase pattern switched to **pace** (`[0, π, 0, π]` — same-side pairs) per user request: simpler, more left-right balanced, OK to crawl on bent limbs.

**Outstanding limitation**: CPG-only motion at zero policy still doesn't translate the body forward — the legs lift but stance phase doesn't push back. Body needs to be lifted off the ground first (z=0.12 → 0.25+) before the legs can plant and push. PPO must learn this body-lift behavior.

**Outcome**: TBD.

---

## 2026-05-10 — serious A→B run with CPG trot baseline

**Run label**: `serious training: CPG trot + heading + 200-step episodes, smoother+stable`

**Goal**: actual locomotion from spawn to tree (3 m), stable + smooth gait.

**Key architectural change**: added a Central Pattern Generator (CPG) trot baseline in `envs/walk_env_mjx.py`. Sinusoidal pitch + elbow/knee oscillation with diagonal-pair phase offset (FL+BR vs FR+BL, π apart) at GAIT_PERIOD_STEPS=25 (≈0.8 s per cycle). Even with a zero policy, the robot now bobs up/down and side-shifts at ~1 Hz — verified via snapshot. Policy learns *corrections* to the CPG, not gait from scratch.

**Why this was needed**: prior runs showed the policy outputs converge to a *single static target pose* — joints reach it, no further motion. PPO + reward shaping alone is too slow to discover rhythmic locomotion from random init. CPG injects the rhythm; PPO refines amplitude/phase/balance/direction.

**Other changes vs prior baseline**:
- `MAX_STEPS = 60 → 200` (1.9 s → 6.4 s) — success bonus (W_SUCCESS=200) was unreachable at 60 steps; now achievable.
- `ENT_COEF = 0.4 → 0.08` — CPG provides exploration; policy can converge.
- `W_SMOOTH = 0.15 → 0.40`, `W_JVEL = 0.02 → 0.05` — stability matters now that locomotion exists.
- `FLIP_PENALTY = 30 → 60` — falling is no longer "fine for exploration".

**Outcome**: TBD.

---

## 2026-05-10 — trapezoidal motors + heading + recentered actions

**Run label**: `trapezoidal motors + heading + recentered actions`

**Goal**: fix discrete/jerky motor motion + give policy a heading signal so it stops squirming sideways toward the tree.

**Changes vs prior baseline**:
- Action mapping re-centered: `action=0 → joint=0` (extended pose) instead of asymmetric-range midpoint. Eliminated the "dead spider" rest pose.
- Added bounded-acceleration motor profile (`MOTOR_MAX_ACCEL=0.03`) — true trapezoidal velocity ramps instead of velocity step-functions.
- Added heading reward (`W_HEADING=8.0`) — torso forward axis dotted with to-tree direction.
- `MOTOR_MAX_SPEED 0.55 → 0.12`, `ACTION_SMOOTH_ALPHA 0.25 → 0.5`, `ENT_COEF → 0.2`.
- Re-trained from scratch (action mapping is structurally incompatible with old policy).

**Outcome**: TBD — let it run and append metrics here once update 50+ has data.

---

## Backlog of hypotheses to test (move to dated entries when run)

- [ ] Per-limb curriculum (one arm or one leg actuated at a time, others frozen) — see `BACKLOG.md`.
- [ ] Lower entropy (`ENT_COEF=0.05`) once gait emerges, to stop residual jitter.
- [ ] Add an "extension" reward (penalty on `sum(joint_angle²)` for elbows/knees) if the policy still curls limbs after re-centered action mapping.
