# Training Lessons

Cumulative insights from training runs. Append entries newest-first. Each entry: date, run label or MLflow run_id, what worked / what didn't, and the param diff that caused it.

Use MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) for the metrics/curves; this file is for the *prose* — context, hypotheses, gotchas.

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
