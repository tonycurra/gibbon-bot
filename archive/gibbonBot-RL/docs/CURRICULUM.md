# Curriculum — multi-skill training

The robot's "brain" gets built up by training atomic skills in sequence,
transferring policy weights from each skill to the next. This means each new
skill starts from a competent policy, not from scratch — actual transfer
learning, not 4 isolated random restarts.

## The skills

| Skill | Reward (minimal) | CPG | Success criterion |
|---|---|---|---|
| **stand** | upright + height + alive − flip | off | `mean_upright > 0.90` for 20 updates |
| **walk_forward** | linvel_x + alive − flip − ground_dip | pace | `mean_speed_x > 0.20 m/s` for 20 updates |
| **turn_left** | +angvel_z − \|linvel_xy\| − flip + alive | off | `mean_angvel_z > 0.40 rad/s` for 20 updates |
| **turn_right** | −angvel_z − \|linvel_xy\| − flip + alive | off | `mean_angvel_z < −0.40 rad/s` for 20 updates |

Plus the legacy walk-to-tree task (`walk`), which is a *compound* skill
(walk → stand on tree). It can use the curriculum-trained walk_forward as
a starting point.

## Why this works (and what we are NOT doing)

- **Frozen 88-dim observation contract** across every task. Same proprioception,
  same target-slot, same gait clock. Policy weights transfer without
  first-layer surgery.
- **Minimal reward per task** — one dominant signal each. PPO finds the
  gradient quickly because the reward isn't a 12-term soup.
- **CPG (pace gait) baseline** for walking; disabled for stand/turn (those
  aren't gaits).
- **Rest pose** (knees/elbows pre-bent 30°) is the *innate* posture — same
  for all tasks. Action=0 → standing-ish posture, not starfish.
- We are NOT yet doing recurrent policies, world models, hierarchical RL,
  or auto-meta-policy. Those are P2/P3 in `BACKLOG.md`. Multi-skill needs
  to work first with a flat MLP.

## Launch the full curriculum

```bash
python scripts/curriculum.py
```

This runs each task to its success criterion (or budget cap), then transfers
the best keeper to the next task. Total wall time: ~1–2 GPU-hours for the
first pass; thereafter `--from <task>` picks up where you left off.

Useful flags:

```bash
python scripts/curriculum.py --dry-run               # plumbing test, no compute
python scripts/curriculum.py --from walk_forward     # restart partway
python scripts/curriculum.py --task-budget 5_000_000 # smaller budget per task
python scripts/curriculum.py --watch                 # auto-spawn viewer per task
```

## Train a single task (manual)

The curriculum runner is just a sequencer over `train_mjx.py`. Run any task
directly:

```bash
python train_mjx.py --task stand --no-resume --auto-push --label "manual stand"
python train_mjx.py --task walk_forward \
    --resume-from runs/ape_stand_mjx/keepers/auto_upr0.93_z0.29_step8000000.npz \
    --auto-push --label "walk_forward from a stand keeper"
```

Early-stop on a sustained metric:

```bash
python train_mjx.py --task stand \
    --early-stop-metric mean_upright \
    --early-stop-op gt \
    --early-stop-value 0.92 \
    --early-stop-window 20
```

## Inspect per-task progress

```bash
# live log
tail -f runs/ape_stand_mjx/train.log

# MLflow UI (per-task experiment)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000, pick experiment "ape_stand_mjx"

# render a few frames from latest checkpoint
python scripts/snapshot.py --frames 6 --gap 0.7 --policy latest

# compare metrics across tasks
python scripts/compare_runs.py --top 8

# best keeper for a task
python -m utils.best --task stand
```

## When a task gets stuck

If a task never crosses its success criterion:

1. **Check obs contract** — fresh runs reset `min_dist`, etc., but `mean_upright`
   should rise on stand within a few minutes. If it stays at 0.5 forever, the
   spawn yaw or rest_pose is wrong.
2. **Check the visual** — `python view.py --env <task>`; if the robot doesn't
   even look like it's trying the task, the reward weights need tuning.
3. **Lower entropy** if stuck oscillating between strategies (`ENT_COEF` in
   `envs/params_<task>_mjx.py`).
4. **Skip ahead** — `--from next_task` if you can't afford another budget.
   Best-effort keeper is still promoted; you'll have a partially-good policy.

## Architecture pointers

- Trainer: `train_mjx.py` (one file, task-parameterized)
- Each env: `envs/<task>_env_mjx.py` — only `reset` / `step` / `is_keeper` differ
- Shared:
  - `envs/_shared_load_model.py` — physics, motors, rest pose
  - `envs/_shared_dynamics.py` — substep loop, motor profile, CPG
  - `envs/_shared_obs.py` — 88-dim obs contract
- Curriculum: `scripts/curriculum.py`
- Comparison: `scripts/compare_runs.py`
- Long-form architecture vision: `docs/BRAIN_CORE.md`
