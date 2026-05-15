# LIANA — Architecture & Persistence

> Demo north star: **robot walks to the olive tree on command, reliably**.

This document answers the question: **when something works, where does
that knowledge live?** And: **how does LIANA connect to the brain?**

## The five places knowledge lives

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   CODE & SCENE (immutable per commit)                                   │
│     git: envs/env.py · data/scene.xml · data/gibbonbot.xml              │
│     ← reward shaping, reset logic, motor limits, robot mesh, gravity    │
│                                                                         │
│   ─ feeds ▼                                                             │
│                                                                         │
│   TRAINER (train_mjx.py, GPU MJX, many envs in parallel)                │
│     produces ┐                                                          │
│              │                                                          │
│   ─────────── ▼ ────────────                                            │
│                                                                         │
│   ┌─ runs/ape_<task>_mjx/        ── checkpoints & per-run logs ──┐      │
│   │     mjx_<step>.npz            rolling, last few              │      │
│   │     keepers/<tag>.npz         promoted on success_frac > 0   │      │
│   │     keepers/<tag>.json        sidecar: metrics + mlflow run  │      │
│   │     train.log                 per-update metrics, plain text │      │
│   │     events.jsonl              structured event stream        │      │
│   │     autopilot_state.json      phase tracker (legacy)         │      │
│   │     lineage.json              what previous .npz seeded this │      │
│   └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
│   ┌─ mlruns/  +  mlflow.db ─── tracking & comparison ──────────┐        │
│   │     experiment = ape_<task>_mjx                            │        │
│   │     run_name   = YYYYMMDD-HHMMSS                           │        │
│   │     params     = every W_*, MOTOR_*, PPO knob              │        │
│   │     metrics    = mean_rew, mean_dist, success_frac, …      │        │
│   │     tags       = task, inherited_from, jax_backend, …      │        │
│   │     fingerprint= sha256 of scene + env code + params       │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                         │
│   ┌─ data/brain/                  brain memory (SQLite) ───────┐        │
│   │     brain_state.sqlite        episodic memories, rewards   │        │
│   │     policy_db.sqlite          policy index across runs     │        │
│   │   restored by trainer on every start (see brain_hook)      │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                         │
│   ┌─ LIANA.md  +  LIANA_LOG.md  · human narrative ─────────────┐        │
│   │     index of north star, tasks, lessons, manual notes      │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Which one answers which question

| Question | Answer source |
|---|---|
| What policy walks best? | `runs/ape_walk_mjx/keepers/` — filenames encode `succ` and `min_d` |
| Which params produced that policy? | The run's `params` in MLflow (per-key) |
| Were physics / scene different that day? | The run's `fingerprint` tag in MLflow (sha256 of scene + env code) |
| Where did this policy come from before? | `runs/ape_<task>_mjx/lineage.json` + MLflow `inherited_from` tag |
| What episodes has the brain remembered? | `data/brain/brain_state.sqlite` (hippocampus) |
| What did I try last Tuesday? | `LIANA_LOG.md` |
| What does this whole project aim at? | `LIANA.md` (north star, tasks summary) |

## Versioning model

- **Scene + env code + params live in git.** A commit captures all three.
- **Policy weights live on disk (`runs/…/.npz`)** and are gitignored.
  They’re linked back to git via the MLflow run’s scene/params
  fingerprint.
- **Training data (raw rollouts) is not persisted.** Mirror only the
  artifacts you care about — currently keepers + train logs — to
  Scaleway (per `feedback_track_training_data.md`).

When the scene XML or env code changes, MLflow records a new fingerprint
and the trainer's `--resume` either continues from the previous .npz
(force-resume) or branches into a new lineage.

## How the brain integrates with LIANA

The `brain/` package contains 18 bioinspired modules (ApeBrain,
Hippocampus, BasalGanglia, etc.). Today it's wired as follows:

```
TRAINER (train_mjx.py)
  │
  ├─ on start  → brain_hook.restore()    ← loads data/brain/*.sqlite
  ├─ per ep    → brain_hook.observe(...) ← logs reward/punishment
  └─ on stop   → brain_hook.save()       ← persists episodic memory

DASHBOARD (dashboard/server.py)
  │
  ├─ /api/brain/modules    ← list of 18 brain modules
  ├─ /api/brain/state      ← current state of each module
  └─ /api/brain/memory     ← last N hippocampus entries

LIANA (the user-facing layer)
  │
  ├─ ./liana go            ← launches dashboard
  ├─ ./liana log "…"       ← appends to LIANA_LOG.md
  └─ LIANA.md              ← human knowledge index
```

The brain currently **records** what happens during training. It does
not yet **drive** training decisions (curriculum, action priors). That's
the next integration step; for the demo we don't need it.

## Per-task reward switching

A single `envs/env.py` serves both `walk` and `stand`. The trainer calls
`env.read_params(task)` and gets a different reward tuple back:

| Constant         | `walk` | `stand` | Note                                 |
|------------------|-------:|--------:|--------------------------------------|
| `W_PROGRESS`     |   80.0 |     0.0 | tree-progress signal off in stand    |
| `W_FWD_VEL`      |   25.0 |     0.0 |                                      |
| `W_PROXIMITY_TREE` | 1.0  |     0.0 |                                      |
| `W_HEADING`      |    8.0 |     0.0 |                                      |
| `W_SUCCESS`      |  200.0 |     0.0 |                                      |
| `W_UPRIGHT`      |    3.0 |    12.0 | boosted for stand — the whole task   |
| `W_HEIGHT`       |    0.5 |     5.0 | boosted for stand                    |

Everything else (motor / smoothing / penalties / PPO) is the same.
Checkpoints land in segregated folders (`runs/ape_walk_mjx/` vs
`runs/ape_stand_mjx/`) so the two tasks never share weights.

## When something works

1. Trainer auto-saves a keeper with a descriptive filename:
   `auto_min_d0.07_succ1.00_step4416.npz`
2. Sidecar JSON `auto_min_d0.07_succ1.00_step4416.json` records the
   metrics, the params snapshot, the scene fingerprint, the env code
   fingerprint.
3. The trial subprocess automatically picks the highest-success keeper
   to play in the viewer (see `scripts/play_trial.py:latest_checkpoint`).
4. To preserve forever: commit the relevant params + scene XML to git;
   the keeper itself stays on disk (Scaleway-mirrored).

That's the persistence model. Nothing else needs to move.

## Library — named saved policies

`runs/library/` holds policy snapshots with friendly names. Created by
the **Library** card in the dashboard:

- **Save**: name the current best keeper of the active task. The .npz
  is copied into `runs/library/<slug>.npz` plus a `.json` sidecar with
  succ / min_dist / source path.
- **Use**: seed the active task's `runs/ape_<task>_mjx/` with a library
  entry as the highest-step checkpoint, so the next training run
  resumes from it.
- **Delete**: remove the entry (npz + sidecar).

Lets you preserve good poses across scenes / params resets / branching
runs without leaning on git or remembering filenames.

## Comparing runs inside LIANA (no MLflow UI needed)

The dashboard has a **Runs** card on the left column showing the last 20
training runs. It reads `mlflow.db` directly (no separate UI server) and
displays:

| Column | What it tells you |
|---|---|
| when | start time (newest first) + status dot |
| task | walk / stand label |
| scene | first 8 chars of the scene+code fingerprint |
| succ | last `success_frac` (fraction of envs reaching the tree) |
| rew | last `mean_rew` |
| W_PROG, SPEED, FLIP | the headline reward / motor params |
| envs | parallel env count |

Pick two rows with similar `scene` fingerprints and you can see which
**params** made a difference. Different fingerprint = different
scene/code, so direct comparison is invalid (the trainer flags this).

Backed by `/api/runs?limit=20` which queries `mlflow.db` via the MLflow
Python client. No external MLflow server is launched.
