# gibbonBot-RL

PPO + MuJoCo MJX (JAX, GPU-accelerated) for a 4-limbed robotic ape.
Currently learning to walk to a tree on flat ground.

## Quickstart

```bash
# train locally on GPU (everyday flow)
python train_walk_mjx.py --no-resume \
    --auto-push \
    --label "what changed this run"
# --auto-push: scripts/sync.sh fires after every checkpoint save → commits
#              runs/, mlruns/, mlflow.db, train.log and pushes to Codeberg.
# --label    : free-form note attached to the MLflow run as a tag.

# train remote
scripts/remote.sh start         # H100, walk task, autopilot, watchdog, all in one
scripts/remote.sh status        # GPU util + log tail + tmux state
scripts/remote.sh log           # live tail of training log
scripts/remote.sh stop          # graceful save + exit

# pull artifacts down + watch policy
scripts/pull_checkpoints.sh &   # syncs runs/ + mlruns/ from remote, every 30s
python view.py --env walk        # viewer hot-swaps to newest checkpoint
python view.py --env walk --best # loads best-by-mlflow-mean_rew checkpoint

# evaluate
python -m utils.eval --task walk         # cross-run summary
python -m utils.eval --task walk --plot  # writes reward.png
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# attach a free-form note to the latest MLflow run
python scripts/note.py "rew jumped at update 200 — heading reward dominates"

# back up before destroying instance
scripts/archive_runs.sh
```

## Layout

```
.
├── train_walk_mjx.py    main PPO trainer (the script you actually run)
├── view.py              live MuJoCo viewer (hot-swaps to newest checkpoint)
├── README.md            this file
├── requirements.txt
├── envs/                MJX environments + their hyperparams
├── data/                scene.xml, robot meshes, motor specs, terrain
├── trees/               olive-tree generation pipeline
├── utils/               autopilot, eval, best-checkpoint loader, limb dynamics
├── scripts/             remote.sh (drive remote GPU), monitor.py (watchdog),
│                        pull_checkpoints.sh, archive_runs.sh, gen_terrain.py, gen_trees.py
├── runs/                training output (mostly gitignored, except keepers/)
├── mlruns/              mlflow tracking (gitignored, synced from remote)
├── docs/                long-form: PLAN, CORE, BIOLOGY, REMOTE_GPU, BRANCHES, etc.
└── archive/             defunct code paths kept for reference
```

## What gets preserved

| What | Where | In git? |
|---|---|---|
| Auto-promoted curated checkpoints | `runs/<task>/keepers/auto_*.npz` + `.json` | **yes** |
| Reproducibility sidecar (params + fingerprint) | `runs/<task>/keepers/auto_*.json` | **yes** |
| Streaming/intermediate checkpoints | `runs/<task>/mjx_*.npz` | **yes** (auto-pushed to Codeberg) |
| MLflow run/metric history (sqlite) | `mlflow.db` | **yes** |
| MLflow legacy filesystem store | `mlruns/` | **yes** (deprecated, kept for old runs) |
| Per-run lineage | `runs/<task>/lineage.json` | yes |
| Training log | `runs/<task>/train.log` | yes |
| Lessons learned (prose) | `LESSONS.md` | yes |
| Hypothesis backlog | `BACKLOG.md` | yes |
| Robot URDF meshes | `data/meshes/*.stl` | yes |
| Scene XML | `data/scene.xml` | yes |

**Auto-push**: when `--auto-push` is passed to `train_walk_mjx.py`, `scripts/sync.sh`
runs after every checkpoint save (~every 32K env steps, ~30 s on the 4070). It
commits the data paths above and pushes to Codeberg. Idempotent — bails if
nothing changed. Worst-case data loss: ~30 s of training compute.

A keeper is auto-saved every PPO update where `min_dist < 1.5 m` OR
`success_frac > 5 %` is achieved. Each keeper carries a sidecar JSON with
the params, scene hash, and metric snapshot that produced it — full
reproducibility unit.

## Docs

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — paste-able commands, deeper reference
- **[docs/PLAN.md](docs/PLAN.md)** — autopilot training plan (phases A→D)
- **[docs/CORE.md](docs/CORE.md)** — multi-task framework + per-limb diagnostics
- **[docs/BIOLOGY.md](docs/BIOLOGY.md)** — evolutionary priors built into the policy
- **[docs/REMOTE_GPU.md](docs/REMOTE_GPU.md)** — remote H100 deep-dive
- **[docs/IMITATION_PLAN.md](docs/IMITATION_PLAN.md)** — ape-pose imitation roadmap
- **[docs/BRANCHES.md](docs/BRANCHES.md)** — branch atlas
- **[docs/GROVE_MASTERPLAN.md](docs/GROVE_MASTERPLAN.md)** — olive-grove environment
- **[BACKLOG.md](BACKLOG.md)** — prioritized list of unimplemented ideas
- **[LESSONS.md](LESSONS.md)** — cumulative training insights (prose; numbers in MLflow)

## Motor model

Per-actuator velocity & acceleration limits are derived from `data/motors.md`
(Waveshare ST3235 / ST3215 / ST3215-HS specs):

- ST3235 / ST3215: 4.71 rad/s (45 RPM), 2.94 N·m
- ST3215-HS (gripper/foot): 11.10 rad/s (106 RPM), 1.96 N·m

Acceleration is bounded so each joint follows a trapezoidal velocity profile —
smooth start *and* stop. Tunable in `envs/params_walk_mjx.py`:

- `MOTOR_SPEED_FRAC` (default 1.0) — global scale; <1 = slower than reality
- `MOTOR_ACCEL_RATIO` (default 5.0/s) — how quickly joints reach max speed
