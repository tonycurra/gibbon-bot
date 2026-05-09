# gibbonBot-RL

PPO + MuJoCo MJX (JAX, GPU-accelerated) for a 4-limbed robotic ape.
Currently learning to walk to a tree on flat ground.

## Quickstart

```bash
# train remote (the everyday workflow)
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
mlflow ui                                # http://localhost:5000

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
| Streaming/intermediate checkpoints | `runs/<task>/mjx_*.npz` | gitignored, pulled by `pull_checkpoints.sh` |
| MLflow run/metric history | `mlruns/` | gitignored, pulled |
| Per-run lineage | `runs/<task>/lineage.json` | yes |
| Robot URDF meshes | `data/meshes/*.stl` | yes |
| Scene XML | `data/scene.xml` | yes |

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
