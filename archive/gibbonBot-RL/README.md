# gibbonBot-RL

PPO + MuJoCo MJX (JAX, GPU-accelerated) for a 4-limbed robotic ape. Current
focus: walk to a tree without dragging the torso. Long-term: olive grove
navigation, climbing, and harvesting.

## Where to start

| You want to… | Read |
|---|---|
| Train on a remote H100 (the main workflow) | [QUICKSTART.md](QUICKSTART.md) |
| Active autopilot plan (walk-to-tree phases) | [PLAN.md](PLAN.md) |
| Deep-dive on remote training (cost, teardown, perf) | [docs/REMOTE_GPU.md](docs/REMOTE_GPU.md) |
| Plan ape-imitation training (AMP-style) | [docs/IMITATION_PLAN.md](docs/IMITATION_PLAN.md) |
| Find which branch has which experiment | [docs/BRANCHES.md](docs/BRANCHES.md) |
| Phased plan for the olive grove | [docs/GROVE_MASTERPLAN.md](docs/GROVE_MASTERPLAN.md) |

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo layout

```
.                           ← daily-use scripts + top-level docs
├── train_walk_mjx.py       ← MJX/JAX PPO training loop (the main script)
├── view.py                 ← live MuJoCo viewer, hot-swaps to latest checkpoint
├── remote.sh               ← drive the remote GPU from local (start/status/log/...)
├── pull_checkpoints.sh     ← rsync runs + mlruns + rollouts from remote
├── gen_terrain.py          ← procedural terrain (slope/roughness/soil)
├── gen_trees.py            ← procedural capsule trees
│
├── envs/                   ← MJX environments + their hyperparams
│   ├── walk_env_mjx.py     ← walk-to-tree (current focus)
│   ├── stand_env_mjx.py    ← simple stand-up
│   ├── params_walk_mjx.py
│   └── params_mjx.py
├── data/                   ← scene.xml, motors.md, terrain assets
├── trees/                  ← olive-tree generation pipeline (Blender + Sapling)
├── utils/                  ← lineage logger, checkpoint tagger, rollout loader
├── runs/                   ← training output (gitignored except lineage.json + keepers/)
│
├── docs/                   ← long-form historical / planning docs
└── archive/                ← old code paths kept for reference (CPU trainer, browser viewer, etc.)
```

## The training loop in one paragraph

`train_walk_mjx.py` runs PPO with all envs vmapped on a single GPU. Every
update, MLflow logs `sps`, `mean_rew`, `mean_dist`, all hyperparams, and the
latest checkpoint name. With `--save-rollouts-every N`, every N-th update
also dumps the rollout buffer (obs/act/rew/done/val/logp/adv/ret) so the
GPU's compute isn't only producing weights — it produces a reusable
transition dataset too. See [REMOTE_GPU.md](REMOTE_GPU.md) for the rollout
flag details.

## What's not in the active workflow

Anything in `archive/` (old `train_mjx.py` for stand task, the CPU
stable-baselines3 trainer, the browser MJPEG viewer, the `mjx_history/` snapshot
pile) — kept for reference only. See [archive/README.md](archive/README.md).
