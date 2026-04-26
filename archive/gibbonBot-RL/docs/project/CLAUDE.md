# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Robotic ape research project: a 4-limb robot built from 4× RoArm M3 Pro arms (24 actuated joints + 6-DOF floating torso), simulated in MuJoCo and trained with PPO. The eventual goal is brachiation/climbing using ape video as motion reference.

## Hard rules

- **Never edit** files under `roarm_ws/roarm_description/urdf/roarm_m3/` or `roarm_ws/roarm_description/meshes/` — those are upstream. Edit `roarm_ws/roarm_description/ape.urdf` instead.
- **After every meaningful change**: commit to git AND update `DEVLOG.md` (and `README.md` if workflow/interface changed). The devlog is the lab notebook — log what was tried, why, and the result, including negative results. This is a research project; reproducibility matters.
- All simulation/RL work happens on branch `sim/mujoco-setup`. `master` is the original ROS2 baseline.

## Commands

Activate the venv first:
```bash
source .venv/bin/activate
```

### Training
```bash
python run_train_mjx.py                        # GPU PPO via JAX/MJX, 1024 envs (~46M steps/hr on RTX 4070)
python run_train_mjx.py --envs 2048 --inherit  # more envs, warm-start from prior best
python train.py --task stand --watch           # CPU PPO via SB3 + live viewer
python train.py --task stand --resume          # continue same run
python train.py --task stand --inherit         # new run, inherit weights from best (use after hardware param changes)
python train.py --lineage --task stand         # show lineage chain
```

### Viewing / evaluation
```bash
python src/gibbonbot/utils/watch.py --task stand   # live viewer, hot-swaps checkpoints from runs/ape_stand/best/
python src/gibbonbot/utils/watch_mjx.py            # live viewer for MJX training
tensorboard --logdir runs/
```

### Scene rebuild
```bash
python tools/build_scene.py                     # regenerate scene.xml from ape.urdf
```

## Architecture

### Two training paths share one scene + reward design

- **CPU path** (`train.py` + `envs/stand_env.py` / `envs/ape_env.py`): Gymnasium + Stable-Baselines3 PPO. Slow (~1M steps/hr) but supports a built-in live viewer and **hot-reloaded reward shaping** via `envs/hints.py` — edit `hints.py` while training is running and changes apply at the next episode.
- **GPU path** (`train_mjx.py` + `envs/stand_env_mjx.py`): pure JAX/MJX, 1024+ parallel envs on the GPU. Reward shaping in `hints.py` is **baked in at JIT startup** — restart training to apply changes. Checkpoints land in `runs/ape_stand_mjx/` as `.npz`, with an SB3-compatible `best_model.zip` exported so `tools/watch.py` works for both paths.

Keep the two env implementations behaviourally aligned — same OBS_DIM/ACT_DIM, same reward terms — so policies can be evaluated and viewed interchangeably.

### Scene + robot

- `scene.xml` — MJCF scene compiled from `ape.urdf` plus floor/lighting/actuators. Uses **capsule geometry** (no meshes referenced at runtime) so it stays MJX-compatible.
- `roarm_ws/roarm_description/ape.urdf` — the robot description: 1 floating torso + 4 prefixed arms (`fl_`, `fr_`, `bl_`, `br_`). Edit this to change the robot. Mount rpy/xyz of the four arms is what defines pose stance.
- Position actuators with `kp=8`. Changing `kp` or `forcerange` is a hardware change — bump generation by retraining with `--inherit`.
- **Servo specs** (RoArm-M3-Pro, see `docs/hardware/roarm_m3_pro_specs.md`):
  - Standard joints: **ST3235** → 30 kg.cm = **2.943 Nm**, 45 RPM = 4.71 rad/s, 70.5 g
  - Shoulder: **dual-drive (2× ST3235)** → 60 kg.cm = **5.886 Nm** (reflected in `scene.xml`)
  - Gripper: **ST3215-HS** → 20 kg.cm = **1.962 Nm**
  - Arm weight: **1020.8 g** per complete arm (link masses in scene.xml scaled to match)

### Lineage

`lineage.py` is invoked by both training entrypoints to append an entry per run to `runs/<task>/lineage.json`, capturing the scene's physics params (kp, force range) and which checkpoint the run inherited from. This is how generational transfer is tracked — when you change physics, `--inherit` continues the chain.

### Tasks

- **stand** (`stand_env.py` / `stand_env_mjx.py`): lift torso to 0.30 m, hold stable. 59-dim obs, 24-dim action, height + upright + alive − ctrl² − jvel² − Δaction². Termination: torso z < 3 cm. 12 physics steps per action.
- **reach** (`ape_env.py`): walk toward tree at (5,0,0). 63-dim obs (adds tree direction + distance), reward is progress + alive − ctrl² − jvel².

### Reward shaping (`envs/hints.py`)

Single source of truth for auxiliary reward weights (e.g. `spread_legs`, `front_arms_down`, `symmetry`, `stillness`). Both training paths import from it; the CPU path re-imports per-episode, the MJX path closes over the values during JIT compilation.

## Outputs

| Path | Contents |
|------|----------|
| `runs/ape_<task>/` | SB3 checkpoints + TensorBoard logs |
| `runs/ape_<task>/best/best_model.zip` | best policy (watched by `tools/watch.py`) |
| `runs/ape_<task>/lineage.json` | per-run hardware params + inheritance chain |
| `runs/ape_stand_mjx/` | MJX `.npz` checkpoints + exported `best_model.zip` |

## Remote training

`remote_sync.sh` runs a bidirectional rsync loop with the H100 server. Train remotely while watching locally via `tools/watch.py` — `envs/hints.py` edits propagate through the sync to steer the remote run live (CPU path) or at the next restart (MJX). See `REMOTE.md`.
