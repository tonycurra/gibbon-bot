# gibbonBot-RL

Reinforcement learning for a 4-limbed robotic ape. PPO + MuJoCo MJX (GPU-accelerated, JAX). Goal: lift the torso to a target height and stay upright.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [QUICKSTART.md](QUICKSTART.md) for the recommended local workflow, or [REMOTE_GPU.md](REMOTE_GPU.md) for renting + training on a remote H100.

## Scripts

| Script | What it does |
|--------|-------------|
| `train_mjx.py` | MJX/JAX GPU training loop with vmapped parallel envs |
| `utils/watch_mjx.py` | Live MuJoCo viewer — hot-swaps to the latest checkpoint |
| `utils/tag.py` | Tag checkpoints with notes ("this one was good") |
| `utils/lineage.py` | Show run history + per-checkpoint params snapshot table |
| `envs/params_mjx.py` | All MJX training params — edit values in this file, comments explain ranges |
| `envs/stand_env.py` | CPU Gymnasium reference env (kept for reference, not the main path) |
| `envs/params.py` | Params for the CPU reference env |

## Two modes

- **GO** *(default)* — `python train_mjx.py` — runs **1 env** by default (debug-friendly), spawns the viewer alongside. `SAVE_EVERY` adapts to env count so 1-env mode still produces checkpoints quickly. Use `--envs 1024` (or any number) for parallel training when you want speed.
- **TUNE** — `python train_mjx.py --tune` — 32 envs, frequent saves, edit `envs/params_mjx.py` directly while training runs (params hot-reload between rollouts).

## Parameters (MJX)

All MJX params live in `envs/params_mjx.py` — edit values directly, the file has detailed comments above each section explaining ranges and trade-offs.

Reload behaviour:

- **Hot-reload** (next rollout, no restart): rewards (`W_HEIGHT`, `W_UPRIGHT`, `W_SMOOTH`, `W_JVEL`, `W_ENERGY`, `W_ALIVE`, `FLIP_PENALTY`, `W_PROXIMITY`), motor (`MOTOR_MAX_SPEED`, `ACTION_SMOOTH_ALPHA`), goal (`TARGET_Z`, `TARGET_TOL`, `ARM_SAFETY_MARGIN`), learning (`LEARNING_RATE`, `ENT_COEF`, `CLIP_EPS`).
- **Restart required** (baked into MJX/JIT): `PHYSICS_STEPS_PER_ACTION`, `JOINT_DAMPING`, `MOTOR_KP`.

## Saving notes & checkpoint history

Every saved checkpoint automatically gets a `.params.json` sidecar capturing the *exact* params active at save time — so you can always go back and see what produced a good policy.

```bash
python utils/tag.py "first stable stand, smooth motion"   # tag latest checkpoint
python utils/tag.py 262144 "before the W_SMOOTH bump"     # tag by partial match
python utils/tag.py --list                                # one-line per checkpoint w/ tags
python utils/tag.py --show                                # full notes.md
python utils/lineage.py --checkpoints                     # tabular params per checkpoint
```

Notes go to `runs/ape_stand_mjx/notes.md` (git-tracked, survives across machines). The `.params.json` sidecars are gitignored but always present locally.

## Files

```
envs/
  stand_env_mjx.py — MJX environment (stateless, vmappable)
  params_mjx.py    — MJX params (plain config file, no UI)
  stand_env.py     — CPU reference env
  params.py        — CPU params (CPU env only)
data/
  scene.xml        — Robot MJCF (joint ranges/torques match real ST32xx motors)
  motors.md        — Motor specs (ST3235 / ST3215 / ST3215-HS) + per-joint mapping
  controls.md      — Real-robot web UI, safety, startup, sim-to-real notes
utils/
  watch_mjx.py     — Live checkpoint viewer; replays latest .npz at real time
                     (separate process from training — closing it doesn't pause training)
  lineage.py       — Run metadata logger
view_scene.py      — Standalone interactive viewer for scene.xml (pose the robot manually)
runs/
  ape_stand_mjx/
    mjx_<steps>.npz             — policy weights
    mjx_<steps>.params.json     — params snapshot at save time (auto)
    lineage.json                — per-run hardware/env metadata
    notes.md                    — your tagged checkpoints (via utils/tag.py)
```
