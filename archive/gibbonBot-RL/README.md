# gibbonBot-RL

Reinforcement learning for a 4-limbed robotic ape. PPO + MuJoCo MJX (GPU-accelerated, JAX). Goal: lift the torso to a target height and stay upright.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [QUICKSTART.md](QUICKSTART.md) for the recommended workflow.

## Scripts

| Script | What it does |
|--------|-------------|
| `train_mjx.py` | MJX/JAX GPU training loop with vmapped parallel envs |
| `utils/watch_mjx.py` | Live MuJoCo viewer — hot-swaps to the latest checkpoint |
| `envs/params_mjx.py` | Run directly to open the categorised slider UI for live tuning |
| `envs/stand_env.py` | CPU Gymnasium reference env (kept for reference, not the main path) |
| `envs/params.py` | Params for the CPU reference env |

## Two modes

- **GO** *(default)* — `python train_mjx.py` — auto-picks 2048 envs on GPU (64 on CPU), spawns the viewer only. Terminal shows per-episode `reward≈X ↑/↓ (best Y)` so you can tell if it's learning. No params UI to tempt you.
- **TUNE** — `python train_mjx.py --tune` — 32 envs, frequent saves, params UI auto-launches alongside the viewer for live tweaking.

## Parameters (MJX)

All MJX params live in `envs/params_mjx.py`. Reward, motor, smoothness and learning params hot-reload between rollouts; `PHYSICS_STEPS_PER_ACTION` (marked ⚙ in the UI) is baked into the JIT and needs a restart.

Categories in the slider UI (top → bottom by visible impact):

- ⚡ **Motion** — `PHYSICS_STEPS_PER_ACTION` ⚙, `MOTOR_MAX_SPEED`, `ACTION_SMOOTH_ALPHA`, `VIEWER_SPEED`
- 🧊 **Smoothness Penalties** — `W_SMOOTH`, `W_JVEL`, `W_ENERGY`
- 🎯 **Goal Shaping** — `TARGET_Z`, `TARGET_TOL`, `W_HEIGHT`, `W_UPRIGHT`
- 💀 **Survival** — `W_ALIVE`, `FLIP_PENALTY`
- 🤝 **Arm Safety** — `ARM_SAFETY_MARGIN`, `W_PROXIMITY`
- 🧠 **Learning** — `LEARNING_RATE`, `ENT_COEF`, `CLIP_EPS`

Live feedback bars next to motor / smoothness / goal params are driven by `runs/ape_stand_mjx/live_stats.json` (written by the viewer). Bottom panel shows checkpoint, torso z, upright, ep length, rolling flip rate.

Three preset buttons at the top: **Smooth**, **Fast train**, **Reset defaults**.

## Files

```
envs/
  stand_env_mjx.py — MJX environment (stateless, vmappable)
  params_mjx.py    — MJX params + categorised slider UI
  stand_env.py     — CPU reference env
  params.py        — CPU params + slider UI
data/
  scene.xml        — Robot MJCF
utils/
  watch_mjx.py     — Live checkpoint viewer (filters by --since timestamp)
  lineage.py       — Run metadata logger
runs/
  ape_stand_mjx/   — MJX checkpoints (mjx_<steps>.npz) + lineage.json
```
