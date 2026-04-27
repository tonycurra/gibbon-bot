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

`train_mjx.py --test` auto-launches both the watcher and the params UI.

## Parameters (MJX)

All MJX params live in `envs/params_mjx.py`. Reward, motor and learning params hot-reload between rollouts; `PHYSICS_STEPS_PER_ACTION` is baked into the JIT and needs a restart.

Categories in the slider UI:

- **Viewer** — `VIEWER_SPEED` (display only, doesn't affect physics)
- **Motor** — `MOTOR_MAX_SPEED`
- **Height Goal** — `TARGET_Z`, `TARGET_TOL`, `W_HEIGHT`, `W_UPRIGHT`
- **Arm Spacing** — `ARM_SAFETY_MARGIN`, `W_PROXIMITY`
- **Survival** — `FLIP_PENALTY`, `W_ALIVE`
- **Smoothness** — `W_SMOOTH`, `W_JVEL`, `W_ENERGY`
- **Learning** — `LEARNING_RATE`, `ENT_COEF`, `CLIP_EPS`

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
