# gibbonBot-RL

Reinforcement learning for a 4-limbed robotic ape. PPO (Stable Baselines 3) + MuJoCo. Goal: lift the torso to a target height and keep it level.

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
| `test.py` | CPU training with embedded viewer — watch the actual training robot |
| `train.py` | GPU training (auto-detect), headless + separate replay viewer |
| `envs/params.py` | Run directly to open the slider UI for live parameter tuning |
| `utils/watch.py` | Standalone replay viewer — hot-swaps to latest checkpoint |
| `utils/sim.py` | Open the MuJoCo scene without any training |

## Parameters

All params live in `envs/params.py` and hot-reload within one episode. Run `python envs/params.py` to open the slider UI.

| Param | Effect |
|-------|--------|
| `REALTIME_FACTOR` | Viewer speed (1 = real-time, 100 = as fast as possible) |
| `TARGET_Z` | Target torso height in metres |
| `TARGET_TOL` | Gaussian sigma — how precisely height is rewarded |
| `W_HEIGHT` | Weight on height reward |
| `W_UPRIGHT` | Weight on staying level |
| `MOTOR_MAX_SPEED` | Max joint movement per action step |
| `PHYSICS_STEPS_PER_ACTION` | Simulation steps per policy decision |
| `ARM_SAFETY_MARGIN` | Soft exclusion zone radius around each arm |
| `W_PROXIMITY` | Strength of the arm-spacing penalty (0 = off) |
| `FLIP_PENALTY` | One-time penalty for flipping upside-down |

## Files

```
envs/
  stand_env.py   — Gymnasium environment (reward, observations, termination)
  params.py      — Hot-reloadable parameters + slider UI
data/
  scene.xml      — Robot MJCF definition
utils/
  watch.py       — Live checkpoint replay viewer
  sim.py         — Scene viewer (no policy)
  lineage.py     — Run metadata logger
runs/
  ape_stand/     — Checkpoints from train.py
  ape_stand_test/ — Checkpoints from test.py
```
