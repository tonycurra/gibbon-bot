# gibbonBot-RL

Reinforcement learning for a 4-limb robotic ape. SB3 PPO on CPU with MuJoCo.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt install python3-tk   # for the params UI (WSL/Ubuntu)
```

## Workflow

```bash
# Train with live viewer + params UI (default)
python train.py

# Train without viewer
python train.py --no-watch

# Continue from latest checkpoint (fresh step counter, same weights)
python train.py --inherit

# Continue exact run
python train.py --resume

# Replay a checkpoint
python analyze.py
python analyze.py ape_stand_50000.zip

# Open viewer only (no training)
PYTHONPATH=src python src/gibbonbot/utils/watch.py --task stand

# Open params UI standalone
PYTHONPATH=src python src/gibbonbot/utils/params_ui.py
```

## Real-time parameter control

A floating slider panel opens automatically with `train.py` or `watch.py`.
Move a slider → `params.py` is updated → the env picks it up within one step.

Key params to tune while watching:

| Param | Effect |
|-------|--------|
| `MOTOR_MAX_SPEED` | How fast joints move (too low = frozen, too high = shaking) |
| `PHYSICS_STEPS_PER_ACTION` | Simulation time per policy step (higher = slower training) |
| `TARGET_Z` | Target shoulder height in metres |
| `W_HEIGHT` | How much the height goal dominates |
| `W_STABLE` | Bonus for holding the target height while still |
| `W_PROXIMITY` | Penalty for arms crossing each other |

## Files to edit

| File | What to change |
|------|----------------|
| `src/gibbonbot/envs/params.py` | All reward weights and motor constants (hot-reloaded each step) |
| `src/gibbonbot/envs/hints.py` | Extra shaped rewards (hot-reloaded each episode) |
| `src/gibbonbot/envs/stand_env.py` | Reward logic, observations, episode termination |
| `data/scene.xml` | Robot physics, joint limits, masses |

Checkpoints saved to `runs/ape_stand/`. Viewer auto hot-swaps to latest checkpoint.

## Goal

Lift the 4 arm-shoulder attachment points to **10 cm off the ground** and hold
them stable. The height reward is a tight Gaussian (σ = `TARGET_TOL`) centred on
`TARGET_Z`. A `W_STABLE` bonus fires every step the robot is at height and has
low joint velocity (`STABLE_VEL`).
