# gibbonBot-RL

Reinforcement learning for a 4-limb robotic ape. SB3 PPO on CPU with MuJoCo.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

```bash
# Train with live viewer (default)
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
```

## Files to edit

| File | What to change |
|------|----------------|
| `src/gibbonbot/envs/stand_env.py` | Reward function, observations, episode termination |
| `data/scene.xml` | Robot physics, joint limits, masses |

Checkpoints saved to `runs/ape_stand/`. Viewer auto hot-swaps to latest checkpoint.
