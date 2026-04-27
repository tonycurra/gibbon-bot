# gibbonBot-RL

Reinforcement learning for a 4-limb robotic ape. PPO on CPU with MuJoCo MJX.

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

# Continue from latest checkpoint
python train.py --inherit

# Replay a checkpoint
python analyze.py
python analyze.py mjx_00010000000.npz
```

## Files to edit

| File | What to change |
|------|----------------|
| `src/gibbonbot/envs/stand_env.py` | Reward function, observations, episode termination |
| `src/gibbonbot/envs/stand_env_mjx.py` | MJX parallel env (same logic, JAX-compatible) |
| `src/gibbonbot/config/train_mjx.py` | PPO hyperparameters (lr, epochs, clip, etc.) |
| `data/scene.xml` | Robot physics, joint limits, masses |

Checkpoints saved to `runs/ape_stand_mjx/`.
