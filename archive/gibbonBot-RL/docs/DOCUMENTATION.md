# gibbonBot-RL Documentation

## 📋 Overview

gibbonBot-RL is a reinforcement learning framework for training a 4-limb robotic ape to perform various tasks using MuJoCo physics simulation.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install CUDA for GPU acceleration
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. Train Your First Model

```bash
# CPU training (slower but with viewer)
python train.py stand --cpu

# GPU training (faster, requires CUDA)
python train.py stand

# Reach task (more challenging)
python train.py reach --cpu
```

### 3. Watch Training

Open a second terminal:
```bash
python tools/watch.py --task stand
```

## 🎯 Available Tasks

### 1. Stand Task
**File**: `envs/stand_env.py`
**Goal**: Lift torso to 0.40m height and maintain balance
**Difficulty**: ⭐⭐ (Good for starting)
**Features**:
- Upright torso reward
- Arm spacing hints
- Stability penalties

### 2. Reach Task
**File**: `envs/ape_env.py`
**Goal**: Move robot to reach a tree at (5, 0, 0)
**Difficulty**: ⭐⭐⭐⭐ (Requires balance + movement)
**Features**:
- Progress reward (moving closer)
- Proximity reward (exponential)
- Alignment reward (facing tree)

### 3. Unified Environment
**File**: `envs/unified_env.py`
**Goal**: Multi-task environment (experimental)
**Difficulty**: Configurable
**Features**:
- Multiple tasks in one environment
- Realistic physics
- Adaptive joint limits

## 📁 File Structure

```
gibbonBot-RL/
├── envs/                  # Environment definitions
│   ├── ape_env.py         # Reach task
│   ├── stand_env.py       # Stand task (CPU)
│   ├── stand_env_mjx.py   # Stand task (GPU)
│   └── unified_env.py    # Multi-task environment
│
├── tools/                 # Utility scripts
│   ├── watch.py          # Live viewer
│   ├── watch_multi.py    # Multi-view viewer
│   ├── eval.py           # Evaluation tool
│   └── fix_collisions.py # Collision setup
│
├── config/                # Configuration
│   └── train_mjx.py      # GPU training config
│
├── runs/                  # Training outputs (auto-generated)
│   ├── ape_stand/        # CPU training checkpoints
│   └── ape_stand_mjx/    # GPU training checkpoints
│
├── scene.xml              # Original scene
├── scene_fixed.xml        # Fixed collision scene
├── scene_backup.xml       # Backup
│
├── train.py              # Unified training script
├── train_fixed.py         # CPU training entry
├── lineage.py             # Training lineage tracking
│
├── requirements.txt      # Dependencies
├── README.md              # Main documentation
└── DOCUMENTATION.md       # Detailed documentation
```

## 🔧 Configuration

### Training Parameters

Edit `train.py` or use command-line arguments:

```bash
# Number of parallel environments
python train.py stand --envs 8

# Inherit from best checkpoint
python train.py stand --inherit

# Force CPU/GPU mode
python train.py stand --cpu
python train.py stand --gpu

# Disable auto-viewer
python train.py stand --no-watch
```

### Physics Parameters

Edit `scene_fixed.xml` for:
- Joint limits and damping
- Collision parameters
- Friction coefficients
- Initial pose

### Reward Shaping

Edit `envs/hints.py` for:
- Upright torso reward
- Arm spacing penalties
- Movement smoothness
- Energy efficiency

## 📊 Training Tips

### 1. Start Simple
```bash
# Begin with stand task
python train.py stand --cpu

# Once stable, try reach task
python train.py reach --cpu --inherit
```

### 2. Monitor Progress
```bash
# TensorBoard visualization
tensorboard --logdir runs/

# Check best model
ls -la runs/ape_stand/best/
```

### 3. Curriculum Learning
The reach task has built-in curriculum:
- Stage 0: Tree at 2.0m (easy)
- Stage 1: Tree at 3.5m (medium)
- Stage 2: Tree at 5.0m (hard)

## 🐛 Troubleshooting

### Arms Below Surface
**Solution**: Use `scene_fixed.xml` which has:
- Raised torso position (20cm above ground)
- Proper collision detection
- Joint limit enforcement

### Training Too Slow
**Solution**: 
- Use GPU training: `python train.py stand`
- Reduce environments: `--envs 4`
- Install CUDA JAX

### Falling Immediately
**Solution**:
- Increase `MIN_HEIGHT` in environment
- Check initial pose
- Reduce randomization

## 🔬 Technical Details

### Observation Space
- **Stand task**: 59 dimensions
  - Torso position (3) + orientation (4)
  - Torso velocity (6)
  - Joint positions (24) + velocities (24)

### Action Space
- 24 dimensions (one per joint)
- Normalized to [-1, 1] range
- Mapped to joint position limits

### Physics
- Time step: 0.002s
- Physics steps per action: 15 (30ms per action)
- Damping: 2.0 on all joints
- Friction: 1.0 sliding, 0.01 spinning, 0.001 rolling

## 🎓 Advanced Usage

### Custom Tasks
Create new environments in `envs/`:
```python
from gymnasium import spaces
import mujoco
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)
        # Define observation and action spaces
        # Implement step() and reset() methods
```

### Custom Rewards
Modify reward functions:
```python
def _calculate_reward(self):
    # Height reward
    height_reward = np.exp(-abs(torso_height - target)**2)
    
    # Stability reward
    stability_reward = -np.sum(joint_velocities**2)
    
    # Custom rewards
    custom_reward = your_custom_logic()
    
    return height_reward + stability_reward + custom_reward
```

## 📚 References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [JAX Documentation](https://jax.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📜 License

[MIT License](LICENSE)
