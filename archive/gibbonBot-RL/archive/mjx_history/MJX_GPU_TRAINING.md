# MJX GPU Training Guide

This document consolidates all knowledge about MuJoCo MJX GPU training in this repository.

## Overview

MJX (MuJoCo JAX) enables GPU-accelerated reinforcement learning by running thousands of parallel environments entirely on the GPU via JAX. This achieves **46× speedup** over CPU training (46M vs 1M steps/hour on RTX 4070).

## Key Files

### Core Training Files
- `train_mjx.py` - Main training script (PPO in JAX/Flax/Optax)
- `envs/stand_env_mjx.py` - Pure-JAX environment implementation
- `runs/ape_stand_mjx/` - Checkpoint directory for MJX training

### Supporting Files
- `data/scene.xml` - Robot MJCF definition (capsule tree trunk for MJX compatibility)
- `envs/params.py` - Hot-reloadable training parameters
- `utils/lineage.py` - Run metadata logger

## Setup

### Requirements
- NVIDIA GPU with CUDA support
- Python 3.9+
- JAX with CUDA enabled
- MuJoCo MJX 3.8.0

Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
mujoco-mjx==3.8.0
jax[cuda]==0.4.13
flax==0.7.0
optax==0.1.5
```

## Usage

### Basic Training
```bash
python train_mjx.py              # 1024 envs, train from scratch
python train_mjx.py --envs 2048  # More envs (needs more VRAM)
python train_mjx.py --resume     # Continue from latest checkpoint
```

### Remote Training (H100 Example)
```bash
# Start training
ssh root@<GPU_HOST> "cd /root/gb2.1 && source .venv/bin/activate && nohup python3 train_mjx.py --envs 4096 > train_mjx.log 2>&1 &"

# Monitor logs
ssh root@<GPU_HOST> "tail -30 /root/gb2.1/train_mjx.log && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# Kill training
ssh root@<GPU_HOST> "pkill -f train_mjx"
```

## Architecture

### Environment Design
- **Stateless functions**: `reset()` and `step()` operate on MJX data pytrees
- **Parallelism**: `jax.vmap` provides free N-env parallelism
- **Collision handling**: Mesh geom collisions disabled to avoid JIT explosion
- **Physics**: 16 physics steps per action (0.032s per action)

### Training Loop
- **Network**: Flax `ActorCritic` with tanh-squashed Gaussian policy
- **PPO Hyperparameters**:
  - N_STEPS = 128
  - EPOCHS = 4
  - MINIBATCHES = 8
  - GAMMA = 0.99
  - GAE_LAMBDA = 0.95
  - CLIP_EPS = 0.2
  - LR = 3e-4

### Performance
- **1024 envs on RTX 4070**: ~46M steps/hour
- **Checkpointing**: Saves every 1M steps to `runs/ape_stand_mjx/mjx_*.npz`
- **Logging**: Every 10 updates

## Key Engineering Decisions

### Mesh Collision Disabling
```python
# In stand_env_mjx.py
for i in range(mj_model.ngeom):
    if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
        mj_model.geom_contype[i] = 0
        mj_model.geom_conaffinity[i] = 0
```

**Reason**: STL meshes cause combinatorial explosion of collision kernel compilation pairs in MJX.

### Tree Trunk Geometry
- Changed from cylinder to capsule in `scene.xml`
- **Reason**: MJX doesn't implement cylinder-box collision

### Hot-Reloadable Parameters
```python
# In train_mjx.py
def read_rparams():
    # Reads from envs/params.py, allows live tuning between rollouts
```

## Monitoring and Debugging

### Log Monitoring
```bash
# Wait for first training step
tail -f runs/train_mjx.log | grep -m 1 "steps="

# Filter important events
grep -E 'steps=|Compiling|Training|Error|Traceback' runs/train_mjx.log | tail -20
```

### GPU Monitoring
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## Checkpoint Management

### Checkpoint Format
- Saved as `mjx_<steps>.npz` files
- Contains flattened parameter trees
- Compatible with SB3 `best_model.zip` export

### Resuming Training
```bash
python train_mjx.py --resume
# Automatically finds latest checkpoint in runs/ape_stand_mjx/
```

## Performance Optimization

### VRAM Usage
- 1024 envs: ~8GB VRAM
- 2048 envs: ~12GB VRAM
- 4096 envs: ~20GB VRAM (H100 recommended)

### Speed vs Envs
| Envs | Steps/hour (RTX 4070) |
|------|----------------------|
| 1024 | ~46M |
| 2048 | ~60M |
| 4096 | ~75M (H100) |

## Troubleshooting

### Common Issues

**JIT Compilation Hang**:
- Cause: Mesh collisions enabled
- Fix: Ensure mesh geom collisions are disabled in `load_model()`

**CUDA Out of Memory**:
- Reduce `--envs` parameter
- Try 512 or 256 envs for smaller GPUs

**Slow Performance**:
- Check `nvidia-smi` for GPU utilization
- Ensure JAX is using CUDA: `print(jax.devices())` should show GPU devices

## Development Notes

### Key Commits
- `67f032a`: Initial MJX implementation (46× speedup)
- `076795f`: Robust path resolution
- `a5af3b5`: Training script updates
- `c5ca157`: Inheritance fix for MJX weights
- `cf6614d`: Live viewer for MJX training

### Future Improvements
- Auto-detect GPU VRAM to set optimal N_ENVS
- Mixed precision training
- Gradient accumulation for larger batch sizes
- Distributed training across multiple GPUs

## Migration from CPU Training

### Key Differences
| Aspect | CPU (train.py) | GPU (train_mjx.py) |
|--------|---------------|------------------|
| Framework | Stable Baselines 3 | JAX/Flax/Optax |
| Parallelism | ~16 envs | 1024+ envs |
| Speed | ~1M steps/hour | ~46M steps/hour |
| Checkpoints | .zip files | .npz files |
| Viewer | Integrated | Separate watch_mjx.py |

### Compatibility
- MJX checkpoints can be exported to SB3 format
- `watch.py` can visualize MJX training progress
- Parameters in `envs/params.py` work for both

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [MuJoCo MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [Optax Documentation](https://optax.readthedocs.io/)

## Appendix: Full Parameter List

### Training Parameters (envs/params.py)
- `TARGET_Z`: Target torso height
- `TARGET_TOL`: Gaussian sigma for height reward
- `W_HEIGHT`: Height reward weight
- `W_UPRIGHT`: Upright reward weight
- `ARM_SAFETY_MARGIN`: Arm spacing safety margin
- `W_PROXIMITY`: Arm proximity penalty weight
- `FLIP_PENALTY`: Upside-down penalty

### Environment Constants
- `MAX_STEPS`: 2000 steps per episode
- `PHYSICS_STEPS_PER_ACTION`: 16 (0.032s per action)
- `OBS_DIM`: 59 (torso z + quat + velocities + joint positions/velocities)
- `ACT_DIM`: 24 (joint actions)
