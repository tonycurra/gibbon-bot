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

## Floor Collision Fix

If robot parts pass through the floor during simulation, use the fixed scene file:

```bash
# Generate fixed scene with proper collision detection
python src/gibbonbot/utils/fix_collisions.py

# The fixed scene enables:
# - Proper floor-robot collision detection
# - Realistic friction parameters
# - Self-collision prevention between robot parts
```

The training script automatically uses `data/scene_fixed.xml` which has these fixes applied.

## Floor Contact Fixes

Fixed floor penetration issues by:
- Aligning visual and physics floor positions (both now at z=0)
- Fixing STL mesh contact affinity (changed from 1 to 31 for proper floor collisions)
- Reducing floor friction for stable contacts
- Updating constraint dimensions for proper plane contacts


## Technical Details: Floor Contact System

The floor contact system has been completely overhauled to prevent penetration issues:

### Key Fixes

1. **Visual-Physics Alignment**
   - Floor visual position: 
   - Floor physics position: 
   - Floor physics position: FLOOR_Z = 0.0
   - Result: Perfect alignment between visual and physics floors

2. **STL Mesh Contact Affinity**
   - Changed all robot mesh geoms from  to 
   - Floor can now properly contact all robot mesh surfaces
   - Fixes the "parts going below floor" issue

3. **Physics Parameters**
   - Reduced friction:  (was 5.0 0.5 0.1)
   - Fixed constraint dimensions:  (was 4)
   - Result: Stable contacts without jittering

### Scene History

Historical scene files from different development stages are preserved in the  folder. This allows:
Historical scene files from different development stages are preserved in the scenes_history/ folder. This allows:

- Comparing evolution of robot configuration
- Debugging regressions by reverting to known-working configurations
- Researching the development history of simulation parameters

## GPU Training

For GPU-accelerated training, use the  script:
For GPU-accelerated training, use the train_gpu.py script:

Usage examples:
  python train_gpu.py               # GPU training + live viewer
  python train_gpu.py --no-watch    # GPU training only
  python train_gpu.py --envs 4      # More parallel environments

### GPU Requirements

- PyTorch with CUDA support
- NVIDIA GPU with updated drivers
- CUDA toolkit compatible with your PyTorch version

### GPU vs CPU Comparison

| Aspect | train.py (CPU) | train_gpu.py (GPU) |
|--------|---------------|------------------|
| Speed | Slower | Faster (when GPU available) |
| Device | CPU only | GPU + CPU |
| Parallelism | Limited | Better parallelization |
| Fallback | Always works | Falls back to CPU if GPU unavailable |
