# 🏗️ Lean RL - Minimal Robot Learning Setup

## 📁 Folder Purpose

This `lean-rl` folder contains **only the essential files** needed to train and visualize the robot standing task. It's a clean, minimal setup with no extra clutter.

## 📚 Files Included

### 1. `train.py` - Unified Training & Visualization Script
**The main entry point - this is what you run!**

```bash
# Start training with live visualization
python train.py --watch

# Just visualize existing checkpoints  
python train.py --visualize

# Frame-by-frame analysis
python train.py --visualize mjx_00010000000.npz --step
```

**Features:**
- GPU-accelerated training with JAX
- Live visualization during training
- Checkpoint management
- Step-by-step debugging
- Progress tracking

### 2. `stand_env_mjx.py` - GPU Training Environment
**Where the robot learns to stand (fast GPU version)**

**Key improvements made:**
- ✅ **Coordination rewards** - Encourages 1-2 joints moving at a time
- ✅ **Upside-down penalty** - Strong -2.0 penalty for flips
- ✅ **Joint movement limits** - Penalizes moving too many joints simultaneously
- ✅ **Stable initial pose** - Prevents immediate backflips
- ✅ **Symmetric limb rewards** - Encourages coordinated movement patterns

### 3. `stand_env.py` - Visualization Environment
**Where you see the robot's progress (CPU version with rendering)**

**Same improvements as MJX version for consistency:**
- Same coordination rewards
- Same penalty structure
- Used for visualization and debugging

### 4. `scene.xml` - Robot Definition
**The robot's physics model (MuJoCo XML format)**

**Contains:**
- Robot body structure and joints
- Physics parameters (mass, friction, etc.)
- Actuators and sensors
- No URDF - MuJoCo uses its own XML format

## 🚀 Quick Start

### Install dependencies (if not already installed)
```bash
pip install jax mujoco numpy
```

### Start training with visualization
```bash
python train.py --watch
```

### What you'll see:
1. **Early training**: Random movements, immediate falls
2. **Mid training**: 1-2 joints moving at a time, brief standing
3. **Late training**: Smooth coordination, long duration standing

## 🎯 Training Phases

| Steps | Behavior | Expected Reward |
|-------|----------|-----------------|
| 0-1M | Random exploration | -5.0 to -1.0 |
| 1M-5M | Occasional brief standing | -1.0 to 0.5 |
| 5M-20M | Consistent short standing | 0.5 to 2.0 |
| 20M-50M | Long duration standing | 2.0 to 5.0 |
| 50M+ | Mastery | 5.0+ |

## 🔧 Key Improvements

### Problem: Robot was flipping and moving erratically
**Solutions implemented:**

1. **Fixed Upside-Down Reward Bug**
   - Before: `w * w` always positive (rewarded flips!)
   - After: `jnp.where(w > 0, 0.3 * w * w, 0.0)` (only reward upright)

2. **Added Joint Movement Limits**
   - Penalizes moving >6 joints simultaneously
   - Encourages 1-2 joints at a time
   - Formula: `-0.1 * (joints_moving - 6)`

3. **Added Coordination Rewards**
   - Rewards symmetric limb movement
   - Compares front/front, back/back, left/left, right/right limbs
   - Bonus: +0.05 per limb pair similarity

4. **Stabilized Initial Pose**
   - Reduced extreme joint angles
   - Lower initial noise (±0.2 vs ±0.4)
   - More balanced starting configuration

## 📊 Monitoring Progress

During training, watch for:
- ✅ **Increasing reward** over time
- ✅ **Longer episodes** before falling
- ✅ **Fewer joints moving** simultaneously (1-2 is ideal)
- ✅ **More coordination** between limbs
- ✅ **Higher maximum height** achieved

## 🎬 Visualization Controls

- **Close MuJoCo window** → Skip current visualization
- **Press 'Q'** → Exit visualization mode
- **--step flag** → Frame-by-frame analysis
- **--speed flag** → Control playback speed

## 🏆 Success Criteria

You're winning when:
- ✅ Robot moves 1-2 joints at a time (not all at once)
- ✅ No immediate backflips on spawn
- ✅ Can maintain balance for >100 steps
- ✅ Achieves height > 0.3m consistently
- ✅ Reward > 2.0 sustained

## 🔄 Checkpoint System

- **Saved every 500K steps** to current directory
- **Format**: `mjx_00005000000.npz` (500K steps)
- **Continue training**: `python train.py --inherit`
- **Visualize specific**: `python train.py --visualize mjx_00010000000.npz`

## 🎯 Philosophy

**Lean RL** = Only what's necessary, nothing more.
- ❌ No unnecessary scripts
- ❌ No complex configurations
- ❌ No unused dependencies
- ✅ Just the core training loop
- ✅ Clean visualization
- ✅ Effective learning

This folder gives you a **fresh start** with minimal context - just the essential components to train a robot to stand using coordinated, stable movements.