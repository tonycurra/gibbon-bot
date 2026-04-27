# 🎯 Slow, Steady Training Guide

## Problem Solved
You want to **watch the robot learn to stand slowly** during training, not just see the best checkpoint. The robot was doing immediate backflips because:
1. ❌ Watching "best" checkpoints (already learned behavior)
2. ❌ Unstable initial conditions
3. ❌ Poor coordination rewards

## ✅ Solution: Live Training Visualization

```bash
# Start training with live visualization (RECOMMENDED)
python scripts/train_with_visualization.py
```

### What This Does:
- **🎬 Shows CURRENT policy** every 10 training updates (not best checkpoint)
- **🐢 Slow, steady learning** - watch it improve from scratch
- **📈 Real-time progress** - see how the robot learns step by step
- **🔄 Interactive** - press 'Q' in MuJoCo window to skip visualizations

## 🎛️ Customization Options

```bash
# Visualize more frequently (every 5 updates)
python scripts/train_with_visualization.py --freq 5

# Use fewer environments for more stable training
python scripts/train_with_visualization.py --envs 256

# Start from previous best checkpoint
python scripts/train_with_visualization.py --inherit

# Visualize very frequently (every 3 updates)
python scripts/train_with_visualization.py --freq 3 --envs 256
```

## 📈 What You'll See

### Early Training (First 1M steps)
```
📈 Step      131,072 | SPS     2,176 | Reward -2.121 | Update 2
🎬 VISUALIZING CURRENT POLICY
   Training Step: 131,072
   Update: 2
   Press 'Q' in MuJoCo window to skip visualization
============================================================
📊 VISUALIZATION RESULTS:
   Outcome: FALLEN
   Reward: -12.45
   Steps: 42
   Height: 0.02m
   Duration: 1.3s
```
**Behavior:** Random movements, immediate falls, no coordination

### Mid Training (10M-50M steps)
```
📈 Step   12,345,678 | SPS     3,456 | Reward 0.456 | Update 246
🎬 VISUALIZING CURRENT POLICY
   Training Step: 12,345,678
   Update: 246
============================================================
📊 VISUALIZATION RESULTS:
   Outcome: ALIVE
   Reward: 45.67
   Steps: 300
   Height: 0.28m
   Duration: 9.2s
```
**Behavior:** Coordinated movements, 1-2 joints at a time, maintains balance

### Late Training (50M+ steps)
```
📈 Step   56,789,012 | SPS     3,678 | Reward 1.234 | Update 1134
🎬 VISUALIZING CURRENT POLICY
   Training Step: 56,789,012
   Update: 1134
============================================================
📊 VISUALIZATION RESULTS:
   Outcome: ALIVE
   Reward: 123.45
   Steps: 300
   Height: 0.35m
   Duration: 9.3s
```
**Behavior:** Smooth standing, minimal joint movement, stable posture

## 🔧 Key Improvements Made

### 1. **Fixed Upside-Down Reward Bug** 🐛
- **Before:** Robot got rewarded for flips (w² always positive)
- **After:** Only rewards upright position (w > 0), -2.0 penalty for upside down

### 2. **Added Joint Movement Limits** 🚦
- **Before:** 10+ joints moving simultaneously
- **After:** Penalizes moving >6 joints at once, encourages 1-2 joints

### 3. **Added Coordination Rewards** 🤝
- **Before:** Random, uncoordinated limb movements
- **After:** Rewards symmetric limb movement patterns

### 4. **Stable Initial Conditions** 🏗️
- **Before:** Extreme joint angles causing immediate flips
- **After:** Moderate, balanced starting pose

## 🎯 Training Strategy for Slow, Steady Learning

### Phase 1: Random Exploration (0-5M steps)
- Robot tries random movements
- Learns that flipping = bad (thanks to -2.0 penalty)
- Discovers that standing = good (height rewards)

### Phase 2: Basic Coordination (5M-20M steps)
- Learns to move 1-2 joints at a time
- Develops basic stability
- Starts lifting torso consistently

### Phase 3: Refined Control (20M-50M steps)
- Smooth, coordinated movements
- Maintains balance for longer periods
- Optimizes joint positions for stability

### Phase 4: Mastery (50M+ steps)
- Effortless standing
- Minimal energy expenditure
- Robust to perturbations

## 📊 Monitoring Tips

### Watch For:
- ✅ **Progressive improvement** in standing duration
- ✅ **Fewer joints moving** simultaneously (1-2 is ideal)
- ✅ **More coordination** between limbs
- ✅ **Higher rewards** over time
- ✅ **Longer episodes** before falling

### Warning Signs:
- ❌ **No improvement** after 10M steps → adjust learning rate
- ❌ **Still flipping** → increase upside-down penalty
- ❌ **Too many joints moving** → increase movement penalty
- ❌ **Not lifting torso** → increase height reward magnitude

## 🎬 Alternative: Step-by-Step Analysis

```bash
# Watch a specific checkpoint in slow motion
python scripts/watch_single.py --checkpoint mjx_00010000000.npz --speed 0.1

# Frame-by-frame analysis
python scripts/watch_single.py --checkpoint mjx_00010000000.npz --step
```

## 🔄 Training Checkpoints

Checkpoints are saved every 500,000 steps to `runs/ape_stand_mjx/`:
- `mjx_000000500000.npz` - 500K steps
- `mjx_000001000000.npz` - 1M steps  
- `mjx_000010000000.npz` - 10M steps
- etc.

Use these to analyze progress at specific milestones!

## 🎯 Expected Timeline

| Steps | Behavior | Reward Range |
|-------|----------|--------------|
| 0-1M | Random flailing, immediate falls | -5 to -1 |
| 1M-5M | Occasional brief standing | -1 to 0.5 |
| 5M-20M | Consistent short standing | 0.5 to 2.0 |
| 20M-50M | Long duration standing | 2.0 to 5.0 |
| 50M+ | Mastery, robust standing | 5.0+ |

Now you can **watch the robot learn to stand slowly and steadily** from scratch! 🚀