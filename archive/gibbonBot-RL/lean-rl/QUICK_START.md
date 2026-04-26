# 🚀 Lean RL Quick Start

## 📁 What's in this folder

```
lean-rl/
├── README.md          # Detailed explanation
├── QUICK_START.md     # This file (fast reference)
├── test_lean_rl.py    # Setup verification script
├── train.py           # Main training script (RUN THIS)
├── stand_env_mjx.py    # GPU training environment
├── stand_env.py       # Visualization environment
└── scene.xml          # Robot physics definition
```

## 🎯 Just Run This

```bash
# Start training with live visualization
python train.py --watch
```

## 📊 What to Expect

### First 10 minutes (0-1M steps)
- Robot moves randomly
- Immediate falls
- Low rewards (-5 to -1)
- **This is normal!**

### After 1 hour (1M-5M steps)
- Robot moves 1-2 joints at a time
- Occasional brief standing
- Rewards improve (-1 to 0.5)

### After 5 hours (5M-20M steps)
- Consistent short standing
- Coordinated movements
- Positive rewards (0.5 to 2.0)

### After 10+ hours (20M+ steps)
- Long duration standing
- Smooth coordination
- High rewards (2.0+)

## 🔧 Common Commands

```bash
# Start fresh training with visualization
python train.py --watch

# Continue from where you left off
python train.py --inherit --watch

# Just visualize latest progress
python train.py --visualize

# Analyze specific checkpoint
python train.py --visualize mjx_00050000000.npz

# Frame-by-frame debugging
python train.py --visualize --step

# Faster training (more GPU memory needed)
python train.py --envs 2048 --watch

# Slower but more stable training
python train.py --envs 256 --watch
```

## 🎬 Visualization Tips

- **Close MuJoCo window** → Skip current visualization
- **Press 'Q'** → Exit visualization mode
- **--step flag** → Go frame by frame
- **--speed 0.1** → Faster playback
- **--speed 0.02** → Slower, more detailed playback

## 🏆 Success Checklist

- [ ] Robot moves 1-2 joints at a time (not all at once)
- [ ] No immediate backflips when spawning
- [ ] Can stand for >50 steps without falling
- [ ] Achieves height > 0.25m consistently
- [ ] Reward > 1.0 sustained
- [ ] Smooth, coordinated limb movements

## 🔄 Checkpoints

- Saved every **500,000 steps**
- Format: `mjx_00005000000.npz` (500K steps)
- Location: Same folder where you run the script
- Continue training: `python train.py --inherit`

## ⚡ Requirements

```bash
pip install jax mujoco numpy
```

## 🎉 You're Ready!

Just run:
```bash
cd lean-rl
python train.py --watch
```

Then sit back and watch the robot learn to stand **slowly and steadily**!

**No more backflips, no more spastic movements** - just coordinated, stable learning. 🚀