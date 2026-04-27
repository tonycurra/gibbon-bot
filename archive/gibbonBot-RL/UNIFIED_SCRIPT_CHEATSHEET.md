# 🎯 UNIFIED TRAINING SCRIPT - CHEAT SHEET

## 📋 One Script to Rule Them All

```bash
python scripts/train.py [options]
```

## 🚀 QUICK START

### 1. **Fast Training Only** (No visualization)
```bash
python scripts/train.py
```
- Fastest GPU training
- No interruptions
- Best for overnight training

### 2. **Train + Watch Progress** (Recommended)
```bash
python scripts/train.py --watch
```
- Shows current policy every 10 training updates
- See slow, steady improvement
- Press 'Q' in MuJoCo window to skip visualizations

### 3. **Just Visualize Existing Results**
```bash
python scripts/train.py --visualize
```
- Shows latest checkpoint
- No training, just visualization
- Great for checking progress

## 🎛️ CUSTOMIZATION OPTIONS

### Training Options
```bash
# Use fewer environments (more stable, slower)
python scripts/train.py --envs 256

# Continue from where you left off
python scripts/train.py --inherit

# Fast training with frequent checkpoints
python scripts/train.py --envs 512 --watch --vis-freq 5
```

### Visualization Options
```bash
# Watch specific checkpoint
python scripts/train.py --visualize mjx_00010000000.npz

# Frame-by-frame analysis (press Enter for each step)
python scripts/train.py --visualize --step

# Faster playback
python scripts/train.py --visualize --speed 0.1

# Slower playback (more detail)
python scripts/train.py --visualize --speed 0.02
```

## 📈 TYPICAL WORKFLOW

### Day 1: Start Training
```bash
# Start training with visualization
python scripts/train.py --watch --envs 512

# Let it run, observe progress every 10 updates
# Press 'Q' to skip visualizations if needed
# Ctrl+C to stop when satisfied
```

### Day 2: Check Results
```bash
# See what you trained yesterday
python scripts/train.py --visualize

# Analyze specific checkpoint in detail
python scripts/train.py --visualize mjx_00050000000.npz --step
```

### Day 3: Continue Training
```bash
# Continue from best checkpoint
python scripts/train.py --inherit --watch

# Use more environments for faster training
python scripts/train.py --inherit --envs 1024
```

## 🎬 VISUALIZATION FEATURES

### Real-time Metrics
During visualization, you'll see:
```
📊 Step: 1234 | Reward: 45.67 | Height: 0.28m | Status: ALIVE
```

### Episode Summary
After each episode:
```
📈 EPISODE SUMMARY:
   Outcome: ALIVE
   Total Reward: 123.45
   Steps: 300
   Height: 0.35m
   Duration: 9.3s
```

## 🔧 TROUBLESHOOTING

### Problem: Robot still flipping
**Solution:** The coordination fixes are already in place. If still flipping:
- Train longer (need more steps)
- Try continuing from checkpoint: `--inherit`
- Reduce environments: `--envs 256`

### Problem: Training too slow
**Solution:** 
- Use more environments: `--envs 2048` (needs more GPU VRAM)
- Disable visualization: remove `--watch`
- Train overnight without visualization

### Problem: Want to see progress more/less often
**Solution:**
```bash
# More frequent visualization
python scripts/train.py --watch --vis-freq 5

# Less frequent visualization  
python scripts/train.py --watch --vis-freq 20
```

## 📊 EXPECTED PROGRESS

| Training Steps | Behavior | Command to Check |
|---------------|----------|------------------|
| 0-1M | Random movements, immediate falls | `--visualize` |
| 1M-5M | Occasional brief standing | `--visualize` |
| 5M-20M | Consistent short standing | `--visualize mjx_00100000000.npz` |
| 20M-50M | Long duration standing | `--visualize mjx_00200000000.npz` |
| 50M+ | Mastery, robust standing | `--visualize mjx_00500000000.npz` |

## 🎯 PRO TIPS

### 1. **Multi-terminal Workflow**
- **Terminal 1:** Training (`python scripts/train.py`)
- **Terminal 2:** Visualization (`python scripts/train.py --visualize`)
- **Terminal 3:** Monitoring (`watch nvidia-smi`)

### 2. **Checkpoint Management**
- Checkpoints saved every 500K steps to `runs/ape_stand_mjx/`
- Format: `mjx_000000500000.npz` (500K steps)
- Latest checkpoint used automatically with `--inherit`

### 3. **Performance Monitoring**
- **SPS** = Steps Per Second (higher is better)
- **Reward** = Average reward per step (should increase over time)
- **Update** = Number of policy updates

## 🎉 SUCCESS CRITERIA

You're doing great when:
✅ **Reward > 0.5** - Robot can stand briefly
✅ **Reward > 2.0** - Robot can stand consistently  
✅ **Reward > 5.0** - Robot masters standing
✅ **Steps > 300** - Robot maintains balance for full episode
✅ **Height > 0.3m** - Robot reaches target height

Now you have **one powerful script** that replaces all the others! 🚀