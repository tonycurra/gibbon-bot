# 🎯 Lean RL - Gradual Workflow Summary

## 📁 What You Have

A **clean, minimal folder** with everything needed for gradual, CPU-based testing and analysis:

```
lean-rl/
├── .gitignore          # Proper ignore rules
├── WORKFLOW_SUMMARY.md # This file
├── GRADUAL_WORKFLOW.md # Detailed workflow guide
├── QUICK_START.md      # Fast reference
├── README.md           # Complete documentation
├── analyze_policy.py   # CPU policy analysis
├── scene.xml           # Robot physics definition
├── stand_env.py        # Visualization environment
├── stand_env_mjx.py    # GPU training environment
├── test_physics.py     # Quick physics testing
└── train.py            # Unified training script
```

## 🚀 How to Use It (Gradual Approach)

### 1️⃣ **Test Physics First**
```bash
cd lean-rl
python test_physics.py
```
- Tests passive physics (zero actions)
- See if robot falls immediately
- Adjust `scene.xml` if needed
- Use `--step` for frame-by-frame analysis

### 2️⃣ **Analyze Policies**
```bash
python analyze_policy.py
```
- Loads latest checkpoint
- Runs 10 episodes with detailed metrics
- Use `--step` to see exactly what's happening
- Adjust rewards in `stand_env.py` if needed

### 3️⃣ **Make Small Adjustments**
- Edit `scene.xml` for physics
- Edit `stand_env.py` for rewards  
- Edit `train_mjx.py` for hyperparameters
- Test immediately with the scripts above

### 4️⃣ **Train Small Scale**
```bash
python train.py --envs 64 --watch
```
- Start with small scale (64 environments)
- Watch progress every 10 updates
- Adjust based on what you see

### 5️⃣ **Scale Up Gradually**
```bash
python train.py --envs 256 --watch
python train.py --envs 512 --watch
python train.py --envs 1024      # Full scale
```

## 🎬 Key Commands

| Goal | Command |
|------|---------|
| Test physics | `python test_physics.py` |
| Test with random actions | `python test_physics.py --random` |
| Frame-by-frame physics | `python test_physics.py --step` |
| Analyze latest policy | `python analyze_policy.py` |
| Analyze specific checkpoint | `python analyze_policy.py mjx_00010000000.npz` |
| Slow motion analysis | `python analyze_policy.py --speed 0.02` |
| Frame-by-frame policy | `python analyze_policy.py --step` |
| Small scale training | `python train.py --envs 64 --watch` |
| Medium scale training | `python train.py --envs 256 --watch` |
| Full scale training | `python train.py --envs 1024` |

## 📊 What to Look For

### Physics Testing
- ✅ Robot doesn't fall immediately with zero actions
- ✅ Joints move smoothly (no oscillation)
- ✅ Contacts are stable (no jittering)
- ✅ Passive damping works well

### Policy Analysis
- ✅ Moves 1-2 joints at a time (coordination working)
- ✅ Avoids upside-down positions (penalty working)
- ✅ Lifts torso gradually (learning working)
- ✅ Maintains balance >100 steps (stability working)

## 🔧 Files to Edit

### Physics (`scene.xml`)
```xml
<!-- Adjust these for stability -->
<joint damping="0.1" />
<geom friction="0.8" />
<body mass="1.2">
```

### Rewards (`stand_env.py`)
```python
# Adjust these for desired behavior
upside_down_penalty = -2.0  # Try -1.0 to -3.0
coordination_score = 0.05   # Try 0.02 to 0.10
too_many_joints_penalty = -0.1  # Try -0.05 to -0.2
```

### Training (`train_mjx.py`)
```python
# Adjust these for learning
LR = 3e-4        # Try 1e-4 to 5e-4
ENT_COEF = 0.03  # Try 0.01 to 0.05
```

## 🎯 Success Criteria

**Physics is good when:**
- Robot can stand passively (zero actions) for >50 steps
- No oscillation or jittering
- Smooth joint movements

**Policy is good when:**
- Moves 1-2 joints at a time
- Achieves height >0.3m consistently
- Maintains balance >200 steps
- Reward > 2.0 sustained

**Ready for GPU when:**
- Physics is stable
- Rewards make sense
- You understand the behavior
- Small scale training works

## 📈 Expected Progress

| Phase | Duration | Goal | Tool |
|-------|----------|------|------|
| Physics | 10-30 min | Stable passive physics | `test_physics.py` |
| Rewards | 30-60 min | Good reward signals | `test_physics.py` + manual edits |
| Small Training | 1-2 hours | Basic standing | `train.py --envs 64` |
| Medium Training | 4-8 hours | Consistent standing | `train.py --envs 256` |
| Full Training | 10+ hours | Mastery | `train.py --envs 1024` |

**Key:** Spend time in early phases to save GPU time!

## 🎓 Pro Tips

1. **Start simple**: Test with zero actions first
2. **Change one thing**: Isolate variables
3. **Observe carefully**: Use `--step` mode
4. **Document changes**: Keep notes
5. **Be patient**: Good physics takes iteration

```
❌ Bad: Change 5 things → Train 10 hours → Wonder what worked
✅ Good: Change 1 thing → Test 5 min → Observe → Decide
```

## 🚀 You're Ready!

Just run:
```bash
cd lean-rl
python test_physics.py      # Start with physics
python analyze_policy.py    # Then analyze policies
python train.py --watch     # Finally train
```

**Gradual, careful, CPU-based workflow** - exactly what you wanted! 🐢 → 🏋️ → 🚀