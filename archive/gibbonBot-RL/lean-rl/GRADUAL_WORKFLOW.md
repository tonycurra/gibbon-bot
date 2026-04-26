# 🐢 Gradual Testing & Analysis Workflow

## 🎯 Philosophy: Slow, Careful, CPU-Based

**No GPU needed** → Use CPU for careful observation
**No rush** → Take time to analyze behavior
**No guesswork** → Test physics changes immediately

## 🔬 Step 1: Test Physics Changes

```bash
# Quick physics test with zero actions (passive physics)
python test_physics.py

# Test with random actions
python test_physics.py --random

# Frame-by-frame analysis
python test_physics.py --step

# Adjust speed for detailed observation
python test_physics.py --speed 0.1
```

**What to observe:**
- Does the robot fall immediately? → Physics too unstable
- Does it maintain balance passively? → Good stability
- Do joints move smoothly? → Good damping
- Are collisions handled properly? → Check contact points

## 🔍 Step 2: Analyze Learned Policies

```bash
# Analyze latest checkpoint
python analyze_policy.py

# Analyze specific checkpoint
python analyze_policy.py mjx_00010000000.npz

# Slow motion analysis
python analyze_policy.py --speed 0.02

# Frame-by-frame policy analysis
python analyze_policy.py --step
```

**What to observe:**
- Does policy move 1-2 joints at a time? → Good coordination
- Does it avoid backflips? → Upside-down penalty working
- Does it lift torso gradually? → Good learning
- Are movements symmetric? → Good coordination rewards

## 🏋️ Step 3: Iterative Improvement

```bash
# Make small physics change in scene.xml
# Test immediately
python test_physics.py

# Adjust reward function in stand_env.py
# Test policy behavior
python analyze_policy.py

# Tweak hyperparameters in train_mjx.py
# Train small batch
python scripts/train.py --envs 64 --watch --vis-freq 2
```

**Iteration cycle:**
```
Edit physics/rewards → Test → Observe → Adjust → Repeat
```

## 📊 Step 4: Detailed Metrics

### Physics Test Metrics
```
Step: 1234 | Height: 0.28m | Reward: -0.45 | Status: ALIVE
```

**Good values:**
- Height > 0.1m: Robot can partially stand
- Reward > -1.0: Not falling immediately
- Steps > 100: Some stability

### Policy Analysis Metrics
```
Episode 1/10
  Step: 300 | Reward: 45.67 | Height: 0.32m | Status: ALIVE
  📊 Summary: ALIVE | Reward: 45.67 | Steps: 300
```

**Good values:**
- Reward > 20: Learning progress
- Height > 0.3m: Reaching target
- Steps > 200: Good stability
- ALIVE status: No falls

## 🎬 Visualization Controls

### Physics Test
- **ESC**: Exit viewer
- **Mouse**: Rotate/zoom scene
- **Space**: Advance frame (in --step mode)
- **Q**: Quit early

### Policy Analysis
- **Close window**: Exit
- **Enter**: Next frame (in --step mode)
- Runs 10 episodes automatically
- Shows detailed metrics per episode

## 🔧 Common Adjustments

### Physics Tweaks (scene.xml)
```xml
<!-- Adjust joint damping -->
<joint name="*" damping="0.1" />

<!-- Adjust friction -->
<geom friction="0.8" />

<!-- Adjust mass -->
<body mass="1.2">
```

### Reward Tweaks (stand_env.py)
```python
# Adjust upside-down penalty
upside_down_penalty = -2.0 if w < 0 else 0.0

# Adjust coordination reward
coordination_score = 0.05 * limb_similarity

# Adjust joint movement penalty
too_many_joints_penalty = -0.1 * (joints_moving - 6)
```

### Training Tweaks (train_mjx.py)
```python
# Learning rate
LR = 3e-4  # Try 1e-4 to 5e-4

# Entropy coefficient
ENT_COEF = 0.03  # Try 0.01 to 0.05

# Joint movement limit
WATCH_EVERY = 6  # Try 4 to 8
```

## 🎯 Gradual Improvement Checklist

### Physics Stability
- [ ] Robot doesn't fall immediately with zero actions
- [ ] Joints move smoothly without oscillation
- [ ] Contacts are stable (no jittering)
- [ ] Passive damping works well

### Policy Quality
- [ ] Moves 1-2 joints at a time
- [ ] Avoids upside-down positions
- [ ] Lifts torso gradually
- [ ] Maintains balance >100 steps
- [ ] Achieves height >0.3m consistently

### Coordination
- [ ] Limbs move symmetrically
- [ ] No spastic, random movements
- [ ] Smooth transitions between poses
- [ ] Energy-efficient movements

## 🚀 When to Move to GPU Training

**Move to GPU when:**
1. ✅ Physics is stable (test_physics.py works well)
2. ✅ Rewards make sense (manual testing shows good behavior)
3. ✅ Hyperparameters are reasonable
4. ✅ You understand what the robot should do

**Then run:**
```bash
# Start with small scale
python scripts/train.py --envs 64 --watch

# Scale up gradually
python scripts/train.py --envs 256 --watch
python scripts/train.py --envs 512 --watch
python scripts/train.py --envs 1024  # Full scale
```

## 📈 Expected Progress Timeline

| Phase | Tool | Duration | Goal |
|-------|------|----------|------|
| Physics Testing | `test_physics.py` | 10-30 min | Stable passive physics |
| Reward Tuning | Manual editing + `test_physics.py` | 30-60 min | Good reward signals |
| Small-Scale Training | `train.py --envs 64` | 1-2 hours | Basic standing |
| Medium-Scale Training | `train.py --envs 256` | 4-8 hours | Consistent standing |
| Full-Scale Training | `train.py --envs 1024` | 10+ hours | Mastery |

**Key:** Spend time in early phases to avoid wasting GPU time on bad physics/rewards!

## 🎓 Pro Tips

1. **Start simple**: Test with zero actions first
2. **Isolate variables**: Change one thing at a time
3. **Observe carefully**: Use --step mode for critical moments
4. **Document changes**: Keep notes on what works
5. **Be patient**: Good physics takes iteration

```
Bad: Change 5 things → Train 10 hours → Wonder what worked
Good: Change 1 thing → Test 5 min → Observe → Decide
```

**Now you have a complete gradual workflow!** 🐢 → 🏋️ → 🚀