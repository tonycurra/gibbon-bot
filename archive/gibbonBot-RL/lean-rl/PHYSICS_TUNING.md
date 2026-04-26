# 🎛️ Physics Tuning Guide

## 🔍 Diagnosing the Backflip Issue

**Symptoms:** Robot does immediate backflip on spawn

**Likely causes:**
1. **Low joint damping** - joints don't resist motion
2. **Unstable initial pose** - center of mass issues
3. **High initial noise** - too much randomness
4. **Weak gravity compensation** - not enough torque

## 🔧 Quick Fixes to Try

### 1. Increase Joint Damping (scene.xml)

**Current:** `damping="2"` (too low)

**Try:** `damping="5"` to `damping="10"`

```xml
<!-- Change ALL joint damping values -->
<joint name="*" damping="5" />  <!-- Try 5-10 -->
```

**Effect:** Joints will resist motion more, preventing sudden flips

### 2. Stabilize Initial Pose (stand_env_mjx.py)

**Current:** Moderate angles with noise

**Try:** More conservative pose + less noise

```python
# More stable initial pose
_STAND_INIT = np.array([
    # Front Left limb - more conservative
    0.1,  # base joint - less outward
    0.3,  # shoulder - less forward
    1.0,  # elbow - less bent
    0.0, 0.0, 0.0,
    # ... (repeat for all limbs)
], dtype=np.float32)

# Reduced noise
noise = jax.random.uniform(k_jnt, (24,), minval=-0.1, maxval=0.1)  # ±0.1 instead of ±0.2
```

### 3. Increase Torque Limits (scene.xml)

**Current:** Limited torque range

**Try:** Higher force limits

```xml
<joint actuatorfrcrange="-5 -5" />  <!-- Try -10 to 10 -->
```

### 4. Adjust Center of Mass (scene.xml)

**Current:** Default CoM positions

**Try:** Lower CoM for stability

```xml
<inertial pos="0 0 -0.02" />  <!-- Move CoM down slightly -->
```

## 🧪 Testing Procedure

```bash
# After each change:
cd lean-rl
python test_physics.py --step
```

**Observe:**
- Does robot fall immediately? → Need more damping/torque
- Does robot oscillate? → Need more damping
- Does robot stand briefly? → Good progress!

## 📊 Recommended Settings

### Conservative Settings (Most Stable)
```xml
<!-- scene.xml -->
<joint damping="10" />          <!-- High damping -->
<joint actuatorfrcrange="-10 10" /> <!-- High torque -->
```

```python
# stand_env_mjx.py
_STAND_INIT = np.array([0.1, 0.3, 1.0, 0.0, 0.0, 0.0, ...])  # Conservative
noise = jax.random.uniform(k_jnt, (24,), minval=-0.1, maxval=0.1)  # Low noise
```

### Balanced Settings (Good Stability + Movement)
```xml
<!-- scene.xml -->
<joint damping="5" />           <!-- Medium damping -->
<joint actuatorfrcrange="-5 5" />  <!-- Medium torque -->
```

```python
# stand_env_mjx.py
_STAND_INIT = np.array([0.2, 0.4, 1.2, 0.0, 0.0, 0.0, ...])  # Moderate
noise = jax.random.uniform(k_jnt, (24,), minval=-0.15, maxval=0.15)  # Moderate noise
```

### Aggressive Settings (More Movement, Less Stability)
```xml
<!-- scene.xml -->
<joint damping="2" />           <!-- Low damping -->
<joint actuatorfrcrange="-3 3" />  <!-- Low torque -->
```

```python
# stand_env_mjx.py
_STAND_INIT = np.array([0.3, 0.5, 1.5, 0.0, 0.0, 0.0, ...])  # Aggressive
noise = jax.random.uniform(k_jnt, (24,), minval=-0.2, maxval=0.2)  # High noise
```

## 🎯 Step-by-Step Tuning

### Step 1: Start with Maximum Stability
```bash
# Set conservative values first
# Test with zero actions
python test_physics.py
```

### Step 2: Gradually Reduce Damping
```bash
# Once stable, reduce damping slightly
# Find the minimum damping that prevents flips
python test_physics.py
```

### Step 3: Adjust Initial Pose
```bash
# Find pose that stands passively
# Then add small noise
python test_physics.py
```

### Step 4: Test with Random Actions
```bash
# Once passive physics works
# Test with random actions
python test_physics.py --random
```

## 📈 Physics Parameters Cheat Sheet

| Parameter | Too Low | Just Right | Too High |
|-----------|---------|------------|----------|
| Damping | Flips, oscillates | Smooth movement | Stiff, slow |
| Torque limit | Can't move | Good control | Jerky |
| Initial pose | Unstable | Balanced | Too rigid |
| Noise | Unpredictable | Varied | Too random |

## 🔬 Real Gravity Testing

To test with **real gravity effects**, use:

```bash
# Test passive physics (zero actions = pure gravity test)
python test_physics.py

# Watch for:
- Does robot fall immediately? → Gravity too strong OR damping too weak
- Does robot oscillate? → Damping needs adjustment
- Does robot stand still? → Good balance
```

### Gravity Adjustment

If gravity seems too strong in `scene.xml`:

```xml
<!-- MuJoCo uses Earth gravity by default (9.81 m/s²) -->
<!-- To reduce gravity effect, increase damping or torque -->

<option gravity="0 0 -9.81">  <!-- Default Earth gravity -->
<!-- Try reducing if needed -->
<option gravity="0 0 -5.0">    <!-- Half gravity for testing -->
```

## 🏆 Success Criteria

**Physics is tuned well when:**
- ✅ Robot stands >100 steps with zero actions
- ✅ No oscillation or jittering
- ✅ Smooth response to disturbances
- ✅ Can recover from small pushes

**Then move to:**
```bash
python analyze_policy.py  # Test learned policies
python train.py --envs 64 --watch  # Small scale training
```

## 🎓 Pro Tips

1. **Start conservative** - High damping, stable pose
2. **Change one thing at a time** - Isolate variables
3. **Test with zero actions first** - Pure physics test
4. **Use --step mode** - See exactly what happens
5. **Document changes** - Keep track of what works

```
❌ Bad: Change 5 parameters → Test → Wonder what helped
✅ Good: Change damping → Test → Observe → Decide
```

**Now you can systematically tune the physics!** Start with high damping and work down gradually. 🎛️