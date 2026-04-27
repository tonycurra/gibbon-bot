# Robot Training Feedback Guide

Use this guide to provide structured feedback when observing robot simulations.

## How to Use the Single Viewer

```bash
# Slow continuous playback (default)
python scripts/watch_single.py

# Frame-by-frame analysis
python scripts/watch_single.py --step

# Custom speed (e.g., 0.1 seconds per frame)
python scripts/watch_single.py --speed 0.1

# Specific checkpoint
python scripts/watch_single.py --checkpoint mjx_000001000000.npz
```

## Observation Checklist

### 1. **Posture Issues**
- [ ] Robot frequently flips upside down
- [ ] Robot leans excessively to one side
- [ ] Arms/legs in unnatural positions
- [ ] Joints hyper-extended or overly bent

### 2. **Movement Patterns**
- [ ] Erratic, jerky movements
- [ ] Repetitive unsuccessful strategies
- [ ] Lack of coordination between limbs
- [ ] Overuse of specific joints

### 3. **Stability Problems**
- [ ] Falls immediately after reset
- [ ] Unable to maintain balance
- [ ] Oscillations or shaking
- [ ] Slow drift in one direction

### 4. **Reward System Feedback**
- [ ] Upside-down penalty working effectively
- [ ] Height rewards appropriate
- [ ] Upright rewards encouraging good posture
- [ ] Energy penalties preventing excessive movement

## Detailed Feedback Template

```
**Checkpoint:** mjx_[step_number].npz
**Episode Outcome:** ALIVE/FALLEN after [X] steps
**Total Reward:** [Y]

**Observed Issues:**
1. [Description of main problem]
2. [Description of secondary problem]

**Positive Behaviors:**
1. [What the robot is doing well]
2. [Any emerging good strategies]

**Suggested Improvements:**
- [Specific training parameter adjustments]
- [Reward function modifications]
- [Curriculum learning suggestions]
```

## Common Problems & Potential Fixes

### Problem: Robot flips upside down frequently
**Possible causes:**
- Insufficient upside-down penalty
- Reward function encourages extreme movements
- Learning rate too high causing unstable updates

**Potential fixes:**
- Increase upside-down penalty (already set to -2.0)
- Add gradient clipping or reduce learning rate
- Modify curriculum to start with easier tasks

### Problem: Robot doesn't lift off ground
**Possible causes:**
- Height reward too weak
- Energy penalty too strong
- Initial joint positions unfavorable

**Potential fixes:**
- Increase height reward magnitude
- Temporarily reduce energy penalty
- Adjust initial spawn position

### Problem: Erratic, uncontrolled movements
**Possible causes:**
- High learning rate
- Low entropy coefficient
- Insufficient action smoothing penalty

**Potential fixes:**
- Reduce learning rate
- Increase entropy coefficient
- Add stronger action delta penalty

## Technical Details Displayed

During playback, you'll see real-time metrics:
- **Step:** Current step in episode
- **Reward:** Cumulative reward
- **Height:** Torso height above ground (m)
- **Upright:** Upright reward component
- **Status:** ALIVE or FALLEN

Use these to correlate visual behavior with reward signals!