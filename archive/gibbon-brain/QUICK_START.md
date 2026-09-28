# Quick Start: Agentic Control for GibbonBot

## 1. Run the Agent

```bash
cd /home/t/code/gibbon-brain
python3 agent_control.py
```

**This will:**
- Launch the MuJoCo viewer with the gibbonBot
- The FL (front-left) gripper will follow a circular path
- Data will be logged to `logs/agent_run_*.csv`
- After you close the viewer (press ESC), you'll see analysis printed

## 2. Try Different Modes

Edit `agent_control.py` and change the `MODE` variable:

```python
MODE = 'CIRCLE'   # FL gripper follows a circle
MODE = 'SQUARE'   # FL gripper follows a square
MODE = 'REACH'    # FL gripper reaches for a fixed point
MODE = 'STAND'    # All limbs try to stand up
```

## 3. Adjust Parameters

Tune the controller by editing these in `agent_control.py`:

```python
KP = 30.0   # Position gain - higher = faster but may oscillate
KD = 3.0    # Damping gain - higher = smoother but slower
PATH_RADIUS = 0.25  # Size of the path
PATH_SPEED = 0.8    # How fast to follow the path
```

## 4. Evaluate Results

After running, evaluate with:

```bash
python3 evaluate.py           # Evaluate latest run
python3 evaluate.py --all     # Evaluate all runs
python3 evaluate.py --mode CIRCLE  # Only CIRCLE mode runs
```

**What to look for in the viewer:**

### Good Behavior
- [ ] FL gripper follows the target path smoothly
- [ ] Robot doesn't shake or vibrate excessively
- [ ] Torso stays relatively stable (doesn't drift much)
- [ ] Other limbs don't flail wildly
- [ ] Robot doesn't fall over

### Bad Behavior (and likely causes)

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Gripper doesn't move at all | KP too low | Increase KP |
| Gripper shakes/vibrates | KP too high or KD too low | Decrease KP, increase KD |
| Gripper moves slowly | KP too low | Increase KP |
| Gripper overshoots target | KP too high | Decrease KP, increase KD |
| Robot falls over | Path too far from center | Decrease PATH_RADIUS |
| Torso drifts | Control affecting torso | Check joint mappings |
| Robot moves jerkily | DT too large | Decrease DT |
| Limbs move in wrong direction | Sign error in control | Check control mapping |

## 5. Understanding the Data

The CSV log file contains:
- **Target positions**: Where we wanted the gripper to go
- **Actual positions**: Where the gripper actually went
- **Error**: Difference between target and actual
- **Control signals**: What commands were sent to joints
- **Torso position**: Where the main body is

**Key metrics from evaluation:**
- **Error Mean/Max**: How far off from target (lower is better, <0.05m is good)
- **Path Fidelity**: How well the path was followed (higher % is better)
- **Oscillation Score**: How much shaking (lower is better)
- **Torso Movement**: How much the body moved (lower is better, <0.05m is good)

## 6. Visualization

The `evaluate.py` script generates plots showing:
1. **XY Trajectory**: Target path vs actual gripper path
2. **Error over Time**: How tracking error changes
3. **Control Signals**: Joint commands sent
4. **End Effector Heights**: Z positions of all limbs

## 7. Troubleshooting

**If you see Python errors:**
```bash
pip install mujoco numpy pandas matplotlib
```

**If the viewer doesn't open:**
- Make sure you have a working OpenGL display
- Try running on a machine with a GPU

**If the robot looks wrong:**
- Check that the scene file loads correctly
- Verify mesh files exist in `gibbonBot-RL-brain-modeling_data/gibbonbot-rl/meshes/`

## 8. Next Steps

Once you have basic control working, you can:

1. **Implement more complex paths**: Edit the target generators
2. **Add obstacle avoidance**: Check for collisions, adjust path
3. **Coordinate multiple limbs**: Control FR, BL, BR together with FL
4. **Implement IK**: Use proper inverse kinematics instead of PD
5. **Add learning**: Use RL to optimize the controller

## Quick Feedback Loop

```bash
# Run agent
python3 agent_control.py

# Describe what you see:
# - Does the gripper follow the circle?
# - Is it smooth or jerky?
# - Does the robot stay stable?

# Evaluate
python3 evaluate.py

# Adjust parameters based on results
nano agent_control.py  # or edit in your IDE

# Repeat!
```
