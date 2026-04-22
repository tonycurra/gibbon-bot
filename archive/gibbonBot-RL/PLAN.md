# Plan — What to Try Next

## Where we are

The robot is training on CPU/GPU with PPO to reach a tree 5m away.
Best result so far: closest approach ~2.78m (tree success = 1.5m).
Training is functional. Viewer works. GPU (RTX 4070) is now enabled.

---

## Immediate (do these first)

### 1. Let training run to completion on GPU
- 10M steps should now finish in ~30–60 min with RTX 4070
- Watch with `watch_multi.py --n 4` to see if the robot learns to walk
- Goal: see `closest` drop below 1.5m and `SUCCESS` episodes appear

### 2. Scale up training for GPU
Now that GPU works, increase parallelism in `train.py`:
```python
N_ENVS = 32       # was 8 — GPU can handle many more parallel envs
batch_size = 1024  # was 256
```

### 3. Curriculum: start tree closer
The 5m goal is hard from scratch. Try a curriculum where the tree starts at 2m and moves further as the robot improves. Add `TREE_POS` as a parameter to `ApeEnv` and vary it via a callback.

---

## Short term (once basic locomotion works)

### 4. Tune mount angles for better stance
Arms currently point straight down (90°). A quadruped stance needs arms angled outward and forward. Experiment with `rpy` in the 4 `torso_to_*_base_link` joints in `ape.urdf`.

### 5. Add feet / contact sites
Add MuJoCo `<site>` elements at each gripper tip to read contact forces. Feed contact info into the observation to help the robot learn to push off the ground properly.

### 6. Reward shaping improvements
- Add upright bonus: reward torso staying level (penalise tilt from vertical)
- Add foot-on-ground bonus: at least 2 gripper tips in contact with terrain
- Reduce alive bonus (0.1) once basic locomotion works — it can mask bad behaviour

### 7. Hyperparameter sweep
Try:
- `ent_coef`: 0.01 → 0.02 (more exploration early)
- `gamma`: 0.99 → 0.995 (value longer-horizon rewards more)
- `learning_rate`: 3e-4 with linear decay schedule

---

## Medium term

### 8. Motion priors (AMP)
Record or download ape locomotion video. Extract skeleton keypoints (MediaPipe / ViTPose). Use Adversarial Motion Priors to make the robot move like a real ape rather than discovering an arbitrary gait.

### 9. Obstacle course / harder terrain
- Steeper hills, rocks, gaps
- Multiple trees at varying distances
- Require navigation (don't just move toward fixed TREE_POS)

### 10. Sim-to-real transfer prep
- Verify real STS3215 torque curve matches URDF limits
- Add motor latency / noise in simulation
- Add domain randomisation (mass, friction, joint stiffness perturbation)

---

## Later

### 11. Brachiation
Once quadruped locomotion works, teach the robot to swing between branches — the original goal. Needs: overhead grip sites, branch objects, hang/swing reward shaping.

### 12. Real hardware deployment
- Connect to physical RoArm M3 Pro units via ROS2
- Use trained policy for sim-to-real transfer
- Close the loop with onboard camera + depth sensor
