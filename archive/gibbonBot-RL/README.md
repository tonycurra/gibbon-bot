# Robotic Ape Project

A 4-limb robotic ape built from **4x RoArm M3 Pro** arms, trained with reinforcement learning to walk, climb, and eventually brachiate — using real ape video as motion reference.

---

## Status

| Layer | Tool | Status |
|-------|------|--------|
| Physics sim | MuJoCo 3.x | ✅ Working |
| Robot description | URDF → MJCF | ✅ Working |
| RL training | SB3 PPO + GPU | ✅ Working |
| Live viewer | `watch_multi.py` | ✅ Working |
| Motion reference | AMP / ViTPose | ⬜ Next |
| Real hardware | ROS2 / MoveIt | ⬜ Later |

---

## The robot

- **Torso**: 2 kg floating base (6 DOF free joint), 30×20×10 cm
- **4 limbs**: RoArm M3 Pro, each with 5 DOF + gripper = **24 actuated joints**
- **Total mass**: ~3.8 kg (2 kg torso + ~0.45 kg × 4 arms)
- **Actuators**: position-controlled, kp=20, torque limits from STS3215 servo datasheet

```
        [fl_]  [fr_]
          |      |
        +---------+
        |  torso  |
        +---------+
          |      |
        [bl_]  [br_]
```

---

## File map

```
gb2.1/
│
├── README.md               ← you are here
├── DEVLOG.md               ← full change history with rationale
├── PLAN.md                 ← next steps and ideas
├── QUICKSTART.md           ← original project roadmap
│
│── Training & eval
├── train.py                ← PPO training (GPU auto-detected), saves to runs/ape_reach/
├── eval.py                 ← evaluate a saved checkpoint, prints episode stats
├── watch.py                ← persistent single-env viewer, hot-swaps checkpoints
├── watch_multi.py          ← split-screen viewer (N envs in a grid), hot-swaps policy
│
│── Simulation core
├── ape_env.py              ← Gymnasium environment (obs, action, reward)
├── terrain.py              ← generates 64×64 agricultural heightfield at runtime
├── build_scene.py          ← converts ape.urdf → scene.xml (run after any URDF edit)
├── scene.xml               ← generated MJCF — do NOT edit by hand
│
│── Dev tools
├── view_robot.py           ← MuJoCo viewer (full physics, grass terrain)
├── test_joints.py          ← manual joint tester (gravity off, torso pinned, Ctrl+A)
│
└── roarm_ws/
    └── roarm_description/
        ├── ape.urdf            ← EDIT THIS to change the robot
        ├── urdf/roarm_m3/
        │   └── roarm_m3.xacro  ← original single-arm URDF (DO NOT EDIT)
        └── meshes/roarm_m3/    ← STL mesh files (DO NOT EDIT)
```

---

## Quick start

```bash
source .venv/bin/activate

# View the robot on terrain
python view_robot.py

# Train (uses GPU if available, CPU otherwise)
python train.py

# Watch training live — split screen, N parallel envs
python watch_multi.py           # 2×2 grid
python watch_multi.py --n 8     # 2×4 grid

# Evaluate a checkpoint
python eval.py
python eval.py --model runs/ape_reach/ape_reach_50000_steps.zip
```

### Two-terminal workflow

```
Terminal 1:  python train.py
Terminal 2:  python watch_multi.py --n 4
```

The viewer auto-loads new checkpoints as training saves them.

### After editing ape.urdf

```bash
python build_scene.py   # regenerates scene.xml
python view_robot.py    # verify visually
```

---

## Environment details

| Item | Value |
|------|-------|
| Observation | 63-dim: torso z/quat/vel + 24 joint pos/vel + tree direction + distance |
| Action | 24-dim joint position targets, normalised to [-1, 1] |
| Reward | `progress×2 + 0.1 - 5e-4×ctrl² - 1e-3×jvel²` |
| Termination | Fallen (torso < 0.15m) or success (dist < 1.5m) |
| Timeout | 2000 steps |
| Goal | Reach the tree at (5, 0, 0) — 5m from spawn |

---

## Git branches

| Branch | Purpose |
|--------|---------|
| `master` | baseline snapshot of original ROS2 package — nothing changed |
| `sim/mujoco-setup` | all simulation and RL work |
