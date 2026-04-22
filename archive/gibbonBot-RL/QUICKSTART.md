# Robotic Ape Project — Quick Start Guide

## What You're Building

A 4-limb robotic ape using **4x RoArm M3 Pro** as both arms and legs, mounted on a floating torso. Trained with reinforcement learning to walk, climb, and eventually brachiate — using real ape video as motion reference.

---

## The Stack

| Layer | Tool | Why |
|---|---|---|
| Physics sim | **MuJoCo** | Best contact dynamics, pip install, used by all the key research papers |
| RL training | **MuJoCo MJX + JAX** | Runs on H100, 4096+ parallel envs |
| RL algorithm | **PPO** to start, then **AMP** | PPO for basic locomotion, AMP for natural movement from video |
| Motion reference | **MediaPipe / ViTPose** | Extract ape pose landmarks from YouTube video |
| Robot format | **URDF / xacro** | Describes robot structure, loads directly into MuJoCo |
| Cloud GPU | **Scaleway H100** | ~30 min for basic walking, 2-4 hrs for uneven terrain |

**Not needed right now:** ROS2, MoveIt2, RViz, Gazebo, Isaac Lab. Useful later for real hardware. Ignore for now.

---

## The Robot

- **Torso:** floating base link (free 6DOF in space, not fixed to ground)
- **4 limbs:** 4x RoArm M3, each with 5 DOF + gripper = **24 actuated joints + 6DOF base = 30 DOF total**
- **Stance:** quadruped-style (4-limb), not bipedal
- **Foot contact:** gripper tip of each arm touches the ground

### Joint limits (from the actual URDF):
| Joint | Range |
|---|---|
| Base (yaw) | ±180° — full rotation |
| Shoulder | ±90° |
| Elbow | -57° to +169° |
| Wrist pitch | ±90° |
| Wrist roll | ±180° — full rotation |
| Gripper | 0° to 86° |

### Known gap:
`effort="0"` and `velocity="0"` on all joints — **no motor torque limits defined**. For meaningful force simulation you'll need real servo specs from the RoArm M3 Pro hardware docs.

---

## Immediate Next Steps

### Step 1 — Finish the ROS install (or skip it)
If still installing: you can **cancel it**. For pure simulation you only need:
```bash
pip install mujoco numpy gymnasium stable-baselines3
```
ROS2/MoveIt only matters when talking to real hardware.

### Step 2 — Convert roarm_m3.xacro to a prefixed macro
The existing xacro has hardcoded link names. To use 4 arms in one robot, wrap it in a macro:
```xml
<xacro:macro name="roarm_m3" params="prefix parent *origin">
  <!-- all link/joint names prefixed with ${prefix} -->
</xacro:macro>
```
Then instantiate 4 times with `fl_`, `fr_`, `bl_`, `br_` prefixes.

### Step 3 — Build the ape URDF
```xml
<xacro:roarm_m3 prefix="fl_" parent="torso">
  <origin xyz="0.12  0.10 -0.05" rpy="0  0.785 0"/>  <!-- front-left  -->
</xacro:roarm_m3>
<xacro:roarm_m3 prefix="fr_" parent="torso">
  <origin xyz="0.12 -0.10 -0.05" rpy="0  0.785 0"/>  <!-- front-right -->
</xacro:roarm_m3>
<xacro:roarm_m3 prefix="bl_" parent="torso">
  <origin xyz="-0.12  0.10 -0.05" rpy="0 -0.785 0"/> <!-- back-left   -->
</xacro:roarm_m3>
<xacro:roarm_m3 prefix="br_" parent="torso">
  <origin xyz="-0.12 -0.10 -0.05" rpy="0 -0.785 0"/> <!-- back-right  -->
</xacro:roarm_m3>
```
`xyz` = position on torso. `rpy` = which direction the arm faces (Y rotation tilts it toward the ground).

### Step 4 — Verify in MuJoCo viewer
```python
import mujoco, mujoco.viewer
model = mujoco.MjModel.from_xml_path("ape.urdf")
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```
Drop it, watch it fall. Adjust mount angles until the stance looks right.

### Step 5 — Basic RL (walking on flat ground)
```python
# reward function to start
reward = (
    + forward_velocity * 2.0
    - energy_consumed * 0.01
    - fall_penalty * 10.0
)
```
Run PPO with 4096 parallel envs on H100. Expect basic gait in **30 min – 1.5 hrs**.

### Step 6 — Add terrain curriculum
Start flat → add small bumps → rougher terrain → slopes. Each stage adds 1-2 hrs training.

### Step 7 — AMP (motion from video)
1. Find YouTube video of ape walking/climbing
2. Extract landmarks per frame with MediaPipe:
   ```bash
   pip install mediapipe
   ```
3. Retarget joint angles to robot proportions
4. Feed reference clips to AMP discriminator
5. Robot learns to move like the ape, not like a broken machine

---

## Training Time on H100 (realistic estimates)

| Task | Time |
|---|---|
| Basic 4-limb walking, flat | 30 min – 1.5 hrs |
| Uneven terrain | 2 – 4 hrs |
| Grasping + locomotion | 4 – 8 hrs |
| AMP natural gait | 2 – 6 hrs |
| Brachiation | days, needs curriculum |

---

## What AMP Is (one paragraph)

Adversarial Motion Priors (Berkeley/CMU, 2021). A discriminator trains on real animal video clips and learns to tell "real ape motion" from "robot motion". The robot policy gets rewarded both for completing the task (walk forward) and for fooling the discriminator (look like a real ape). Natural locomotion emerges automatically without hand-tuning any motion parameters. **ASE** (Adversarial Skill Embeddings) extends this to multiple skills — walk, climb, swing — and can blend between them. That's the eventual target.

---

## Files in This Repo That Matter Now

| File | What it is |
|---|---|
| `src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro` | The arm URDF — convert this to a prefixed macro |
| `src/roarm_main/roarm_description/meshes/roarm_m3/` | STL mesh files — reuse for all 4 arms |
| `src/roarm_main/roarm_moveit/config/roarm_m3/joint_limits.yaml` | Joint limits — all zeros for effort/velocity, needs real values |

---

## What to Ignore for Now

- `moveit_servo`, `moveit_task_constructor`, `roarm_moveit_cmd` — motion planning for real hardware
- `roarm_driver` — serial communication with physical arm
- `roarm_moveit_mtc_demo` — task demos, not relevant yet
- RViz — visualizer only, no physics
- Gazebo — worse physics than MuJoCo, more setup pain
