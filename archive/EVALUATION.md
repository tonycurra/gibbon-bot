# Overview — locomotion and control (April–July)

24-DoF robotic ape (4 × Waveshare RoArm-M3). Code: parent archive / `gibbonBot-RL` and siblings. Branch atlas: `WORK_OVERVIEW.md`. What to do next: `NEXT_STEPS.md`.

## Result

The intended goal — walk to the olive tree on command, reliably — **was not achieved**. There is no keeper checkpoint (`.npz`), no MLflow database, and no recorded success metric in the tree. The “demo walk” is **scripted kinematics**, not a learned gait.

What exists is a **software and simulation factory**: MJX/PPO, LIANA dashboard, brain modules that record but do not drive, remote-GPU playbooks. The plant was **not identified against hardware**. Motor numbers came from ST32xx datasheets. Limb collisions were often disabled for speed. Joint homes in the scene do not match firmware (`gibbonBot-RL/data/controls.md`). Training in that simulation is a **different dynamical system** than the real arms.

Policy weights were gitignored (Scaleway / ephemeral GPUs). They are not on this disk. Treat them as lost unless `gibbonbot-training/` still exists.

## Mechanisms

**PPO + dense reward.** On this morphology (asymmetric joint ranges, mirrored limb quaternions), PPO converges to a **static pose**. Extra reward terms did not produce a gait.

**CPG.** Added so PPO would learn corrections, not invent rhythm. At zero policy the CPG **lifts limbs but does not translate the body** (no stance push). Front limbs propel mainly via yaw, rear via pitch.

**Simulation ≠ hardware.** Datasheet damping/kp; self-collision off; contact exclusions; gripper/elbow ranges disagree with firmware homes.

**July PD (`gibbon-brain`).** Cartesian error mapped linearly onto joint indices (**no IK**), no clamp to joint limits, workspace not verified.

**Brain / LIANA.** Logging and trainer UI. They do not enforce limits or produce stance.

**OpenApePose / Go2.** Pose-GMM priors from 2D keypoints (unmerged); equal arm/leg length blocks strict ape imitation. Unitree Go2 is a different robot.

## Goals versus outcome

| Goal | Outcome |
|---|---|
| Walk to tree on command | Not achieved |
| Stand / hold height | Not evidenced in retained artefacts |
| Skill curriculum (stand → walk → turn) | Specified in docs; not completed in artefacts |
| Hardware-valid plant | Not done |
| GPU training throughput | Demonstrated (software/infra) |

## Methods

| Method | Outcome |
|---|---|
| PPO + shaped reward (CPU then MJX) | No gait; static pose |
| Scale envs on H100 / L4 | Throughput worked; metrics not retained |
| CPG + PPO corrections | Right idea; no stance push at zero policy |
| Scripted kinematic demo | Looks like walking; is not a policy |
| PD Cartesian control | Failed without IK and limits |
| Datasheet motors in XML | Hypothesis; not identified on the arm |

## What to keep vs park

**Keep:** MJCF/meshes as a *starting* twin; `controls.md` / `motors.md`; CPG as a prior *after* hardware can crawl; MJX/PPO infra once the env is a trusted twin; fingerprints + keepers with a restore drill; Jade pose priors and Go2 as **references**.

**Park:** LIANA growth, quantum/brain-as-policy, grove/Blender as training deps, reward soup, 16384 envs on an untrusted plant, treating the scripted demo as success.

## Repositories

| Folder | Role |
|---|---|
| `gibbonBot-RL` | Main MJX/PPO/CPG/LIANA/brain stack (`9e2474a`) |
| `gibbon-rl-mujoco` | Early four-arm URDF/xacro |
| `olivabot-mujoco-cpu` | CPU PPO stand-up (`jade` is the later line) |
| `Roarm-m3pro-busServoDriverHAT_A-battery` | Hardware PDF |
| `gibbon-brain` | July PD experiments (no project git) |
