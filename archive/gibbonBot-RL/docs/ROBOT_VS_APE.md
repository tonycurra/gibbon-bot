# Robot joint → Ape anatomy mapping

The robot has **24 actuators**: 6 per limb × 4 limbs (FL = front-left, FR = front-right, BL = back-left, BR = back-right). Plus a **rigid torso** (single body, free-floating).

Real apes have ~30 articulated DOF in arms + legs alone, plus a flexible spine. The robot is a simplification.

## Per-limb joint chain (6 DOF, robot)

Joint order in `scene.xml` and the policy action vector (`a[0]..a[5]` for FL):

| Idx | Robot joint | Ape equivalent | Function |
|-----|-------------|----------------|----------|
| 0 | `shoulder_yaw` / `hip_yaw` | Shoulder/hip horizontal abduction-adduction | Rotate limb around vertical axis (turn it inward/outward at the body) |
| 1 | `shoulder_pitch` / `hip_pitch` | Shoulder/hip flexion-extension | Swing limb forward/back (the main propulsive joint) |
| 2 | `elbow` / `knee` | Elbow/knee flexion-extension | Bend the limb at its midpoint |
| 3 | `forearm_roll` / `shin_roll` | Forearm pronation-supination / tibial rotation | Rotate the lower limb around its long axis (palm rotation in arm) |
| 4 | `wrist` / `ankle` | Wrist flexion-extension / ankle dorsiflexion-plantarflexion | Flex the hand/foot at the base |
| 5 | `gripper` / `foot` | Finger grip (composite) / toe grip | 1-DOF grasp — apes have grasping feet too (unlike humans) |

Action vector layout (24-dim): `[FL_0..FL_5, FR_0..FR_5, BL_0..BL_5, BR_0..BR_5]`.

## Limb-to-ape mapping

| Robot limb | Ape equivalent | Notes |
|------------|----------------|-------|
| FL = front-left | Left forelimb (arm) | Apes' arms are *longer* than legs (ratio ~1.3); robot has equal-length limbs |
| FR = front-right | Right forelimb (arm) | |
| BL = back-left | Left hindlimb (leg) | |
| BR = back-right | Right hindlimb (leg) | |

## Torso

| Robot | Ape |
|-------|-----|
| **One rigid body** (the dark cube), free-floating 6-DOF in world | **Articulated spine**: ~24 vertebrae grouped as cervical (7) + thoracic (12) + lumbar (5), with flexion/extension, lateral bending, and torsion at each |

Robot has **0 articulated torso DOF**. Real ape spine contributes:
- Lumbar lateral bending — needed for brachiation (gibbons)
- Thoracic rotation — needed for bipedal walking (chimps when standing up)
- Cervical (neck) — keeps head stable while body moves; robot has no head/neck

The rigid-torso simplification means the policy can't shape locomotion via spine motion. Workarounds: rely on limb pendulum + foot placement to stabilize the torso, accept less ape-like gait.

## What's missing vs. real ape

- **Shoulder abduction / hip abduction**: robot has yaw + pitch but no roll at the shoulder/hip. Apes lift arms sideways; robot can only swing them forward/back.
- **Spine articulation**: see above.
- **Multi-finger grip**: 1-DOF gripper vs. 5-finger grasping hand. Real apes can wrap fingers around branches; robot just opens/closes a parallel gripper.
- **Grasping feet**: same as fingers — apes have opposable big toes; robot has 1-DOF feet.
- **Head/neck**: robot has none. Real apes use head movement for balance and visual scanning.
- **Length asymmetry**: ape forelimbs are ~30% longer than hindlimbs (knuckle-walking, brachiation). Robot has equal-length limbs.

## Implication for this codebase

- The **CPG (Central Pattern Generator)** in `envs/walk_env_mjx.py` drives joints `1` (pitch) and `2` (elbow/knee) — the two main propulsive DOF per limb.
- A **trot pattern** (current default — diagonal pairs in phase) maps cleanly to a horse/dog gait. **Knuckle-walking** (chimp-style) would require asymmetric forelimb/hindlimb amplitudes and forward-tilted torso target, neither of which is in the env yet.
- Brachiation (gibbon-style) is **infeasible** with this morphology because there's no torso flex and the limbs are too short.

## Files where the mapping is encoded

- `data/scene.xml` — joint definitions, ranges, motor specs (lines 65-200)
- `data/motors.md` — per-joint motor type (ST3235 / ST3215 / ST3215-HS)
- `envs/walk_env_mjx.py` — CPG limb phasing (lines ~150 with `LIMB_PHASE`)
- `view.py` / `scripts/snapshot.py` — same CPG mirrored for visualization
