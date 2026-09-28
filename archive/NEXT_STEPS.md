# Next steps — limits, control, then twin

April–July treated the robot as a **simulation problem**: PPO on MJX, GPUs, reward soups, a dashboard, a brain that logs. The walk-to-tree goal was not reached. Weights are not in this tree. The physics was not identified on hardware.

Next: **a robot you can stop, limit, measure, and move on purpose**, then grow from **real signals**. Simulation stays as a **twin that must match the arm**, not as the place where learning is declared done.

## Frame

| Then | Now |
|---|---|
| Theoretical MuJoCo / MJX | Forces, delays, limits from the real RoArm-M3 |
| Success = GPU steps / dashboard | Safe motion + logged physics + repeatable tasks on metal |
| Brain modules, 12-term rewards | One control stack with hard limits |
| Scripted demo as walking | Do not ship kinematics as a policy |
| Weights on ephemeral GPU | Artifacts with a restore drill, or they do not exist |
| Many robots (Ant, Go2, ape) | One morphology for the product; others only as tools |

**Two tracks in parallel:**

1. **Real world** — limits, stop, sensing, datasets from the arms you have.
2. **Sim** — only while the twin is checked against (1). RL accelerates what you already understand.

## Principles

1. **Limits before intelligence.** Every command clamped to manufacturer ranges, voltage, torque, and the **cable-safe envelope** if that is tighter. No Cartesian PD that ignores IK.
2. **A stop that works.** Electrical and software e-stop. Slow default speeds.
3. **If it is not logged, it did not happen.** Command vs actual, timestamp. The July CSV had the right shape; it lacked validity checks.
4. **Fingerprint the plant.** Scene + firmware + measured delays. Apply to hardware sessions, not only GPU runs.
5. **Do not train on a lie.** Collision off, limb exclusions, datasheet damping without ID — untrusted until measured.
6. **Lean.** Do not revive 18 brain modules, grove trees, or 16384 envs until one arm tracks a setpoint.

Homes and ranges: `gibbonBot-RL/data/controls.md`, `motors.md`. Hardware PDF: `Roarm-m3pro-busServoDriverHAT_A-battery`.

## Phase 1 — Security and control

**Goal:** one RoArm-M3 (then four) that you can power, home, jog inside limits, and stop, with logs.

1. Power + e-stop (paper + a key software cannot override).
2. Driver: firmware units, read actuals, **reject** out-of-range (datasheet **and** cable hotspots / workspace).
3. FK; no target outside reach.
4. IK or Jacobian-limited Cartesian — never XYZ → joint index.
5. Hold pose, then slow line, then circle in a **verified** workspace.
6. Same tests in MuJoCo **after** measured limits and homes are in the XML. If sim ≠ robot, the sim is wrong.

**Done when:** a 10 cm line on the arm, CSV command ≈ actual within a stated error, an intentional limit stop.

This replaces `gibbon-brain/agent_control.py`. Port lessons, not the bug.

## Phase 2 — Empirical data

Legal-range trajectories, contact/current when the gripper hits a known object, multi-limb only after one limb is reliable. Persist raw sessions with firmware + limits hash.

OpenApePose GMM is a **pose prior**, not a controller. Either stay with equal RoArm limbs and drop near-term ape imitation, or change the mechanics. Do not run AMP/DeepMimic on the current URDF.

**Done when:** a folder of sessions a trainer could consume.

## Phase 3 — Perception (thin)

Only after 1–2: proprioception first; then one calibrated exteroceptive channel. Reasoning = state machine + limits + logged faults, not a brain module stack. Reuse only **command saturation** (withdrawal / joint limit) as code.

## Phase 4 — Twin, then RL

**When:** Phase 1 sim-vs-real error on a standard jog is small enough to trust.

**Reuse:** MJCF/meshes as a starting twin (overwrite gains from ID); MJX/PPO infra once the env *is* that twin; CPG as rhythm **if** the real robot can already crawl; fingerprints + keepers with a restore test; one-env viewer for bring-up.

**Park:** LIANA cards and Demo walk; quantum/brain-as-policy; grove/html_viewer as training deps; Autopilot on H100 until Phase 1 is green; Ant-v5 as product; Go2 only as an explicit second platform.

**RL:** named sparse skills (hold height, CPG crawl) on the twin, same safety wrapper on the arm. State the residual (delay, backlash) in numbers.

## Sequence

```
Recover Scaleway keepers if the bucket exists; freeze historical gibbonBot-RL at 9e2474a
Single-arm: limits, FK/IK, e-stop, log, 10 cm line
Four arms after mounts match metal
Datasets; optional camera
When jog matches: twin ID + optional CPG on hardware
Only then: MJX RL on the twin, keepers mirrored, restore drill
```

If something slips, slip **RL and LIANA**, never **limits and stop**.

## Other

- Reconcile sim gripper/elbow homes with firmware before more PD or RL.
- Treat lost `.npz` as lost. Do not plan around keepers you cannot open.
- Give `gibbon-brain` a real git repo (exclude `.venv` and upstream `mujoco/`) or fold into `gibbonBot-RL/control/`.
- Jade pose priors: research branch later; do not merge into the safety stack.
