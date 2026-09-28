# archive — parked RL / sim (April–July 2026)

Not the live robot. Do not train from here. Do not copy this stack onto the HAT.

Workspace status: [`../STATUS.md`](../STATUS.md). Live path: `../roarm-fw` + `../gibbon-console`. One-arm twin (if you need meshes later): `../digital-twin`.

Each project below that has a `.git` is its own repo (most already have a Codeberg remote). `gibbon-brain/` is loose files plus an upstream MuJoCo clone. A push of these repos leaves out virtualenvs, training weights, `gibbon-bot-code-only.zip`, and `gibbon-brain/mujoco/`.

**Read first**

1. [`EVALUATION.md`](EVALUATION.md) — walk-to-tree was not achieved; no keeper weights
2. [`NEXT_STEPS.md`](NEXT_STEPS.md) — why the live path is firmware + limits, then a twin that matches metal
3. [`WORK_OVERVIEW.md`](WORK_OVERVIEW.md) — git atlas of the nested repos

| Folder | |
| --- | --- |
| `gibbonBot-RL/` | MJX/PPO/CPG/LIANA. 4× RoArm-M3 on a torso |
| `gibbon-rl-mujoco/` | early four-arm URDF/xacro |
| `olivabot-mujoco-cpu/` | CPU PPO stand-up (Jade) |
| `gibbon-brain/` | July Cartesian PD (no project git; includes a local MuJoCo clone) |
| `Roarm-m3pro-busServoDriverHAT_A-battery/` | hardware PDF |
| `gibbon-bot-code-only.zip` | old snapshot |

**Keep:** MJCF/meshes as a *starting* twin; `gibbonBot-RL/data/controls.md` / motor notes.

**Park:** LIANA growth, reward soup, treating the scripted walk as a policy, PPO on metal.
