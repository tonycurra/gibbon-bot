# Archive — the simulation season, April–July 2026

Part of [Gibbon Bot](../README.md), Olivabot’s laboratory robot. The robot does not walk. This folder is the earlier attempt to learn a walk in simulation: four RoArm arms on a torso, aimed at an olive tree. The walk was not learned. No trained policy is here.

Tony built most of it. Jade Leurs, from May to July 2026, added logging to the CPU trainer, a pose study from ape video, a quadruped practice environment, and a Unitree Go2 scene. One of her commits is signed `jajayumei`. That is the same person. See [`CONTRIBUTORS.md`](../CONTRIBUTORS.md).

The arm as it stands in the lab is [`roarm-fw`](../roarm-fw/readme.md) and [`gibbon-console`](../gibbon-console/readme.md). The one-arm table model is [`digital-twin`](../digital-twin/readme.md).

Read in this order:

1. [EVALUATION.md](EVALUATION.md) — what was tried, and what the artefacts actually show
2. [NEXT_STEPS.md](NEXT_STEPS.md) — why the next work was firmware, limits, then a twin that matches metal
3. [WORK_OVERVIEW.md](WORK_OVERVIEW.md) — map of that season

| Folder | What a visitor finds |
| --- | --- |
| [gibbonBot-RL](gibbonBot-RL/README.md) | The main trainer. PPO, a gait clock, a dashboard. Jade’s pose study and Go2 scene live under `branches/` |
| [gibbon-rl-mujoco](gibbon-rl-mujoco/README.md) | The first four-arm model, 26 April 2026 |
| [olivabot-mujoco-cpu](olivabot-mujoco-cpu/README.md) | A smaller CPU trainer. Jade added telemetry and tensorboard |
| [gibbon-brain](gibbon-brain/README.md) | July experiments that move a hand in Cartesian space, without inverse kinematics |

These notes are CC BY 4.0. https://creativecommons.org/licenses/by/4.0/
