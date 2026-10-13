# Gibbon Bot

A gibbon-inspired robot in the laboratory at [Olivabot](https://olivabot.com). The arm does not walk. This repository is the public record of the sitting stack and of the simulation season that came before it.

Project page: [olivabot.com/en/projects/gibbon-bot](https://olivabot.com/en/projects/gibbon-bot.html). Company site source: [tonycurra/olivabot.com](https://github.com/tonycurra/olivabot.com).

![Gibbon Bot mk1](https://olivabot.com/assets/images/gibbon-bot/gibbot-bot-mk1/gibbot-bot-mk1-in-action-1.webp)

## The sitting stack

The chip on the arm and the laptop that drives it. Same text on USB and BLE. No motion at boot. The chip owns `estop`. Servos 12 and 13 are one shoulder and move only together, with `raise`.

```
browser ── HTTP/SSE ── gibbon-console :8080 ── USB or BLE ── ESP32 ── Feetech bus ── servos 11–17
```

| Folder | What it is |
| --- | --- |
| [`roarm-fw/`](roarm-fw/readme.md) | ESP-IDF firmware. Bus, estop, raise, hand and twist clamps |
| [`gibbon-console/`](gibbon-console/readme.md) | Django UI. The laptop owns USB and BLE and logs the trial |
| [`digital-twin/`](digital-twin/readme.md) | One RoArm in CPU MuJoCo, bolted to a table. Not yet matched to a sitting |

Research equipment. No warranty. Support the limb before stop. Torque off drops it. A 3S LiPo on the XT60 only. One process owns `/dev/ttyACM0`.

## The parked simulation

April–July 2026 tried to learn a walk on four RoArm arms and a torso. The walk was not achieved. There is no keeper policy in this tree. Read [`archive/EVALUATION.md`](archive/EVALUATION.md) before the training code.

| Folder | What it was |
| --- | --- |
| [`archive/gibbonBot-RL/`](archive/gibbonBot-RL/README.md) | MJX / PPO / CPG / LIANA dashboard |
| [`archive/gibbon-rl-mujoco/`](archive/gibbon-rl-mujoco/README.md) | Early four-arm URDF and xacro, 26 April 2026 |
| [`archive/olivabot-mujoco-cpu/`](archive/olivabot-mujoco-cpu/README.md) | CPU PPO stand-up on `main` |
| [`archive/gibbon-brain/`](archive/gibbon-brain/README.md) | July Cartesian experiments, no inverse kinematics |

## People

Tony — scene geometry, a torso and four arms, in [`archive/gibbonBot-RL/data/scene.xml`](archive/gibbonBot-RL/data/scene.xml) and [`archive/gibbon-rl-mujoco/simulation_code_jade/`](archive/gibbon-rl-mujoco/simulation_code_jade/). Simulation on `main`. Firmware, console, and the one-arm model.

Jade Leurs — simulation, April through July 2026. Pose priors from OpenApePose keypoints (21 May–17 June), a quadruped environment (26–28 June), a Unitree Go2 training setup (13 July), and telemetry on the CPU trainer (1 May). The commits are in [`CONTRIBUTORS.md`](CONTRIBUTORS.md). Those branches are not in this tree.

## Left out on purpose

Private business and subsidy notes. The vendor hardware PDF. Training weights. Addresses of rented machines. The company website, which already lives at [tonycurra/olivabot.com](https://github.com/tonycurra/olivabot.com) and is newer than the copy that used to sit beside this code.

## License

Olivabot code is [Apache-2.0](LICENSE). Sitting notes are CC BY 4.0. Waveshare meshes and URDF kinematics stay MIT. Each folder has a `NOTICE` where that split matters.

Hire and questions: [services@olivabot.com](mailto:services@olivabot.com).
