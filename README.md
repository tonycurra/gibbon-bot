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

Commit list: [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

**Tony** — 22 April to 28 September 2026. [`archive/gibbonBot-RL`](archive/gibbonBot-RL/) (22 April–15 May), the four-arm URDF in [`archive/gibbon-rl-mujoco`](archive/gibbon-rl-mujoco/) (26 April), [`archive/olivabot-mujoco-cpu`](archive/olivabot-mujoco-cpu/) `first commit` (27 April), then [`roarm-fw`](roarm-fw/), [`gibbon-console`](gibbon-console/), and [`digital-twin`](digital-twin/).

**Jade Leurs** — 1 May to 13 July 2026, eleven commits. `add telemetry logs`. OpenApePose commits in [`branches/apenapepose_integration`](archive/gibbonBot-RL/branches/apenapepose_integration/). `Initial commit`, `added custom (Ant-v5 based) environment`, and `unitree_go2 training setup` in [`branches/unigree-go2`](archive/gibbonBot-RL/branches/unigree-go2/).

**jajayumei** — 1 May 2026. `update requirements.txt to include tensorboard`.

## Left out on purpose

Private business and subsidy notes. The vendor hardware PDF. Training weights. Addresses of rented machines. The company website, which already lives at [tonycurra/olivabot.com](https://github.com/tonycurra/olivabot.com) and is newer than the copy that used to sit beside this code.

## License

Olivabot code is [Apache-2.0](LICENSE). Sitting notes are CC BY 4.0. Waveshare meshes and URDF kinematics stay MIT. Each folder has a `NOTICE` where that split matters.

Hire and questions: [services@olivabot.com](mailto:services@olivabot.com).
