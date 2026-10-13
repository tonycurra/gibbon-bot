# Gibbon Bot

Gibbon Bot is a laboratory robot from [Olivabot](https://olivabot.com), in Amsterdam. It is built around the idea of a gibbon: a machine for trees. The public page is [olivabot.com/en/projects/gibbon-bot](https://olivabot.com/en/projects/gibbon-bot.html). The company site is [tonycurra/olivabot.com](https://github.com/tonycurra/olivabot.com).

The robot does not walk. This repository is the public code: the arm as it stands in the lab, and the simulation season that came before it.

![Gibbon Bot mk1](https://olivabot.com/assets/images/gibbon-bot/gibbot-bot-mk1/gibbot-bot-mk1-in-action-1.webp)

## Overview

**The arm in the lab.** One Waveshare RoArm-M3 on a bench. The chip in [`roarm-fw/`](roarm-fw/readme.md) speaks a short text protocol to the motors, on a USB cable and over Bluetooth. It does not move when it powers on. Stop lives on the chip. The two shoulder motors move only as a pair. The laptop program in [`gibbon-console/`](gibbon-console/readme.md) is the screen: lock, stop, and a log of what was commanded and what the motors reported. [`digital-twin/`](digital-twin/readme.md) is a MuJoCo model of that single arm, bolted to a table. The model is not yet lined up with a recording from the metal.

```
browser ── HTTP/SSE ── gibbon-console :8080 ── USB or BLE ── ESP32 ── Feetech bus ── servos 11–17
```

Research equipment. Support the limb before stop. Torque off drops it. A 3S LiPo on the XT60 only. One process owns `/dev/ttyACM0`.

**The simulation, April–July 2026.** Four of the same arms on a torso, in MuJoCo, with an attempt to learn a walk toward an olive tree. The walk was not learned. No trained policy is in this repository. [`archive/`](archive/README.md) keeps the code and the written result. Inside it, Jade Leurs’s later experiments are a pose study taken from ape video, a quadruped practice environment, and a Unitree Go2 training setup.

**The people.** Tony built the arm software and most of the simulation, from April through September 2026. Jade Leurs worked on the simulation from May through July 2026. One of her commits is signed `jajayumei`. That is the same person. The commit list is [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

Olivabot publishes this so the work can be read. Firmware, device software, and simulation for hire: [services@olivabot.com](mailto:services@olivabot.com).

## Where to open

| Folder | What a visitor finds |
| --- | --- |
| [`roarm-fw/`](roarm-fw/readme.md) | Firmware on the chip. Bus, stop, shoulder, hand |
| [`gibbon-console/`](gibbon-console/readme.md) | Laptop screen. USB, Bluetooth, trial log |
| [`digital-twin/`](digital-twin/readme.md) | One arm in MuJoCo, on a table |
| [`archive/`](archive/README.md) | The 2026 walk attempt, and Jade Leurs’s simulation work |

## Left out on purpose

Private business and subsidy notes. The vendor hardware PDF. Training weights. Addresses of rented machines. The company website, which lives at [tonycurra/olivabot.com](https://github.com/tonycurra/olivabot.com).

## License

Olivabot code is [Apache-2.0](LICENSE). Sitting notes are CC BY 4.0. Waveshare meshes and URDF kinematics stay MIT. Each folder has a `NOTICE` where that split matters.
