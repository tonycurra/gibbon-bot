# gibbon-console

Laptop UI for the RoArm ESP32 (`../roarm-fw`). Same text pipe on USB and BLE. Django serves the pages and logs trials.

Firefox (no Web Bluetooth). The **laptop** is the BLE central and USB owner. Phone: same WiFi, this server.

| Doc | |
| --- | --- |
| Run | `run.md` |
| Pages / protocol | `architecture.md` |
| USB vs BLE, reconnect, energy | `link.md` |
| Trials, Sample rows, MySQL | `data.md` |
| Joints, axes, act | `ui.md` |
| Calibrate, wrap Δ, degrees | `calibrate.md` |
| USB flash / signed firmware | `flash.md` |
| 2026-09-14 sitting (this UI) | `2026-09-14.md` |
| What to build next | `next.md` |

Sitting flow: `/calibration/` (walls) → `/j/<slug>/` (jog) → `/imitation/` (teach). Hand graphs: `/raw/`. HAT gold serial: `/hat/`. Axes: `/info/`.

Chip: `docs` in `roarm-fw` (sensors, names, walls, flash). Do not invent HAT IMU / INA219 / LED.
