# gibbon-console

Laptop UI for the RoArm ESP32 (`../roarm-fw`). Same text pipe on USB and BLE. Django serves the pages and logs trials.

Firefox (no Web Bluetooth). The **laptop** is the BLE central and USB owner. Phone: same WiFi, this server.

| Doc | |
| --- | --- |
| Run | `run.md` |
| Layout / protocol / pages | `architecture.md` |
| USB vs BLE, reconnect, energy | `link.md` |
| Trials, Sample rows, MySQL | `data.md` |
| Joints, axes, act | `ui.md` |
| Calibrate, wrap Δ, degrees | `calibrate.md` |
| What to build next | `next.md` |

HAT gold serial (pack off / on, `get`, `batt`, hand-move) is `/hat/`. Hand open/close + sensor graphs: `/raw/` (nav **hand**).

Chip: `docs` in `roarm-fw` (sensors, names, walls, flash). Do not invent HAT IMU / INA219 / LED.
