# gibbon-console docs

Notes for the laptop screen in [Gibbon Bot](../../README.md). The program talks to the chip in [`roarm-fw`](../../roarm-fw/readme.md) over the same text on USB and Bluetooth. Django serves the pages and logs trials. Start here: [`../readme.md`](../readme.md).

Firefox (no Web Bluetooth). The **laptop** is the BLE central and USB owner. Phone: same WiFi, this server.

| Doc | |
| --- | --- |
| Run | `run.md` |
| Pages / protocol | `architecture.md` |
| USB vs BLE, reconnect, energy | `link.md` (3S only on XT60; `../roarm-fw/docs/hardware/lipo.md`) |
| Trials, Sample rows, MySQL | `data.md` |
| Joints, axes, act | `ui.md` |
| Calibrate, wrap Δ, degrees | `calibrate.md` |
| USB flash / signed firmware | `flash.md` |
| 2026-09-14 sitting (this UI) | `2026-09-14.md` |
| What to build next | `next.md` |

Sitting flow: `/calibration/` (walls) → `/j/<slug>/` (jog) → `/imitation/` (teach). Hand graphs: `/raw/`. BLE listen-only matrix (Chromium, USB unplugged): `/monitor/`. HAT gold serial: `/hat/`. Axes: `/info/`.

Chip: `docs` in `roarm-fw` (sensors, names, walls, flash). Do not invent HAT IMU / INA219 / LED.
