# roarm-fw

ESP-IDF firmware for one Waveshare RoArm-M3 Pro on a Bus Servo Driver HAT (A). Part of [Gibbon Bot](https://olivabot.com/en/projects/gibbon-bot.html), Olivabot’s laboratory robot. The arm does not walk. This repository is the chip that sits on the metal.

The chip speaks the same text on USB (`/dev/ttyACM0`, 115200) and BLE (Nordic UART, name `gibbon-XXXX`). It does not move on boot. It owns `estop`. Servos 12 and 13 are one shoulder and move only together, with `raise`.

```
browser ── HTTP/SSE ── gibbon-console :8080 ── USB or BLE ── ESP32 ── Feetech bus ── servos 11–17
```

Laptop UI: [gibbon-console](https://github.com/tonycurra/gibbon-console). One-arm MuJoCo model: [gibbon-digital-twin](https://github.com/tonycurra/gibbon-digital-twin). Map of the whole laboratory: [gibbon-bot](https://github.com/tonycurra/gibbon-bot).

**Status, September 2026.** Safety bring-up is done: no motion at boot, `estop`, and `raise` as a SyncWrite of 12 and 13. Hand walls are gold (open 609, close 2088). The image on the metal is `fw=20260912`. The tree here is stamped `20260915` (`mat`, `sweep`, `lim`) and was not the image on the chip at that date. Wrist and elbow limits are still host notes. This is research equipment, one limb at a time, body supported.

## Safety

No move on boot. The chip owns ESTOP. Never command servo 12 or 13 alone — `raise` writes both. A 3S LiPo on the XT60 only. Research equipment. No warranty. This is not a farm machine.

Support the limb before `estop`. Torque off drops it. One process owns `/dev/ttyACM0`. Stop the Django console before `idf.py flash`.

## Fifteen minutes

ESP-IDF is required. The short loop, after the toolchain is on `PATH`:

```bash
. scripts/idf-env.sh
idf.py -p /dev/ttyACM0 flash
./scripts/mon.sh
```

Full flash, serial, and the Italian-keyboard monitor note: [`docs/framework/espidf.md`](docs/framework/espidf.md). Pack-off USB start that should already look healthy: [`docs/baselines/usb-start.md`](docs/baselines/usb-start.md). Command list: [`docs/software.md`](docs/software.md). Where the sitting stands: [`docs/plans/now.md`](docs/plans/now.md).

## What the chip owns

| On the ESP32 | On the laptop |
| --- | --- |
| Feetech bus, `estop`, `raise`, hand and twist clamps, compact stream, VIN | Degrees, unwrap, 3S percent, trial log, sliders |

Servo inner PID stays in the Feetech. This firmware does not retune it.

| UI name | IDs | Rule |
| --- | --- | --- |
| yaw | 11 | wrap; host tracks delta from this slack |
| shoulder pitch | 12+13 | `raise` only |
| elbow | 14 | host band, hop ≤ 80 until the chip has walls |
| wrist pitch | 15 | stay short of 3134 |
| wrist roll | 16 | clamped on the chip |
| hand | 17 | `hand open` / `hand close`, walls on the chip |

## Layout

| Path | What |
| --- | --- |
| `main/` | Sitting image. `main.c` is the protocol and the bus. `ble_nus.c` is Nordic UART |
| `bringup/` | Two tiny images: USB listen, and BLE `g1`. They are not the sitting firmware |
| `scripts/` | `mon.sh`, `live.py`, `listen.py` |
| `docs/` | Hardware, protocol, baselines, plans |
| `development-logs/` | Dated sitting notes |

Hardware is a Waveshare RoArm-M3 Pro, Bus Servo Driver HAT (A), ESP32-WROOM-32, XT60. There is no HAT IMU, INA219, OLED, or user LED on this board.

## License

Code is [Apache-2.0](LICENSE). Sitting notes and baseline tables are CC BY 4.0. See [NOTICE](NOTICE).

Bugs in `estop` or a joint wall can be reported even when everything else waits. [SECURITY](SECURITY.md). Hire and the project page: [olivabot.com](https://olivabot.com).

---

This is a modified Waveshare RoArm-M3 Pro (XT60 battery for portable use) plus Bus Servo Driver HAT (A), ESP32-WROOM-32.

**Every flash / serial:** `docs/framework/espidf.md` (`./scripts/mon.sh` after flash)  
**Every USB start (pack off):** `docs/baselines/usb-start.md`  
**Tiny bring-up images (USB listen, BLE `g1`):** `bringup/`  
Stop IDF monitor: second terminal `pkill -f esp_idf_monitor`

Firmware: native **ESP-IDF**. No move on boot. READ `get` / `ping` / `batt` / `stream` / `mat` / `who`. Writes: any ID via `id N` (11–15 hop ≤80), **hand** (17) and **twist** (16) clamped. Same text on USB and **BLE** (`gibbon-XXXX`, Nordic UART). Sitting: `../gibbon-console` (Django), `./scripts/mon.sh` (serial), or `scripts/live.py`. Tests: `docs/software.md`.

| | |
| --- | --- |
| Why IDF | `docs/framework/framework.md` |
| Install / flash / serial | `docs/framework/espidf.md` |
| Naming | `docs/framework/naming.md` |
| Pins, IDs, telemetry | `docs/hardware.md` |
| Packs (3S only, storage, BEMF) | `docs/hardware/lipo.md` |
| Motor raw data (`get`) | `docs/hardware/motors/sensors.md` |
| ESP32 ↔ ST bus (signal, packets) | `docs/hardware/motors/esp32-bus.md` |
| What to say (human names) | `docs/hardware/motors/names.md` |
| Joint limits (`pos` walls) | `docs/hardware/motors/joint-limits.md` |
| BLE NimBLE / energy / ghost | `docs/ble.md` |
| Laptop UI (Django) | `../gibbon-console/docs/README.md` |
| 2026-09-09 console + compact get | `development-logs/2026-09-09-console.md` |
| 2026-09-11 calibrate / wrap Δ | `development-logs/2026-09-11-calibrate.md` |
| Minimal firmware (chip vs host) | `docs/plans/firmware-minimal.md` |
| USB start (healthy, pack off) | `docs/baselines/usb-start.md` |
| Bring-up series (tiny `.c` + BLE `/monitor/`) | `bringup/` |
| USB + battery gold | `docs/baselines/usb-battery-on.md` |
| USB no-battery ping gold | `docs/baselines/usb-no-battery.md` |
| Hand-move (USB + pack, `tq=0`) | `docs/baselines/hand-move-usb-battery-on.md` |
| Live sensor matrix (`mat`) | `scripts/listen.py` |
| Fast serial (no IDF) | `scripts/mon.sh` |
| Host watch (hand-move) | `scripts/watch-hand.py` |
| Listen + control (one ACM0) | `scripts/live.py` |
| Software + test catalog | `docs/software.md` |
| Twist walls / check | `docs/baselines/twist.md` |
| Twist sitting 2026-09-05 (trials + errors) | `docs/baselines/twist-sitting-2026-09-05.md` |
| Where we are (safety / limits / cal) | `docs/plans/now.md` |
| Four limbs / rates / ROS2 vs ours | `docs/plans/gibbon-platform.md` |
| Why firmware+console, not official ROS 2 | `docs/plans/ros2-vs-our-stack.md` |
| Motor / arm control (PID, smooth, load-stop) | `docs/plans/control.md` |
| Inside the servo + BLE/energy next | `docs/plans/near-future.md` |
| Hand open/close limits | `docs/baselines/hand-open-close.md` |
| Next: twist, then other walls | `docs/plans/joint-limits.md` |
| Collision vs FK / IK (after walls) | `docs/plans/collision.md` |
| BLE HTML plan (superseded) | `docs/plans/ble-html.md` |
| HAT swap / wiki | `docs/hardware/hardware.md`, `docs/hardware/SBSD-HAT.md` |
| USB then BLE | `docs/plans/usb-then-ble.md` |
| Test notes | `development-logs/` |
| Stock RoArm / motors | `docs/old docs to review/` |
