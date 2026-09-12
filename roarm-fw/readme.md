This is a modified Waveshare RoArm-M3 Pro (Tony: XT60 battery for portable use) plus Bus Servo Driver HAT (A), ESP32-WROOM-32.

**Every flash / serial:** `docs/framework/espidf.md`  
**Every USB start (pack off):** `docs/baselines/usb-start.md`  
Stop monitor: second terminal `pkill -f esp_idf_monitor`

Firmware: native **ESP-IDF**. No move on boot. READ `get` / `ping` / `batt` / `stream` / `who`. Writes: any ID via `id N` (11–15 hop ≤80), **hand** (17) and **twist** (16) clamped. Same text on USB and **BLE** (`gibbon-XXXX`, Nordic UART). Sitting: `../gibbon-console` (Django) or `scripts/live.py`. Tests: `docs/software.md`.

| | |
| --- | --- |
| Why IDF | `docs/framework/framework.md` |
| Install / flash / serial | `docs/framework/espidf.md` |
| Naming | `docs/framework/naming.md` |
| Pins, IDs, telemetry | `docs/hardware.md` |
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
| USB + battery gold | `docs/baselines/usb-battery-on.md` |
| USB no-battery ping gold | `docs/baselines/usb-no-battery.md` |
| Hand-move (USB + pack, `tq=0`) | `docs/baselines/hand-move-usb-battery-on.md` |
| Live sensor table | `scripts/listen.py` |
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
