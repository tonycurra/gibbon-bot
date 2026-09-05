This is a modified Waveshare RoArm-M3 Pro (Tony: XT60 battery for portable use) plus Bus Servo Driver HAT (A), ESP32-WROOM-32.

**Every flash / serial:** `docs/framework/espidf.md`  
**Every USB start (pack off):** `docs/baselines/usb-start.md`  
Stop monitor: second terminal `pkill -f esp_idf_monitor`

Firmware: native **ESP-IDF**. No move on boot. READ `get` / `ping` / `batt`. Writes: **hand** (17) and **twist** (16), clamped. Sitting: `scripts/live.py`. Tests: `docs/software.md`. USB first, BLE later. Host UI (not this folder): `../gibbon-panel`.

| | |
| --- | --- |
| Why IDF | `docs/framework/framework.md` |
| Install / flash / serial | `docs/framework/espidf.md` |
| Naming | `docs/framework/naming.md` |
| Pins, IDs, telemetry | `docs/hardware.md` |
| Motor raw data (`get`) | `docs/hardware/motors/sensors.md` |
| What to say (human names) | `docs/hardware/motors/names.md` |
| Joint limits (`pos` walls) | `docs/hardware/motors/joint-limits.md` |
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
| Motor / arm control (PID, smooth, load-stop) | `docs/plans/control.md` |
| Hand open/close limits | `docs/baselines/hand-open-close.md` |
| Next: twist, then other walls | `docs/plans/joint-limits.md` |
| Collision vs FK / IK (after walls) | `docs/plans/collision.md` |
| Then: BLE HTML dashboard | `docs/plans/ble-html.md` |
| HAT swap / wiki | `docs/hardware/hardware.md`, `docs/hardware/SBSD-HAT.md` |
| USB then BLE | `docs/plans/usb-then-ble.md` |
| Test notes | `development-logs/` |
| Stock RoArm / motors | `docs/old docs to review/` |
