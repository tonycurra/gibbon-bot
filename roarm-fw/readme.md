This is a modified Waveshare RoArm-M3 Pro (Tony: XT60 battery for portable use) plus Bus Servo Driver HAT (A), ESP32-WROOM-32.

**Every flash / serial:** `docs/framework/espidf.md`  
**Every USB start (pack off):** `docs/baselines/usb-start.md`  
Stop monitor: second terminal `pkill -f esp_idf_monitor`

Firmware: native **ESP-IDF**. No move on boot. READ `get` / `ping` / `batt`. The only write is **hand** (ID 17), clamped. USB first, BLE later.

| | |
| --- | --- |
| Why IDF | `docs/framework/framework.md` |
| Install / flash / serial | `docs/framework/espidf.md` |
| Naming | `docs/framework/naming.md` |
| Pins, IDs, telemetry | `docs/hardware.md` |
| Motor raw data (`get`) | `docs/hardware/motors/sensors.md` |
| What to say (human names) | `docs/hardware/motors/names.md` |
| USB start (healthy, pack off) | `docs/baselines/usb-start.md` |
| USB + battery gold | `docs/baselines/usb-battery-on.md` |
| USB no-battery ping gold | `docs/baselines/usb-no-battery.md` |
| Hand-move (USB + pack, `tq=0`) | `docs/baselines/hand-move-usb-battery-on.md` |
| Host watch (hand-move) | `scripts/watch-hand.py` |
| Software (chip + host) | `docs/software.md` |
| Where we are (safety / limits / cal) | `docs/plans/now.md` |
| Hand open/close limits | `docs/baselines/hand-open-close.md` |
| Next: twist, then other walls | `docs/plans/joint-limits.md` |
| Then: BLE HTML dashboard | `docs/plans/ble-html.md` |
| HAT swap / wiki | `docs/hardware/hardware.md`, `docs/hardware/SBSD-HAT.md` |
| USB then BLE | `docs/plans/usb-then-ble.md` |
| Test notes | `development-logs/` |
| Stock RoArm / motors | `docs/old docs to review/` |
