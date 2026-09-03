This is a modified Waveshare RoArm-M3 Pro (Tony: XT60 battery for portable use) plus Bus Servo Driver HAT (A), ESP32-WROOM-32.

**Every flash / serial:** `docs/espidf.md`  
Stop monitor: second terminal `pkill -f esp_idf_monitor`

Firmware: native **ESP-IDF**. On the chip now: USB `info` + servo `ping` (IDs 11–17). USB first, BLE later. No auto-move on boot.

| | |
| --- | --- |
| Why IDF | `docs/framework.md` |
| Install / flash / serial | `docs/espidf.md` |
| Pins, IDs, telemetry | `docs/hardware.md` |
| HAT swap / wiki | `docs/hardware/hardware.md`, `docs/hardware/SBSD-HAT.md` |
| Bring-up tries (this day) | `development-logs/2026-09-03.md` |
| USB then BLE | `docs/usb-then-ble.md` |
| Day notes | `development-logs/` |
| Stock RoArm / motors | `docs/old docs to review/` |
