This is a modified Waveshare RoArm-M3 Pro (Tony: XT60 battery for portable use) plus Bus Servo Driver HAT (A), ESP32-WROOM-32.

**Every flash / serial:** `docs/framework/espidf.md`  
**Every USB start (pack off):** `docs/baselines/usb-start.md`  
Stop monitor: second terminal `pkill -f esp_idf_monitor`

Firmware: native **ESP-IDF**. On the chip now: `listen v1` — USB heartbeats, `state` / `gpio` / `info`, READ-only `ping` / `get` / `batt` / `bench`. No auto-move. USB first, BLE later.

| | |
| --- | --- |
| Why IDF | `docs/framework/framework.md` |
| Install / flash / serial | `docs/framework/espidf.md` |
| Naming | `docs/framework/naming.md` |
| Pins, IDs, telemetry | `docs/hardware.md` |
| USB start (healthy, pack off) | `docs/baselines/usb-start.md` |
| USB + battery gold | `docs/baselines/usb-battery-on.md` |
| USB no-battery ping gold | `docs/baselines/usb-no-battery.md` |
| Next: hand-move, torque off | `docs/plans/hand-move.md` |
| Then: BLE HTML dashboard | `docs/plans/ble-html.md` |
| HAT swap / wiki | `docs/hardware/hardware.md`, `docs/hardware/SBSD-HAT.md` |
| USB then BLE | `docs/plans/usb-then-ble.md` |
| Day notes | `development-logs/` |
| Stock RoArm / motors | `docs/old docs to review/` |
