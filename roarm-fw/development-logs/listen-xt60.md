# Listen + XT60

USB listen, then XT60, then a full power cycle. No auto-move. Chip MAC `44:1d:64:86:3f:54`.

## Firmware

`listen v1`: heartbeats, raw UART1 RX, no ping on boot, no WiFi/BLE, no `WritePos` / torque write. Commands: `state`, `gpio`, `info`, `listen`, `ping`, `bench`, `get`, `batt`, `raw ID`. Console on the main task.

## Power domains (this HAT)

| Supply | What it powers |
| --- | --- |
| USB (ESP32 Type-C) | ESP32 3.3 V. `/dev/ttyACM0` |
| XT60 | Servo VIN. Pack voltage is **inside each ST3215**, 0.1 V LSB. No INA219 |

No USB + no pack = dead. USB unplug with pack already off: ACM gone, silence. USB replug: `reset=poweron`, `t=` restarts, same ACM0.

Pack click **with USB up** (twice each way): ESP32 did not reset, ACM0 stayed, `bus_rx=0` until we ping. Not a proof hot-plug is always safe for the laptop. USB does **not** hold the motors.

Not done: pack on, then unplug USB (does ESP32 live on the 5 V buck?).

## Bus

ST servos do not broadcast. Battery off: `found=0/7`. Battery on: `found=7/7` IDs 11–17, twice. Unplug pack: `0/7` again.

## Telemetry (pack on, sitting, `tq=0`)

`batt pack=11.4V est=25%` (3S rest guess, not a BMS). USB does not charge XT60.

Positions stable across two `get`s (counts ±1). Elbow ~2822, not mid-scale. All `moving=0` `load=0` `cur=0`. No twitch on pack click.

HAT sensors: none. No IMU / INA219 / OLED / user LED.

## Gold

- Every USB start, pack off: `docs/baselines/usb-start.md`
- Pack on: `docs/baselines/usb-battery-on.md`
- Ping-only pack off: `docs/baselines/usb-no-battery.md`

## Host

`dialout` worked without `chmod`. IDF venv `/tmp/py311bin` dies after reboot — `scripts/idf-env.sh` recreates it.

## Next

Hand-move with `tq=0`: USB no pack (expect silence), then USB + pack (expect `pos` to change). `docs/plans/hand-move.md`. Then BLE. VIN-only ESP32 still open. No commanded nudge yet.
