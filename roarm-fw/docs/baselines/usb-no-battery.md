# Baseline: USB, battery unplugged

This is the known-good **USB-only** state for this HAT. Plug USB, leave XT60 off, UART switch **ESP32**, type `bench`. Compare to the gold block below.

Do not look for servo pos/load/V/temp/current, INA219, IMU, or an LED. Those are not on this board / not powered.

## How to run

Monitor already open: type `bench`, Enter.

Or after flash:

```bash
cd ~/code/gibbon-bot/roarm-fw
. scripts/idf-env.sh
sudo chmod 666 /dev/ttyACM0
idf.py -p /dev/ttyACM0 flash monitor
```

Then `bench`.

## Must match (this chip)

| Field | Gold |
| --- | --- |
| `chip=` | `ESP32 cores=2 rev=301 flash=4096KB` |
| `mac=` | `44:1d:64:86:3f:54` |
| `bus=` | `gpio18/19@1Mbps ids=11-17` |
| `ping` | all `miss` |
| `found=` | `0/7` |
| `guess=` | `usb-no-battery` |

Heap, uptime, reset reason **may change**. Ignore them.

`found=0` is also what you get if the UART switch is not ESP32, or motors are unplugged. Battery off is the usual reason.

## Gold `bench` (2026-09-03, XT60 unplugged)

```
bench v1
chip=ESP32 cores=2 rev=301 flash=4096KB
mac=44:1d:64:86:3f:54
bus=gpio18/19@1Mbps ids=11-17
ping 11=miss 12=miss 13=miss 14=miss 15=miss 16=miss 17=miss
found=0/7
guess=usb-no-battery
```

## Later: battery on

Same command. Expect `found=7/7` and `guess=usb-battery-on`. That baseline goes in `usb-battery-on.md` when we have it. Then `get` for joint telemetry. No auto-move.

## Why this, not another USB-only test

USB powers the ESP32 only. The ST3215s need XT60. Position, load, voltage, current, and temperature live **inside the servos**, so those reads stay empty until the pack is on.

Ping on GPIO 18/19 is the useful test in this state: the bus UART is up, and nobody answers.

## First capture (`info` + `ping`, same afternoon)

```
info chip=ESP32 cores=2 rev=301 flash=4096KB reset=poweron
info mac=44:1d:64:86:3f:54 heap=298976 uptime_ms=10550
info idf=v5.4.2 usb=115200 bus=gpio18/19@1Mbps ids=11-17
info note=servos need XT60; ping without battery is silence
ping miss id=11 … 17
ping none 11-17 — expected if battery is off
```

Day narrative: `development-logs/2026-09-03.md`. Every-start listen: `usb-start.md`. Flash loop: `docs/framework/espidf.md`.
