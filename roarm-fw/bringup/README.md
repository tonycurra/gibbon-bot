# Bring-up firmware (one test per file)

Tiny ESP32 images you can read in one sitting. Each test is a `.c` plus a `.md`.

USB + pack together can brown-out / twitch (5 V vs XT60). **02** is BLE monitor: flash USB, unplug, pack on, console `/monitor/`.

One process owns `/dev/ttyACM0`. Stop `idf.py monitor` and the Django console before `flash.sh`.

## Loop

```bash
cd ~/code/gibbon-bot/roarm-fw/bringup
./flash.sh 02-ble-monitor
# unplug USB, pack on
# gibbon-console /monitor/  → connect (BLE)
```

## Tests

| # | File | Proves |
| --- | --- | --- |
| **01** | `01-usb-listen.c` | USB enumerate, banner, 2 s heartbeat. No bus, no BLE, no move |
| **02** | `02-ble-monitor.c` | BLE packed `g1` matrix 50 ms, tq=0, no move. Console `/monitor/` |
| 03 | — | Feetech ping 11–17, pack **off** → all miss |
| 04 | — | Same ping, XT60 **on** → 7/7 ~11 V |

## Rules

- No move on boot. 02 writes `tq=0` once when servos first answer.
- No WiFi. BLE only after 01 USB is boring — 02 is that radio.
- No extra FreeRTOS task that `printf`s.
- This HAT has no user LED, no INA219, no IMU.
- Keep each `.c` reviewable. Do not grow 02 into the sitting firmware.

Board: Waveshare Bus Servo Driver HAT (A), ESP32-WROOM-32. Identity: MAC `44:1d:64:86:3f:54`, BLE `gibbon-3f54`.
