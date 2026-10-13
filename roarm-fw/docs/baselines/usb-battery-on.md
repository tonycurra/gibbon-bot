# USB + battery — healthy bus (XT60 on)

Gold for **after the pack is on**, still no move. USB stays connected. UART switch **ESP32**. Match `usb-start.md` first (pack off), then this page.

If `found` is not `7/7`, stop. Do not move.

Captured 2026-09-04, firmware `listen v1`. Same chip as `usb-start.md` / `usb-no-battery.md`.

## Setup

| | Must be |
| --- | --- |
| USB-C | ESP32 debug port |
| Host node | still only `/dev/ttyACM0` — pack must **not** add a COM port |
| UART switch | **ESP32** (do not flip mid-test) |
| XT60 | **on**, USB still plugged |
| Firmware | listen. No `WritePos`, no torque write |

## Listen window (before ping)

ESP32 did **not** reset. Heartbeats kept going (`t=` ~864000 → ~900000). Heap still `297940`. `bus_rx=0`, no `bus t=… hex=` lines. GPIO 18/19 still idle-high. `tasks=5` unchanged.

Servos waking is not a broadcast. Silence here is normal.

```
--- xt60 on 2026-09-04T13:58:48+02:00 ---
state t=900570 heap=297940 reset=poweron tasks=5 bus_rx=0 dump=1
state mac=44:1d:64:86:3f:54
gpio 18(rx)=1 19(tx)=1 (uart1, not remuxed)
```

## Must match

| Field | Gold |
| --- | --- |
| `mac=` | `44:1d:64:86:3f:54` |
| `chip=` | `ESP32 cores=2 rev=301 flash=4096KB` |
| `bus=` | `gpio18/19@1Mbps ids=11-17` |
| `ping` | all `found` |
| `found=` | `7/7` |
| `guess=` | `usb-battery-on` |
| extra ACM | none |

## Gold `bench` (2026-09-04)

```
bench v1
chip=ESP32 cores=2 rev=301 flash=4096KB
mac=44:1d:64:86:3f:54
bus=gpio18/19@1Mbps ids=11-17
ping 11=found 12=found 13=found 14=found 15=found 16=found 17=found
found=7/7
guess=usb-battery-on
```

## Gold `get` (READ only, this pose)

This pose will change if the arm is moved by hand. **Voltage / temp / tq=0 / moving=0** are the health checks. Positions are “where it was sitting,” not a home.

```
get READ-only present+torque-en. no WritePos, no torque write
get id=11 pos=1900 speed=0 load=0 v_raw=115 v=11.5 temp=26 cur=0 moving=0 tq=0
get id=12 pos=2025 speed=0 load=0 v_raw=114 v=11.4 temp=28 cur=0 moving=0 tq=0
get id=13 pos=2073 speed=0 load=0 v_raw=114 v=11.4 temp=29 cur=0 moving=0 tq=0
get id=14 pos=2821 speed=0 load=0 v_raw=114 v=11.4 temp=28 cur=0 moving=0 tq=0
get id=15 pos=2176 speed=0 load=0 v_raw=114 v=11.4 temp=29 cur=0 moving=0 tq=0
get id=16 pos=4024 speed=0 load=0 v_raw=114 v=11.4 temp=26 cur=0 moving=0 tq=0
get id=17 pos=2088 speed=0 load=0 v_raw=114 v=11.4 temp=28 cur=0 moving=0 tq=0
```

| ID | Joint | This capture | Notes |
| --- | --- | --- | --- |
| 11 | Base | pos 1900 | ~near mid (2048 = 180°) |
| 12 | Shoulder A | 2025 | dual-drive with 13 |
| 13 | Shoulder B | 2073 | 48 counts from 12; not a 1:1 copy in raw |
| 14 | Elbow | 2821 | not 2048; stock home is ~π/2, not 0 |
| 15 | Wrist pitch | 2176 | |
| 16 | Wrist roll | 4024 | near 0/4095 wrap |
| 17 | Gripper | 2088 | |

| Field | Normal now | Stop if |
| --- | --- | --- |
| `v` | ~11.4–11.5 V (3S) | &lt; 10.5 V, or one ID far from the others |
| `temp` | 26–29 °C | climbing fast / &gt; 50 |
| `speed` `load` `cur` `moving` | 0 | non-zero while we are not commanding |
| `tq` | **0** (torque off) | `1` if you did not enable it — EEPROM woke holding |
| `pos` | 0–4095, stable | jump while sitting still |

## What came alive

- **HAT:** same ESP32 process. No new USB device. No INA219 / IMU / OLED / user LED / WiFi.
- **Bus:** IDs 11–17 answer ping. Voltage, temp, current, load, position are **inside each ST3215**.

## How to re-run

After `usb-start.md` matches, plug XT60, wait two heartbeats (`bus_rx` still 0), then `state`, `ping`, `bench`, `get`, `batt`. Still no `move` / `torque 1`.

## Pack voltage (`batt`, 2026-09-04)

HAT has no INA219. Servo VIN **is** the XT60 rail (0.1 V LSB). USB does not charge the pack. `%` is a 3S rest-voltage guess, not a fuel gauge.

```
batt id=11 v=11.5
batt id=12 v=11.4
batt id=13 v=11.4
batt id=14 v=11.4
batt id=15 v=11.4
batt id=16 v=11.4
batt id=17 v=11.3
batt pack=11.4V est=25% cells=3 n=7 ok
```

Cutoff in firmware plan: &lt; 10.5 V. This pack is `ok`.
