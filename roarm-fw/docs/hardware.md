# Hardware (this firmware)

Source: HAT wiki (`hardware/SBSD-HAT.md`) + migration guide + stock RoArm `old docs to review/controls.md`.

## Board

Waveshare RoArm-M3 Pro, modified with XT60 battery for portable use, Bus Servo Driver HAT (A). MCU: **ESP32-WROOM-32**.

**UART control switch = ESP32.** USB or Pi isolates the data pins; bus goes silent.

Power: 9–25 V on the HAT (match the servos). ST3215s: 7–12.6 V typical (12 V 5 A or 3S LiPo). Battery / board switch ON for servos to answer.

USB debug: **115200**. This machine saw the board as `/dev/ttyACM0`.

## Bus

| | |
| --- | --- |
| S_RXD | GPIO 18 |
| S_TXD | GPIO 19 |
| Baud | 1 000 000, half-duplex |

## Servo IDs

| ID | Joint |
| --- | --- |
| 11 | Base rotation |
| 12 | Shoulder A (primary) |
| 13 | Shoulder B (dual-drive mirror) |
| 14 | Elbow |
| 15 | Wrist 1 (pitch) |
| 16 | Wrist 2 (roll) |
| 17 | Gripper |

Same chain is a gibbon **arm** or **leg**; only names change (`old docs to review/controls.md`).

## Telemetry (ST3215)

| Metric | API | Range / scale | Use |
| --- | --- | --- | --- |
| Position | `ReadPos` | 0–4095 (360°) | Command vs actual |
| Speed | `ReadSpeed` | steps/s | Motion |
| Load | `ReadLoad` | −1000–1000 | Stall / collision |
| Voltage | `ReadVoltage` | ÷10 = volts | LiPo cutoff |
| Temp | `ReadTemper` | °C | Overheat |
| Current | `ReadCurrent` | mA | Draw |
| Moving | `ReadMove` | 0/1 | Busy |

No IMU on the servos.

## Safety (keep in firmware)

- Slow default speed.
- Voltage &lt; 10.5 V → torque off IDs 11–17.
- High \|load\| → stop that move.
- Stock INIT homes are not all-zero (elbow ~π/2, gripper display ~π). Do not slam to sim zero.
