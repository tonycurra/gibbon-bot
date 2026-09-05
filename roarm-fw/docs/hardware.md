# Hardware (this firmware)

Source: HAT wiki (`hardware/SBSD-HAT.md`) + migration guide + stock RoArm `old docs to review/controls.md`.

## Board

Waveshare RoArm-M3 Pro, modified with XT60 battery for portable use, Bus Servo Driver HAT (A). MCU: **ESP32-WROOM-32**.

**UART control switch = ESP32.** USB or Pi isolates the data pins; bus goes silent.

Power: 9–25 V on the HAT (match the servos). ST3215s: 7–12.6 V typical (12 V 5 A or 3S LiPo). Battery / board switch ON for servos to answer.

USB debug: **115200**. This machine saw the board as `/dev/ttyACM0`. USB powers the **ESP32 only**. Servos need XT60.

## This HAT vs the old General Driver Board

Tony swapped the kit’s General Driver Board for the Bus Servo Driver HAT (A) so a 3S LiPo can plug in on XT60 (`hardware/hardware.md`).

| On the old board | On this HAT |
| --- | --- |
| User LED (PH2.0) | **None.** Do not blink GPIO 2 expecting a lamp |
| INA219 voltage/current | **None.** Pack voltage will come from servo `ReadVoltage`, not a chip on the HAT |
| 9-axis IMU | **None** |
| OLED I2C | **None** on the HAT silkscreen |

HAT “What’s On Board”: ESP32-WROOM-32, 5 V buck, UART switch (must be **ESP32**), Reset, BOOT, Pi header, two USB-C (ESP32 debug vs direct-to-bus), XT60 / DC / screw terminal, RS485 + TTL servo headers. Wiki dump: `hardware/SBSD-HAT.md`.

## Bus

| | |
| --- | --- |
| S_RXD | GPIO 18 |
| S_TXD | GPIO 19 |
| Baud | 1 000 000, half-duplex |

## Servo IDs

| ID | Joint | Say |
| --- | --- | --- |
| 11 | Base rotation | turn left / right |
| 12+13 | Shoulder (dual-drive) | raise / lower |
| 14 | Elbow | bend / straighten |
| 15 | Wrist 1 (pitch) | wrist up / down |
| 16 | Wrist 2 (roll) | twist CW / CCW |
| 17 | Gripper | open hand / close hand |

Words: `hardware/motors/names.md`.

Same chain is a gibbon **arm** or **leg**; only names change (`old docs to review/controls.md`).

## Telemetry (ST3215)

Raw fields from `get`: `hardware/motors/sensors.md`.

| Metric | API | Range / scale | Use |
| --- | --- | --- | --- |
| Position | `ReadPos` | 0–4095 (360°) | Command vs actual |
| Speed | `ReadSpeed` | steps/s | Motion |
| Load | `ReadLoad` | −1000–1000 | Stall / collision |
| Voltage | `ReadVoltage` | ÷10 = volts | Pack at that servo (XT60 rail). Type `batt` for median + 3S rest-% guess. **No INA219 / charger on this HAT** |
| Temp | `ReadTemper` | °C | Overheat |
| Current | `ReadCurrent` | mA | Draw |
| Moving | `ReadMove` | 0/1 | Busy |

No IMU on the servos.

## Safety (keep in firmware)

- Slow default speed.
- Voltage &lt; 10.5 V → torque off IDs 11–17.
- High \|load\| → stop that move.
- Stock INIT homes are not all-zero (elbow ~π/2, gripper display ~π). Do not slam to sim zero.
