# Motor data (raw)

Each joint is one Feetech ST32xx bus servo. The **only** joint sensor is the **magnetic encoder inside that motor**. The HAT has no extra encoder, IMU, or current chip.

Type `get` (USB, XT60 on). Same fields on IDs **11–17**. Servos stay silent until we READ them.

```
get id=11 pos=1131 speed=0 load=0 v_raw=115 v=11.5 temp=25 cur=0 moving=0 tq=0
```

## Sensors (live inside the servo)

| Field | What it is | Raw | Scale |
| --- | --- | --- | --- |
| `pos` | Magnetic encoder, output horn | 0–4095 | 12-bit, 360°. Mid ≈ 2048. ~0.088°/count |
| `speed` | Encoder rate, computed in the servo | signed | steps/s. 0 when sitting |
| `v_raw` / `v` | VIN at **this** servo (XT60 rail) | 0.1 V LSB | `v = v_raw / 10`. Pack voltage, not USB |
| `temp` | Internal temperature | °C | Overheat watch |
| `cur` | Motor current | signed mA | 0 at rest with torque off is normal |
| `load` | Control effort / strain, **not** a torque cell | signed, often ±1000 | **0** with `tq=0`. With `tq=1` this is the contact / squeeze signal |
| `cur` | Motor current | signed mA | Also rises on stall. Use with `load` |

`pos` is the one you watch when you turn a joint by hand. Hand-move gold (USB + pack): `docs/baselines/hand-move-usb-battery-on.md`. Host loop: `scripts/watch-hand.py`.

## Flags (not sensors)

| Field | What it is |
| --- | --- |
| `moving` | 0/1 — servo thinks it is executing a move |
| `tq` | Torque enable register. **Must stay 0** on this listen firmware. `1` = holding. Unplug XT60 |

## Same data, seven motors

| ID | Joint | Motor |
| --- | --- | --- |
| 11 | Base | ST3235 |
| 12 | Shoulder A | ST3235 |
| 13 | Shoulder B (dual-drive with 12) | ST3235 |
| 14 | Elbow | ST3235 |
| 15 | Wrist pitch | ST3235 |
| 16 | Wrist roll | ST3215 |
| 17 | Gripper | ST3215HS |

All seven share the same present-register block. Dual-drive shoulder: move one horn, **12 and 13** both change, often opposite sign.

## Not on this arm

No IMU. No joint torque/force cell. No gripper force. No HAT INA219 (pack V is the servo VIN reads). No motor serial number. USB-only (pack off): `get` is all `miss` — encoders are unpowered.

`batt` is the same VIN bytes as `v`, then a median + 3S rest-% guess. Not a BMS.
