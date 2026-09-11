# Motor data (raw)

There is **no environment sensor**. No IMU, no force/torque cell, no hall on the loom, no HAT current chip. The only sensors are **inside each Feetech ST32xx**. If we do not READ them while `tq=1`, we are blind and the motor keeps pushing.

Type `get` or `get 16` (USB, XT60 on). Same present block on IDs **11–17**. Servos stay silent until we READ them.

Typed `get` / `get N` prints **every present field including zeros**, plus `hex=` of the 15 bytes @56. **`stream` stays compact**: omits zeros and skips IDs that did not move. Parsers treat missing keys as 0 / `un=pos` / servo. The joint page polls `get N` so the amber line is the full snapshot.

```
get id=16 pos=4068 v=11.3 temp=32
get id=14 pos=2811 speed=150
```

Old gold dumps still look like:

```
get id=16 pos=4068 speed=0 load=0 v_raw=113 v=11.3 temp=32 cur=0 moving=0 tq=0
```

## What is actually in the motor (ST3215 / ST3235)

Wiki “feedback”: **Position, Load, Speed, Input Voltage**. Firmware also reads **temperature** and **current** from the same present block (addr 56, 15 bytes).

| Field | Hardware | What it means | When it is useful |
| --- | --- | --- | --- |
| `pos` | 12-bit **magnetic encoder** on the output horn | 0–4095 = one turn of the magnet. **Wraps.** Not cable turns | Hand-move walls. Host `un` for twist |
| `speed` | Derived from that encoder | signed steps/s | Moving vs stall (pos not advancing) |
| `load` | **Controller effort**, not a torque transducer | signed, often ±1000. PWM / strain the inner PID is applying | **Bind / taut cable / contact** while `tq=1` |
| `cur` | Servo **winding current** estimate | signed mA. Wiki stall **2.5–2.7 A**. Our free moves were **0–9** | Stall / hard fight. Use with `load` |
| `v` / `v_raw` | VIN at **this** servo | 0.1 V LSB. Pack rail, not USB | Brownout / cutoff 10.5 V |
| `temp` | Case / driver temperature | °C | Slow fuse. Not a cable sensor |

`load` **is the torque proxy we have**. It is **always 0 with `tq=0`**. Turning by hand does not show a tight cable. A **rise while moving** (`Δload`) is the stop — not a fixed 1600.

Flags (not sensors): `moving` 0/1, `tq` 0/1.

## What is not on this arm

No IMU. No joint torque cell. No gripper force. No loom hall. No HAT INA219. No ambient / “environment” pack. USB-only (pack off): `get` is all `miss` — the magnet is unpowered.

`batt` is the same VIN bytes as `v`, then a 3S rest-% guess. Not a BMS.

## Listen rules (safety)

Chip must poll ID 16 **while it is pushing** (`twist speed N ≠ 0`), not only when the laptop types `get`. Cut on:

| Signal | Cut |
| --- | --- |
| `Δload` | `\|load\|` ≥ cruise + **100** (this sitting: cruise ~1137, hit started ~1220–1300) |
| `load` hard | `\|load\|` ≥ **1350** |
| `cur` | `\|cur\|` ≥ **80** mA (free was ~0–9) |
| bus miss | **3** failed present reads → `ESTOP` (speed 0, `tq=0`, servo). This is what failed 2026-09-05: host `get` miss, wheel kept spinning |

Commands: `estop` / `twist estop`. `get 16` while moving (full `get` of 7 IDs is slow and can starve the bus).

Host `live.py` is backup only. **Chip owns the cut.** Do not command twist until this firmware is flashed.

Write-up of the failed check: `docs/baselines/twist.md`. Control plan: `docs/plans/control.md`.
