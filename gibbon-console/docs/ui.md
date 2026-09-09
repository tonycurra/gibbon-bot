# UI

Cyberpunk: pure black, cyan `#00f0c8`, magenta `#ff2bd6`, amber for next-action. Mono. Minimal CSS in `static/console/app.css`.

Stand **behind the base**. Words: `../roarm-fw/docs/hardware/motors/names.md`.

## Joints (bus IDs stay 11–17)

| UI | ID | Say |
| --- | --- | --- |
| base | 11 | turn |
| shoulder a / b | 12 / 13 | raise — **one joint, two motors** |
| elbow | 14 | bend |
| wrist | 15 | nod |
| twist | 16 | roll |
| hand | 17 | gripper |

Table columns are Feetech registers, not USB ports: pos **56**, speed **58**, load **60**, V **62**, °C **63**, moving **66**, mA **69**, tq **40**, mode **33**, st **65**, fw **0**.

Log: idle heartbeats/`get` stay out. Notes (`batt note=…`) stay. `get` lines print when pos/load/speed/current/moving/tq actually change.

## Act (`/act/`)

One joint, one op. Firmware clamps; UI does not invent new writes.

| IDs | Ops |
| --- | --- |
| 11–15 | free, hold, nudge Δ (hop ≤80), to pos |
| 12/13 | same — warn dual-drive (they fight if you move one) |
| 16 | free, hold, to (hop ≤900), mid, wheel speed, servo, wheel, here, twist estop |
| 17 | open 689, close 2008, to 689–2075, nudge, free, hold |

`move speed` / `move acc` = next WritePos (default 300 / 20). Header **STOP** = `estop` (all tq=0, twist servo).

No auto-move on connect. Support the arm when XT60 goes on.
