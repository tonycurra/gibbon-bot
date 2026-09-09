# UI

Cyberpunk: pure black, cyan `#00f0c8`, magenta `#ff2bd6`, amber for next-action. Mono. `static/console/app.css`.

Stand **behind the base**. Words + axes: `../roarm-fw/docs/hardware/motors/names.md`. Sitting docs: `/info/`.

## Joints (bus IDs stay 11–17)

| UI | Axis | ID | Say |
| --- | --- | --- | --- |
| shoulder yaw | yaw | 11 | turn |
| shoulder pitch | pitch | **12+13** | raise — **one joint, two motors** |
| elbow | pitch | 14 | bend |
| wrist pitch | pitch | 15 | nod |
| wrist roll | roll | 16 | twist |
| hand | grip | 17 | open / close |

Overview shows those six rows. Click → `/j/<slug>/`. Shoulder pitch actuate sends the **same** op to 12 and 13. If they fight: **STOP**.

Present columns: pos **56**, speed **58**, load **60**, V **62**, °C **63**, mA **69**, moving **66**, tq **40**. EEPROM (`who`) lives on **advanced**.

Footer: `14  pos 2811→2835  spd 150`. Idle heartbeats stay out. `moving`/`load` are **0** while you backdrive (`tq=0`) — watch **pos**.

## Act (on the joint page)

Firmware clamps; UI does not invent new writes. 11–15 hop ≤80. Twist hop ≤900. Hand 689–2075. Shoulder pitch: free/hold/nudge **both**. Wrist roll: no `twist free` during wheel checks. **STOP** / Escape = `estop`.

`move speed` / `move acc` = next WritePos (default 300 / 20).

No auto-move on connect. Support the arm when XT60 goes on.

## Calibrate

`/calibrate/?j=yaw` (etc). tq off, move by hand, mark **home / min / max** from live pos. Stored in `Limit` (host notebook). Chip still owns clamps for hand/twist. Do not command a joint with no wall.
