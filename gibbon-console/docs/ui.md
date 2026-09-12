# UI

Cyberpunk: pure black, cyan `#00f0c8`, magenta `#ff2bd6`, amber for next-action. Mono. `static/console/app.css`.

Stand **behind the base**. Words + axes: `../roarm-fw/docs/hardware/motors/names.md`. Sitting docs: `/info/`. HAT gold serial (pack off / on): `/hat/`. Hand (open/close + graphs): `/raw/`.

## Joints (bus IDs stay 11–17)

| UI | Axis | ID | Say |
| --- | --- | --- | --- |
| shoulder yaw | yaw | 11 | turn |
| shoulder pitch | pitch | **12+13** | raise — **one joint, two motors** |
| elbow | pitch | 14 | bend |
| wrist pitch | pitch | 15 | nod |
| wrist roll | roll | 16 | twist |
| hand | grip | 17 | open / close |

Overview shows those six rows. Click → `/j/<slug>/`. Shoulder pitch: hold/free both IDs; motion is `raise to` **after flash**. If they fight: **STOP**.

Present columns: pos **56**, speed **58**, load **60**, V **62**, °C **63**, mA **69**, moving **66**, tq **40**. EEPROM (`who`) lives on **advanced**.

Footer: `14  pos 2811→2835  spd 150`. Idle heartbeats stay out. `moving`/`load` are **0** while you backdrive (`tq=0`) — watch **pos**.

## Act (on the joint page)

Firmware clamps; UI does not invent new writes. 11–15 hop ≤80. Twist hop ≤900. Hand 689–2075. Shoulder pitch: free/hold **both** as `id 12` / `id 13` until flash. Wrist roll: no `twist free` during wheel checks. **STOP** / Escape = `estop` (queued, jumps).

`move speed` / `move acc` = next WritePos (default 300 / 20).

No auto-move on connect. Support the arm when XT60 goes on.

Home overview slider: wrap joints ignore leftover raw bands (both-positive min/max on 11/16) and ignore negative Δ as a pos band. Twist with real Δ walls (`amin < 0 < amax`) clamps the hop to `[home+amin, home+amax]`.

Shoulder-pitch slider still emits `raise to`. **Do not use it until that firmware is flashed.** Hold/free are two packets: `id 12` / `id 13`.

## Calibrate

Index `/calibrate/`. One joint `/calibrate/<slug>/`. Wrap (11, 16): Δ and degrees. Procedure: `calibrate.md`.

`Limit` is the host notebook. Chip still owns clamps for hand/twist (gold walls). Do not command a joint with no wall. Do not send `raise …` until flash.
