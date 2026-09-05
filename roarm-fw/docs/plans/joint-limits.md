# Joint limits — one move at a time

Encoders work (`docs/baselines/hand-move-usb-battery-on.md`). **Soft stops** so later commands cannot wrap a cable or hit plastic.

Hand (17) is done: `docs/baselines/hand-open-close.md`. Chip may `hand open` / `close` / `to` / `free` — **ID 17 only**. Other joints: you still move, I record. No writer until that wall exists.

Words: `docs/hardware/motors/names.md`.

## Why not 0–4095

The encoder is a full circle. The **arm is not**. Stock UI already clips most joints to ~180° (shoulder, elbow, wrist nod) and the gripper to ~135°. Cables on this HAT swap are the thing that shorts. Stop at the first taut loom or self-hit, then back off.

Software max/min = that pose **minus a margin** (start at ~80 counts / ~7°). Never command the hard stop.

## Order (distal first — less mass, easier to hold)

1. **open hand** / **close hand** (17) — **done**
2. **twist** clockwise / counterclockwise (16) — watch for 0/4095 wrap.
3. **wrist up** / **wrist down** (15)
4. **bend** / **straighten** (14)
5. **raise** / **lower** (12+13) — one motion, both IDs. Do not force.
6. **turn** left / right (11) — wrap possible.

One phrase. Slow. Support the arm. Say the word out loud. Stop when a cable tightens or plastic meets plastic — not after.

## What I write down

For each phrase: `pos` at the stop, sign of Δ (`+` or `−`), whether 12/13 mirrored, cable note. Then a conservative `[min, max]` for firmware later.

Table (empty until we run it):

| Say | ID | Stop `pos` | Δ sign | Cable / hit | Soft limit later |
| --- | --- | --- | --- | --- | --- |
| open hand | 17 | **609** | − | **cable/collide wall.** Do not go below (569 was past it) | ≥ 689 |
| close hand | 17 | **2088** (×4) | + | jaws meet; never above | ≤ 2008 |
| twist CW | 16 | | | | |
| twist CCW | 16 | | | | |
| wrist up | 15 | | | | |
| wrist down | 15 | | | | |
| bend | 14 | | | | |
| straighten | 14 | | | | |
| raise | 12+13 | | | | |
| lower | 12+13 | | | | |
| turn left | 11 | | | | |
| turn right | 11 | | | | |

## How

Same as hand-move: pack on, `tq=0`, `python3 scripts/watch-hand.py`.

If `tq=1`, unplug XT60.
