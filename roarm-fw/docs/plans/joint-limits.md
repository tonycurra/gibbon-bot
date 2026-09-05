# Joint limits — one move at a time

**Numbers:** `docs/hardware/motors/joint-limits.md`. Encoders work (`docs/baselines/hand-move-usb-battery-on.md`). **Soft stops** so later commands cannot wrap a cable or hit plastic.

Hand (17) is done: `docs/baselines/hand-open-close.md`. Chip may `hand open` / `close` / `to` / `free` — **ID 17 only**. Other joints: you still move, I record. No writer until that wall exists.

Words: `docs/hardware/motors/names.md`.

## Why not 0–4095

The encoder is a full circle. The **arm is not**. Stock UI already clips most joints to ~180° (shoulder, elbow, wrist nod) and the gripper to ~135°. Cables on this HAT swap are the thing that shorts. Stop at the first taut loom or self-hit, then back off.

Software max/min = that pose **minus a margin** (start at ~80 counts / ~7°). Never command the hard stop.

## Order (distal first — less mass, easier to hold)

1. **open hand** / **close hand** (17) — **done**
2. **twist** (16) — walls known (you). Slack mid **4068**. Soft `un` ~1058–7046. `docs/baselines/twist.md`. Sitting trials/errors: `docs/baselines/twist-sitting-2026-09-05.md`. **No command until chip ESTOP is flashed.**
3. **wrist up** / **wrist down** (15) — first link-vs-link joint. Config hits: `docs/plans/collision.md`.
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
| twist clockwise (you) | 16 | **978** | − | from slack mid **4068**, no wrap, ≈269° | ≥ 1058 |
| twist counterclockwise (you) | 16 | **3030** (`un` **7126**, +1 wrap) | + | from slack mid **4068**, ≈271° | `un` ≤ 7046 |
| wrist up | 15 | | | | |
| wrist down | 15 | | | | |
| bend | 14 | | | | |
| straighten | 14 | | | | |
| raise | 12+13 | | | | |
| lower | 12+13 | | | | |
| turn left | 11 | | | | |
| turn right | 11 | | | | |

## How

Pack on. **One process:** `python3 scripts/live.py` (listen + type). Walls: `tq=0`, you move, say the word. After a wall exists: `python3 scripts/live.py --check twist` (or later `--check` for that joint). Catalog: `docs/software.md`.

If `tq=1` and we did not ask, or the bus goes `get` miss while moving: unplug XT60.
