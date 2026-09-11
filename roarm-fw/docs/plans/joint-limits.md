# Joint limits — one move at a time

**Numbers:** `docs/hardware/motors/joint-limits.md`. Encoders work (`docs/baselines/hand-move-usb-battery-on.md`). **Soft stops** so later commands cannot wrap a cable or hit plastic.

Hand (17) is done: `docs/baselines/hand-open-close.md`. Chip may `hand open` / `close` / `to` / `free` — **ID 17 only**. Other joints: you still move, I record. No writer until that wall exists.

Words: `docs/hardware/motors/names.md`.

## Why not 0–4095

The encoder is a full circle. The **arm is not**. Stock UI already clips most joints to ~180° (shoulder, elbow, wrist nod) and the gripper to ~135°. Cables on this HAT swap are the thing that shorts. Stop at the first taut loom or self-hit, then back off.

Software max/min = that pose **minus a margin** (start at ~80 counts / ~7°). Never command the hard stop.

## Order (distal first — less mass, easier to hold)

1. **open hand** / **close hand** (17) — **done**
2. **twist** (16) — Sep-5 walls known (you). Slack mid **4068**. Soft `un` ~1058–7046. `docs/baselines/twist.md`. 2026-09-11 slack **here** was **960** — re-learn Δ from that origin; leftover raw 3108/3111 was not a wall. Sitting: `docs/baselines/twist-sitting-2026-09-05.md`, `development-logs/2026-09-11-calibrate.md`. Hop ≤900. Chip ESTOP is in source, **flash owed**.
3. **wrist up** / **wrist down** (15) — host marks 1126 / 3134, home 2091. Command toward 3134 tripped `ESTOP id=15 dload`. Soft band well inside. Config hits: `docs/plans/collision.md`.
4. **bend** / **straighten** (14) — host marks 995 / 2997, home 2347. Hop ≤80 until NVS walls.
5. **raise** / **lower** (12+13) — one motion, both IDs. SyncWrite in source, **not on chip**. Do not send `raise …` until flash. Do not force.
6. **turn** left / right (11) — wrap. Host home **2084**. Same Δ UX as twist. Hop ≤80.

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
| wrist up | 15 | **3134** (host) | | `ESTOP id=15 dload` on the way — not a soft wall yet | well inside 3134 |
| wrist down | 15 | **1126** (host) | | home **2091** | well inside 1126 |
| bend | 14 | **995** / **2997** (host) | | home **2347**. hop ≤80 | |
| straighten | 14 | same | | | |
| raise | 12+13 | | | SyncWrite unflashed | |
| lower | 12+13 | | | | |
| turn left | 11 | | wrap Δ | host home **2084**. not finished | |
| turn right | 11 | | wrap Δ | | |

## How

Pack on. **One process:** `python3 scripts/live.py` (listen + type). Walls: `tq=0`, you move, say the word. After a wall exists: `python3 scripts/live.py --check twist` (or later `--check` for that joint). Catalog: `docs/software.md`.

If `tq=1` and we did not ask, or the bus goes `get` miss while moving: unplug XT60.
