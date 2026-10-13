# 2026-09-11 — calibrate, wrap Δ, UART jam

USB in, pack on, Django `:8080`. Stand behind the base. Firmware **on the chip** is still the previous sitting. SyncWrite `raise`, `id N here`, compact `stream`, and extra guard live in `main/main.c` in this tree — **not flashed**.

Sitting UI: `../gibbon-console`. Durable numbers: `docs/baselines/`. Pile: `docs/plans/now.md`. Next chip shape: `docs/plans/firmware-minimal.md`.

## What we decided

| Layer | Owns |
| --- | --- |
| **Chip** | raw present @56, WritePos, walls that fit 0–4095, SyncWrite 12+13, `estop` / load-cur-miss guard |
| **Host** | unwrap, turns, Δ from slack, degrees (`360°/4096`), calibrate UX, 3S %, send queue |

Do not put unwrap, %, or named poses on the ESP. Do not send `raise …` until that firmware is on the metal.

Encoder: **360° = 0–4095**, resolution **360°/4096**. Wrap is the magnet seam at 0/4095, not a cable wall. Firmware wrap: `|Δpos| > 2048` → `turns ±1`. `WritePos` is linear and **will not wrap**. Hop: twist ≤900; 11–15 ≤80 until both NVS walls exist. Pack-off forgets RAM turns.

Wrist roll: looking along the forearm toward the fingers. **CW = counts fall (−Δ). CCW = counts rise (+Δ).**

IDs: **11 yaw**, **12+13 shoulder pitch (paired, never command alone)**, **14 elbow**, **15 wrist pitch**, **16 twist ST3215**, **17 hand ST3215HS**.

## Firmware in this tree (unflashed)

`main/main.c` grew:

- `raise to G12 G13` — one `INST_SYNC_WRITE`. Solo `id 12 to` / `id 13 to` abort.
- `raise hold` / `raise free` — both motors, one packet (hold) / both `tq=0`.
- `id N here` — zero unwrap at current pos. Pack-off forgets.
- Move guard: `ESTOP id=N …` / `ESTOP raise 12+13 …` on `Δload` / hard load / `cur` / bus miss.
- Compact `stream` (omit zeros, skip still IDs) — still owed on the chip from 2026-09-09.

Until flash, the console **must not** send `raise hold` / `raise free` / `raise to`. Hold/free 12+13 as `id 12 hold` and `id 13 hold` (two packets). Slider/nudge for raise still emits `raise to` — **do not jog shoulder pitch from home until flash**.

## Console (landed, hard-refresh)

### Calibrate

- `/calibrate/` = six-joint index. `/calibrate/<slug>/` = one joint.
- Wrap joints (yaw **11**, wrist-roll **16**): Δ-first. **− / now / +** in **degrees**, counts underneath.
- **calibrate** starts learn (tq off). **done** saves any swept span ≥ **80** counts (~7°). Done does **not** require crossing 0° both ways (that looked broken if you started already at −103°).
- Slider is Δ space from − wall to + wall (not raw `pos`). Disabled during learn and before **here**. Speed bar on wrap pages.
- **here** zeros Δ at slack. Host `wrapSess` unwraps so Δ stays monotonic through 0/4095.
- `turns` = signed magnet turns. A **wraps** trip counter (both directions) was removed.
- Leftover twist min/max **3108 / 3111** (same-pose raw marks) treated as legacy: if both >200, both <4096, `|a−b| < 80`, wrap-page load clears them.
- `syncHomeUn`: stored `Limit.home` is always the origin, even if `turns ≠ 0`.
- Test/sweep still off for wrap until a wrap-walk exists.

### Bus / STOP

Chip does not know `raise hold` / `raise free`. Hold-all + STOP flooded UART; Django is single-threaded; `rx err unknown 'raise hold'` interleaved with `estop`. **ESTOP all landed** (`tq=0` including twist).

Fixes:

- Never send `raise hold` / `raise free`. Use `id 12 hold` / `id 13 hold` (and free). Same in `calibrate.js`, `home.js`, `joint.js`.
- `send()` in `core.js` is a queue. One write at a time. `estop` drops pending commands, then goes.
- Calibrate stays on **`stream 180`**. Do not bump to 100. Stream covers live; extra 80 ms `get` loop on the wrap page is gone.

### Trials

First click or live `get` opens a sitting if none is open. `ESTOP` / `guard` / `lim` lines are `Note` rows. Polls (`get`, `stream`) are not clicks.

## Live marks (host `Limit`, this sitting)

These are the notebook on `/calibrate/`, not gold walls. Gold hand/twist still: `docs/baselines/hand-open-close.md`, `docs/baselines/twist.md`.

| ID | Joint | home | − / min | + / max | Notes |
| --- | --- | --- | --- | --- | --- |
| 17 | hand | **1389** | **2088** / **760** (order may swap) | | Gold walls remain open **609**, close **2088**. Soft 689–2075 |
| 16 | twist | **960** | leftover **3108 / 3111** **cleared** | | Sep-5 slack was **4068**. Do not mix origins. Re-learn Δ from this **here** |
| 15 | wrist pitch | **2091** | **1126** | **3134** | Command toward 3134 → `ESTOP id=15 dload`. Do not treat 3134 as a soft wall yet |
| 14 | elbow | **2347** | **995** | **2997** | Host Δ from here. Chip hop ≤80 until NVS walls |
| 11 | yaw | **2084** | (Δ not finished) | | Wrap joint. Same UX as 16 |
| 12+13 | raise | | | | Pair sign on calibrate. No SyncWrite on chip |

Wrist-roll listen example (after here at 960): now **−103°**, Δ **−1172**, pos **3884**, un **−212**, turns **−1**. Degrees matched (`1172 × 360/4096 ≈ 103`).

Django log does not print compact `get` lines (only ESTOP / guard / abort / non-get tx).

## ESTOPs this sitting

- `ESTOP id=17 dload` — hand blocked while opening (cruise too low on a slow open; later hits at real load ~1080). Cut + `tq=0` is the behaviour we want.
- `ESTOP id=15 dload` — wrist pitch toward marked 3134.
- `ESTOP` on 12/13 `cur` during dual nudge (two packets, they can fight).
- **ESTOP all** after hold-all + STOP jammed on unknown `raise hold`.

## Do not

- Flash unless asked (`ACM0` must be free; stop Django first).
- Command 12 or 13 alone until SyncWrite is **on the chip**.
- Send `raise …` until that flash.
- Wheel-jog twist casually. Hop ≤900. `twist servo` if `mode=1`.
- Trust leftover raw min/max that sit on the same pose (`|a−b|<80`).
- Mix twist home **960** (this sitting) with Sep-5 slack **4068** / walls 978 / `un` 7126.

## Next

1. Flash when asked: SyncWrite raise + guard + compact stream. Then console may send `raise to` / `raise hold` / `raise free`.
2. Finish wrap Δ walls on **16** (and **11**) from a known slack **here**.
3. Soft-band wrist pitch inside 1126… well short of 3134. Then elbow check.
4. Host still owns wrap clamp; chip stores raw 0–4095 walls only for non-wrap IDs.
