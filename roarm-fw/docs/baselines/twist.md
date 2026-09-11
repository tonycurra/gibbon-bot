# Twist limits — wrist roll (ID 16)

**Sitting log (trials, errors, comments):** `twist-sitting-2026-09-05.md`  
Machine notes: `data/twist-sitting-2026-09-05.json`

Stand behind the base. **Twist** = looking along the forearm toward the fingers.

`pos` is only 0–4095. That is **one turn of the encoder**, not one turn of the cable. After a wrap, the same number can be a different loom. We keep `un = pos + turns×4096` and save it in `docs/baselines/data/twist-absolute.json`. Chip RAM `turns` dies on reset; the host file is the memory.

**Pack off / unplugged:** the encoder is dead (`get` miss). If the wrist rolls while unpowered, `turns` is a lie. There is no way to recover absolute cable state from `pos` alone. Operational rule: on XT60 connect, file is **stale**. Look at the loom (paint mark / slack). If it matches home, type `twist here` and we zero turns. If it does not, unwind by hand to the mark first. A hall/index on the loom is the later hardware fix — not another wrap of this encoder.

## Visual center (you)

The seam **~4068** (you, back from CCW). Same raw `pos` is **three cable states**:

| Cable | Meaning | We use |
| --- | --- | --- |
| Slack | loom not wound either way | **home, turns=0** |
| Twisted + | one extra turn one way | same raw, `un += 4096` |
| Twisted − | one extra turn the other way | same raw, `un -= 4096` |

Only the slack one is home. After pack-off, the magnet cannot tell which of the three you are in. Look at the loom, then `twist here`.

**Your clockwise** = counts **fall**. Stop **978**. From mid ≈ **−269°**, no wrap.

**Your counterclockwise** = counts **rise**. Stop raw **3030**, `turns=1`, **`un=7126`**. From mid 4041 → wrap → 3030, **+3085** counts ≈ **271°**. Same raw 3030 without the wrap is not this pose.

Soft band: `un` **~1058…7046** (80 inside each wall). Slack mid **4068** (not 40, not 4050).

## Commanded check — `live.py --check twist`

Same process listens (`get` + host unwrap) and jogs. No `twist free`. Mid = raw **4068** on this side of the seam. `here` must not fire at 0–80 (that is the other side; “CW” from there is your CCW).

**2026-09-05 — failed. Wound the cable.** CSV `data/twist-commanded.csv`. Start **456**. Overshot to **40**, auto-`here`. “CW” ran the plus way. `load` cruise **~1137** then **1220 / 1300** — host abort was 1600 so it kept going. Chip did not poll during wheel. Bus died at `un`~6285, motor still speed 300. You cut XT60. Home file is 4068. **No command until chip `ESTOP` is flashed** (`Δload` / `cur` / bus miss). `docs/hardware/motors/sensors.md`.

`WritePos` is **linear 0–4095**. Goal < current always decreases. It will **not** take the short hop 4090→80. From this home, minus is ~347° down to 80. Plus is ~60 counts to 4095, then stop. A commanded full spin the other way needs motor/stepper mode (not on; that setting is saved in the servo).

## Cable walls (you, generous)

| | `pos` | How we got there |
| --- | --- | --- |
| Cable A | **3798** | Counts **fell** from 3001 through 0. Loom quite twisted. |
| Cable B | **1127** | Counts **rose**, wrapped more than once. Loom quite twisted again. |

We do **not** command these. Encoder wrap 0/4095 is the same circle, not extra range.

## What you showed 2026-09-05 (listen, `tq=0`)

CSV: `docs/baselines/data/twist-user-ccw-cw.csv`. Other joints barely moved.

Start **2463**. Six wraps. Unwrapped **993…9874** = **2.17 turns** peak-to-peak (~781°). Ended **4010**, `turns=0` relative to start.

`load` stayed **0** the whole time. With torque off the servo does not report strain. Cable twist will not show up until `tq=1`. Then a sudden `load` jump is a stop.

## Commanded check (earlier, inside old soft band)

Soft **1800–3200** was a no-wrap guess. Sweep 1800↔3200 parked at 2463. First hop from 1157 went the long way through 0. `load` ~110 one way, ~1100 the other — already asymmetric; treat a surprise spike as bind.

Firmware now **refuses** a `twist to` whose linear hop is >2048 counts (would wrap). Clamp is still 1800–3200 until unwrap goals exist. `twist mid` 2460 is **not** your center — do not use it as home.

Never command a raw `pos` as if it were unique. Never command across 0/4095. Watch `load` whenever `tq=1`.

Raw first watch: `docs/baselines/data/twist-watch.txt`.

## 2026-09-11 — new slack, leftover raw, host Δ

Sitting: `development-logs/2026-09-11-calibrate.md`. Console wrap page: degrees + Δ from **here**.

This sitting’s slack **here** was raw **960**, not 4068. Pack-off forgets turns. Same loom, different encoder park after power. **Do not** apply Sep-5 `un` 1058…7046 to home 960.

Leftover host min/max **3108 / 3111** were two raw marks at the same pose (`|a−b|<3`). Not walls. Console clears that pattern on wrap-page load.

Listen after here at 960: now **−103°**, Δ **−1172**, pos **3884**, un **−212**, turns **−1** (`1172 × 360/4096 ≈ 103`). Degrees matched. Wrap Δ walls were not saved (done needs ≥80 counts of sweep after **calibrate**).

`WritePos` still will not wrap. Hop ≤900. Until flash, do not wheel-jog. `twist to 3108` aborted hop (legacy goal).
