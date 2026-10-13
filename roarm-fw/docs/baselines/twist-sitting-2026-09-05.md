# Twist sitting — 2026-09-05

Wrist roll, ID **16**, ST3215. Stand behind the base. Twist = looking along the forearm toward the fingers.

This page is the **trial log**: walls you set, commanded runs, errors, and comments. Limits gold (how the joint works): `twist.md`. Sensors: `hardware/motors/sensors.md`. Software / how to rerun: `software.md`. Pile: `plans/now.md`.

**Do not command twist again until the chip `ESTOP` guard is flashed** and the loom is slack at raw **4068**.

## Setup

| | |
| --- | --- |
| Chip | ESP32-WROOM-32, MAC `44:1d:64:86:3f:54`, IDF v5.4.2 |
| Port | `/dev/ttyACM0` 115200. UART switch **ESP32** |
| Pack | XT60 → servo VIN ~**11.3 V** (3S). USB powers the chip only |
| Human | Tony. Agent commanded some runs |
| Words | your clockwise = counts **fall**; your CCW = counts **rise** |

## What is true (keep)

Same raw `pos` (0–4095) is **three cable states**: twisted −1 / **slack home** / twisted +1. Only slack is home. Encoder cannot tell which after pack-off.

| Name | raw `pos` | `un` | Notes |
| --- | --- | --- | --- |
| Slack mid (you) | **4068** | **4068** | Seam. You: “here, mid again” after both walls |
| CW wall (you, `tq=0`) | **978** | **978** | No wrap. ≈ **−269°** |
| CCW wall (you, `tq=0`) | **3030** | **7126** | +1 wrap. ≈ **+271°** |
| Soft band | — | **~1058…7046** | 80 inside each wall |

`twist mid` in firmware is a **stale constant (4050 / older 2460)**. Never use it as home.

`load` is **0** with `tq=0`. With `tq=1`, minus cruise ~**110**, plus cruise ~**1130**. Bias, not bind. Bind is a **rise from that cruise**, or `cur` leaving the 0–9 band.

`WritePos` is linear. It will not short-hop across 0. Wheel mode is EEPROM — always restore servo.

## Raw files

| File | What |
| --- | --- |
| `data/twist-user-ccw-cw.csv` | You spun by hand, `tq=0`. 2441 rows. Six wraps. Start 2463, unwrapped 993…9874, end 4010 |
| `data/twist-mid-listen.csv` | Listener after slack 4068. Host file had poisoned `turns=1` (`un=8164`) while you sat at 4068 |
| `data/twist-commanded.csv` | `--check twist` listen+move. **Failed. Wound the cable.** 301 rows. Twist lines below |
| `data/twist-listen.csv` | Earlier listen (552 rows) |
| `data/twist-fullspin.csv` | Stub / aborted full-spin (1 row) |
| `data/twist-watch.txt` | First USB+pack watch, ping 7/7, VIN ~11.4 |
| `data/twist-absolute.json` | Host memory: home **4068**, walls 978 / 7126. Last pose stale after pack pull |

## Trials (this sitting)

### T1 — You find the walls (`tq=0`)

You backdrove. Listener on. `load` stayed **0** (torque off — no strain report).

- Slack mid recorded at **4068**, then again after both walls.
- CW stop **978** (counts fall, no wrap).
- CCW stop **3030**, `un` **7126** (+1 wrap).
- Comment: this is the only trustworthy wall set. Commanded motion is a **check inside** these numbers, not a way to find them.

CSV: `twist-user-ccw-cw.csv`. Also a listen restart at 4068 that wrote `turns=1` into the json (false wrap from a fast 2300→35 sample earlier). **Reset `turns=0` at 4068.** Do not load that `un=8164` as truth.

### T2 — Early commanded jogs (wheel, various “mids”)

Wheel ±270° from guessed mids. Mechanically the horn moved. Home **drifted**:

- `twist free` coasts ~**60°** (example park **692**, later **457**). We then saved the park as if it were mid.
- Fast return 2300→35 fooled unwrap (`turns=+1`).
- `twist servo` / `twist wheel` **miss once** often (`mode=-1`). Next `speed` aborts.
- `WritePos` hop check was circular, then fixed to linear `|goal−cur|>900`.
- Comment: listen process killed, one-shot script started its own `turns=0`. That switch is why mid kept looking “off.”

### T3 — “Try fast” (speed **520**, throwaway script)

Start: slack **4068**, `tq=0`.

| Leg | What happened | End |
| --- | --- | --- |
| CW | `twist wheel` then `speed -520` → **abort mode=-1**. Sat **3404** `HELD` | Never reached 1200 |
| “Mid” | Wheel +520, brake at 3981, coast across 0 | **pos=50**, `un=4146` — called mid |
| CCW | Worked. Last moving 6858, coast | **`un=7022`** (wall 7126; 104 inside) |
| Mid | Crossed wrap 112→4055 | **3890** |
| `free` | Coast | **457**, tracker `turns=1` `un=4553` |

Comment: I printed `AT mid` on 50 and 3890. Then `free` dumped torque. Persist file got **457**. You asked why the offset was wrong — because we never held **4068** and we saved the coast.

### T4 — `live.py --check twist` (listen+move, speed **300**) — **FAILED, cable**

CSV: `data/twist-commanded.csv`. Start leftover park **456**, `tq=0`, VIN 11.3.

| t_s | note | pos | un | load | cur | Comment |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | rest | 456 | 456 | 0 | 0 | Not 4068 |
| 4.0 | to-mid | 3811 | −285 | 109 | 1 | Aimed nearest 4068 (short CW across 0). First sample already past target |
| 5.5 | stop | 3473 | −623 | 17 | 0 | Overshoot + coast |
| 7.7 | hold | 40 | 40 | 0 | 0 | Seam other side. **Script called `here` — wrong home** |
| 9.2 | here | 40 | 40 | 0 | 0 | Home rewritten to 40. From here, “CW 1200” is **your CCW** |
| 10.9–16.5 | “cw” | 3551→1160 | −545→1160 | **1133–1137** | 1–2 | Plus-bias cruise. Going the wrong way |
| 18.1 | cw-stop | 1498 | 1498 | 0 | 0 | Past 1200 on the plus side |
| 21–25 | to-mid | 1457→68 | | **109–117** | 1–2 | Minus cruise. Toward fake home 40 |
| 29.1 | hold | 391 | 391 | 0 | 0 | Near 40, not 4068. Called mid |
| 30.9–48.9 | ccw | 3865→1178 | −231→5274 | **1133–1141** | 1–2 | Long plus run. `cur` still 1–2 |
| **50.0** | ccw | 1503 | **5599** | **1220** | **9** | **Δload +83, cur 2→9. First tighten.** Host abort was 1600 — ignored |
| **51.1** | ccw | 1823 | **5919** | **1300** | **19** | **Δload +163, cur 19. Cable.** Should have ESTOP |
| 52.3 | ccw | 2189 | **6285** | 1217 | 9 | Still speed 300 |
| 53.8 | ccw-stop | 2189 | 6285 | 1217 | 9 | `speed 0` **abort mode=-1**. Still moving |
| 54.9 | hold | 2189 | 6285 | 1217 | 9 | Wheel/servo **miss**. Bus dying |

Then all `get` **miss**, `bus_rx` stuck at 98. Motor still commanded speed 300. **You disconnected XT60** so the cable would not break.

Emergency stop from the laptop after that also missed (bus already dead).

Home in `twist-absolute.json` was restored to **4068**. Last recorded pose 2189 / `un` 6285 is **stale** (pack-off while moving). Look at the loom.

## Errors (keep these)

| ID | Error | What it cost |
| --- | --- | --- |
| E1 | Listen and move in **two processes**. Kill listener → new `turns=0` | Mid “offset” every switch |
| E2 | `twist free` after a check | Coast ~60°; we saved 692 / 457 as home |
| E3 | Call whatever we land **mid** | 50, 3890, 40, 391 ≠ 4068 |
| E4 | Auto-`here` at **40** (seam neighborhood ≠ 4068 side) | Next “CW” was your CCW; wound plus |
| E5 | Abort on **absolute** `load≥1600`, not **Δload** from cruise | Ignored 1137→1220→1300 |
| E6 | Chip **does not poll** during `twist speed N` | When USB/`get` died, wheel kept pushing |
| E7 | Wheel/servo **miss once**; `speed` then `mode=-1` | First CW never started (T3); stop failed (T4) |
| E8 | Full `get` of 7 IDs ~1 s while wheel is live | Starves the bus; last stop missed |
| E9 | Fast sample across 0 fools unwrap (2300→35, 3473→40) | Fake `turns` |
| E10 | Firmware `twist mid` 4050 / old 2460 | Wrong number if anyone types it |

## Comments (do not forget)

- There is **no environment sensor** and **no torque cell**. `load` + `cur` while `tq=1` are the only bind signals. `docs/hardware/motors/sensors.md`.
- Plus-direction `load` ~1100 is **cruise**, not a hit. The hit is the **rise** (T4 t=50–51) and `cur` leaving 0–2.
- `cur` did move: 1–2 → **9** → **19** on the same samples as `load` 1137→1220→1300. We had the sensor. We did not cut.
- After pack-off, `turns` is a lie. File is stale. Loom first, then `here` only on the **4068 side** of the seam (not 0–80).
- One ACM0 process: `scripts/live.py`. Chip must own ESTOP so a dead laptop cannot keep the motor spinning.

## Firmware written, not on the chip yet

`main/main.c`: while `twist speed ≠ 0`, poll ID 16 ~25 ms. `ESTOP` (speed 0, `tq=0`, servo) on `Δload≥100`, `|load|≥1350`, `|cur|≥80`, or 3 bus misses. Commands `estop`, `get 16`.

Flash before any other twist command. USB is enough to flash; XT60 off until then is safer.

## Next (after flash + loom slack)

1. `get` answers on 11–17.
2. You confirm slack mid **4068**. `here` only then.
3. Retry `--check twist` with chip guard live. Soft 1200 / `un` 6900. No `free`. Hold 4068.
4. Then wrist **15** walls by hand, `tq=0`.
