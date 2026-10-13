# Hand-move — USB + XT60, torque off

Gold for **backdrive by hand**. USB in, pack on, UART switch **ESP32**. Firmware does not command motion. You hold the arm and turn joints. Encoders must move on `get`.

Captured 2026-09-05, firmware `listen v1`. Same chip as `usb-start.md` / `usb-battery-on.md`.

If `tq` becomes `1`, stop and unplug XT60.

## Setup

| | Must be |
| --- | --- |
| USB-C | ESP32 debug port, `/dev/ttyACM0` |
| UART switch | **ESP32** |
| XT60 | **on** |
| Firmware | `listen v1` — READ only. No `WritePos`, no torque write |
| Host | `scripts/watch-hand.py` (not `idf.py monitor` at the same time) |

Match `usb-battery-on.md` first (`found=7/7`, `batt`, `tq=0`). Support the arm. Dual-drive shoulder (12+13) is one joint — move slowly.

## Software

Two programs. Chip never writes a position. Host never writes a position.

### On the chip — `listen v1` (`main/main.c`)

Already flashed. USB text commands. `get` is one READ of present (reg 56, 15 bytes) plus torque-enable (reg 40) for IDs 11–17. Fields: `docs/hardware/motors/sensors.md`.

Servos do **not** broadcast. Sitting on heartbeats shows `bus_rx=0` even while you turn a horn. You only see `pos` if something sends `get`.

### On the laptop — `scripts/watch-hand.py`

Python + pyserial. Opens ACM0 at 115200. Sends `state`, `gpio`, `ping`, `batt`, one rest `get`, then `get` in a loop (~1.2 s per sample: 0.85 s drain + 0.35 s pause).

Prints a line only when `|Δpos| ≥ 2` or `speed` / `load` / `cur` / `moving` is non-zero. Idle pose reprints every ~8 samples (`still`). Exits if any `tq=1`.

```bash
cd ~/code/gibbon-bot/roarm-fw
python3 scripts/watch-hand.py
```

Stop: Ctrl+C in that terminal. One process owns ACM0 — do not start `idf.py monitor` while this runs.

This is **not** a firmware `watch` command. The chip still answers one-shot `get`. The laptop is the loop.

## What we learned (this run)

| Field | Hand-move, `tq=0` |
| --- | --- |
| `pos` | **Lives.** Tracks the horn. |
| `speed` | **Lives.** Peak here ~1500 steps/s (elbow). 0 when parked. |
| `load` | Stayed **0**. Not a hand-force signal. |
| `cur` | 0 or 1 (noise). Not useful with torque off. |
| `moving` | Stayed **0**. Means “commanded move,” not “horn turning.” |
| `tq` | Stayed **0**. |
| `v` | Pack ~11.3–11.5 V, flat. |

**12 and 13 are one joint.** Move the shoulder: counts go opposite, same size (example +636 / −644).

One `get` miss on id=11 at t=137 s, then it answered again. Bus glitch, not a dead motor.

## Session (~5 min watch)

Pack `11.4 V` `ok`. `found=7/7`. Chip `mac=44:1d:64:86:3f:54`, heap 297940, `reset=poweron`. Two hand bursts, then park.

Counts: 4096 = 360° (~0.088°/count). Mid ≈ 2048.

### Rest (before any hand move)

```
id=11 base         pos=1131 speed=0 load=0 v=11.5 temp=25 cur=0 moving=0 tq=0
id=12 shoulder-A   pos=1921 speed=0 load=0 v=11.4 temp=26 cur=0 moving=0 tq=0
id=13 shoulder-B   pos=2173 speed=0 load=0 v=11.4 temp=28 cur=0 moving=0 tq=0
id=14 elbow        pos=1904 speed=0 load=0 v=11.4 temp=27 cur=0 moving=0 tq=0
id=15 wrist-pitch  pos=2147 speed=0 load=0 v=11.4 temp=27 cur=0 moving=0 tq=0
id=16 wrist-roll   pos=2893 speed=0 load=0 v=11.3 temp=25 cur=0 moving=0 tq=0
id=17 gripper      pos=2088 speed=0 load=0 v=11.4 temp=27 cur=0 moving=0 tq=0
```

### Burst 1 — t=26–43 s

1. Gripper open–close: 2088 → **932** (`speed` −950) → 2087 (`speed` +1100). ~102° then back.
2. Shoulder only: 12 **1921 → 2557 (+636)**, 13 **2173 → 1529 (−644)**, `speed` ±700.
3. Whole-arm grab: base +33°, elbow +100°, wrist pitch −58°. Shoulder snapped back toward mid (again 12/13 mirror). Wrist roll not turned.

Parked ~3.5 min at:

`11:1510 12:1865 13:2231 14:3038 15:1483 16:2894 17:2087`

### Burst 2 — t=243–270 s

Random hand again. Elbow `speed` −1200 / +1500. Base yaw sweep 1511 → **2020** → **1218** in two samples. Gripper open–close again (2086 → **965** → 2088). Wrist pitch 1483 → 2239 → 1478. **Wrist roll finally moved** 2894 → **3033** (+139, ~12°).

Parked (stable through t≈476 s):

```
11:1237 12:1984 13:2110 14:2836 15:1478 16:3033 17:2086
```

Raw lines: `docs/baselines/data/hand-move-watch.txt`.

### Net, first rest → last park

| ID | Joint | Rest | Last | Δ counts | ≈ deg |
| --- | --- | --- | --- | --- | --- |
| 11 | base | 1131 | 1238 | +107 | +9° |
| 12 | shoulder A | 1921 | 1984 | +63 | +6° |
| 13 | shoulder B | 2173 | 2110 | −63 | −6° |
| 14 | elbow | 1904 | 2836 | +932 | +82° |
| 15 | wrist pitch | 2147 | 1478 | −669 | −59° |
| 16 | wrist roll | 2893 | 3033 | +140 | +12° |
| 17 | gripper | 2088 | 2086 | −2 | opened and closed |

## How to re-run

1. USB in, pack on, switch ESP32. Match `usb-battery-on.md` if this is a new plug.
2. `python3 scripts/watch-hand.py`
3. Confirm rest `get` has seven lines and `tq=0`.
4. Support the arm. Move one joint, then another. Watch Δ lines.
5. Ctrl+C. Paste rest + last `still` + any surprise (tq, miss, load) into this page or a dated log.

Next: names `docs/hardware/motors/names.md`, then one-joint soft stops `docs/plans/joint-limits.md`. Pack-off silence check can wait.
