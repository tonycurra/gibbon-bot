# Software (what we run)

Two sides. Chip talks Feetech on UART1. Laptop talks the same text on USB (`/dev/ttyACM0` 115200) or BLE (`gibbon-XXXX`, Nordic UART). **One process owns ACM0** — do not run `idf.py monitor` and a Python script together.

**Sitting UI:** `../gibbon-console` (Django: overview / joint / calibrate / info / trials). Docs: `../gibbon-console/docs/README.md`. Chip BLE: `docs/ble.md`. CLI: `scripts/live.py`.

Flash / stop monitor: `docs/framework/espidf.md`.

**Sitting tool:** `scripts/live.py` — listen (`get` loop + host unwrap) and send moves in the same process. Do not kill a listener to run a one-shot; that resets `turns` and is why mid drifted.

## Tests (run these)

Stand behind the base. Words: `docs/hardware/motors/names.md`. Walls by hand, `tq=0`. Commanded check only after that wall exists. CSV columns: `t_s,note,id,name,pos,un,turns,speed,load,cur,moving,tq,v,temp`.

| Test | How | Writes | Gold write-up |
| --- | --- | --- | --- |
| USB start, pack off | `state` / heartbeats | — | `baselines/usb-start.md` |
| USB + XT60, `tq=0` | `ping` `get` `batt` | — | `baselines/usb-battery-on.md` |
| Encoders live (backdrive) | `python3 scripts/watch-hand.py` | `data/hand-move-watch.txt` | `baselines/hand-move-usb-battery-on.md` |
| Hand walls (you) | `watch-hand.py` or `live.py`, `tq=0` | — | `baselines/hand-open-close.md` |
| Hand commanded + contact | `watch-csv.py --out …/hand-contact.csv --steps 2008,2030,2050,2065,2075` | `data/hand-contact.csv` | same |
| Twist walls (you) | `live.py` or `watch-csv.py`, `tq=0`, say CW / CCW | `data/twist-user-ccw-cw.csv` | `baselines/twist.md` |
| Twist commanded check | `python3 scripts/live.py --check twist --out docs/baselines/data/twist-commanded.csv` | `data/twist-commanded.csv` | `baselines/twist.md` + **sitting log** `baselines/twist-sitting-2026-09-05.md` |
| Twist sitting (live) | `python3 scripts/live.py` then `here` / `twist …` / `q` | `data/twist-absolute.json` | `baselines/twist.md` |

```bash
cd /home/t/code/gibbon-bot/roarm-fw
python3 scripts/listen.py
python3 scripts/live.py
python3 scripts/live.py --check twist --out docs/baselines/data/twist-commanded.csv
python3 scripts/watch-hand.py
python3 scripts/watch-csv.py --out docs/baselines/data/hand-contact.csv --steps 2008,2030,2050,2065,2075
```

`live.py --check twist`: one unwrap. Wheel toward raw **4068**, `here` **only** if `pos` is on the 4068 side of the seam (not 0–80), CW soft **1200** (wall 978), hold 4068, CCW `un` **6900** (wall 7126), servo-hold 4068. Confirms wheel/servo before speed. **No `twist free`**. Refuses `twist mid` (firmware 4050).

**2026-09-05 first `--check twist` failed.** Started at 456. Overshot seam to 40 and auto-`here` there — then “CW 1200” went the wrong way (your CCW). Later the bus wedged (`mode=-1`, all `get` miss) while still in wheel at `un`~6285. CSV: `data/twist-commanded.csv`. Home in the json is **4068** again. Do not rerun until `get` answers. If it is still spinning: XT60 off.

Next test (no writer yet): wrist up / down (15) by hand in `live.py`.

## On the chip — `main/main.c`

**No move on boot.** Writes only if you type them. **ID 17 (hand)** and **ID 16 (twist)** have named commands. Any ID: `id N free|hold|to G|nudge D`. IDs 11–15 hop ≤ **80** (no walls). Raise 12+13 is dual-drive.

| You type | What it does |
| --- | --- |
| `get` / `get N` | READ present. Omitted keys = default (speed/load/cur/moving/tq/turns/st/mode 0, `un=pos`). `get 16` while twist is moving |
| `stream` / `stream N` / `stream 0` | live `get` when an ID **changes** (pos Δ≥2, speed Δ≥80, load, cur, tq, v, temp). Default 180 ms poll. **Flash owed** for compact lines |
| `who` / `who N` | EEPROM fw/mode/PID/angle limits/goal |
| `id N free\|hold\|to G\|nudge D` | any 11–17. 11–15 hop ≤80 |
| `move speed N` / `move acc N` | next WritePos (default 300 / 20) |
| `estop` | all ids `tq=0` + twist speed 0 + servo. Chip also cuts twist on `Δload` / `cur` / bus miss |
| `hand open` / `close` / `to N` / `free` | ID 17. Soft **689–2008**, `to` max **2075**. Speed 300, acc 20 |
| `twist to N` | ID 16, 0–4095, linear hop ≤ **900** (WritePos does not short-hop across 0) |
| `twist mid` | **do not use** — stale goal **4050**, not slack 4068 |
| `twist free` | torque off 16 (coasts). Prefer `q` on the host |
| `twist here` | print current pos as cable-home; host still owns `turns` |
| `twist servo` / `wheel` | mode 0 / 1 via REG_MODE 33 (EEPROM — always restore servo). Retry on miss |
| `twist speed N` | wheel jog, signed. Abort if not wheel. 0 = stop |
| `batt` `ping` `bench` `state` | read-only |

Pack cutoff 10.5 V in `joint_to`. Hand walls: open **609**, close **2088**. Twist walls: CW **978**, CCW `un` **7126**. Generous cable-only: **1127** / **3798**. Never command the walls.

Host parsers (`listen.py`, `live.py`, `watch-*.py`, gibbon-console) use `key=value` so compact and old fat `get` lines both work.

Chip `turns` starts at 0 each boot. Host `docs/baselines/data/twist-absolute.json` is the memory. Same raw `pos` is three cable states — only slack is home. `load` is 0 with `tq=0`. Host aborts a check on stall or `|load|` past the usual bias (plus ~1200 / minus ~170), not only 2000. IDs 11–15: hop ≤80 only.

## On the laptop

| Script | Why |
| --- | --- |
| `scripts/listen.py` | Live 7-row sensor table (overwrites in place). READ only |
| `scripts/live.py` | Listen + control. `--check twist` for the commanded roll. `q` = servo, no free |
| `scripts/watch-hand.py` | Backdrive only. You turn, it prints Δ |
| `scripts/watch-csv.py` | 7-row table + CSV. `--steps` is the old hand contact probe |

`watch-csv.py` may still log pose into `twist-absolute.json` but must **not** overwrite `home_*` or walls.

## Data we keep

| File | What |
| --- | --- |
| `docs/baselines/data/hand-move-watch.txt` | Raw host printout, first backdrive |
| `docs/baselines/data/hand-contact.csv` | Commanded close-nudge, `load` vs `pos` |
| `docs/baselines/hand-move-usb-battery-on.md` | Encoders live, `tq=0` |
| `docs/baselines/hand-open-close.md` | Hand walls + commanded run + contact probe |
| `docs/baselines/data/twist-user-ccw-cw.csv` | You spun twist; wraps |
| `docs/baselines/data/twist-commanded.csv` | Host `--check twist` (listen + move) |
| `docs/baselines/data/twist-absolute.json` | Last twist pose + **home_raw 4068** + walls. Pose updates; home only on `here` |
| `docs/baselines/twist.md` | Twist walls, unwrap, commanded check |
| `docs/baselines/twist-sitting-2026-09-05.md` | **This sitting:** trials T1–T4, errors E1–E10, load/cur table |
| `docs/baselines/data/twist-sitting-2026-09-05.json` | Same, machine-readable |
| `docs/baselines/data/twist-commanded.csv` | Failed `--check twist` (wound cable) |

## Not this folder

No MySQL. IDs 11–15: hop-limited `id N` only (no walls). BLE is the same text pipe as USB (`docs/plans/ble-html.md`).
