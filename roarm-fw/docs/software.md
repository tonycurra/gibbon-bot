# Software (what we run)

Two sides. Chip talks Feetech on UART1. Laptop talks text on USB (`/dev/ttyACM0` 115200). One process owns ACM0 — do not run `idf.py monitor` and a Python script together.

Flash / stop monitor: `docs/framework/espidf.md`. Commands list there too.

## On the chip — `main/main.c`

Still **no move on boot**. Heartbeats + `get` / `ping` / `batt`. The only write is **ID 17 (hand)**, and only if you type it.

| You type | What it does |
| --- | --- |
| `get` | READ pos/speed/load/V/temp/cur/moving/tq for 11–17 |
| `hand open` | torque on 17, `WritePos` **689** (soft open). Speed 300, acc 20 |
| `hand close` | same, goal **2008** (soft close) |
| `hand to N` | same, N clamped **689–2075** (2075 is inside wall 2088) |
| `hand free` | torque off 17 |
| `batt` `ping` `bench` `state` | read-only, as before |

Clamps and 10.5 V cutoff are in `cmd_hand`. Walls you set: open **609**, close **2088**. We never command those.

Other IDs: no write path yet. Do not add `move 11 …` until that joint has walls.

## On the laptop

| Script | Why |
| --- | --- |
| `scripts/watch-hand.py` | Backdrive. Loops `get`, prints Δ. Use while **you** turn a joint. |
| `scripts/watch-csv.py` | Live 7-row table + CSV. Optional `--steps 2008,2030,…` runs `hand to` then `hand free`. |

```bash
python3 scripts/watch-hand.py
python3 scripts/watch-csv.py --out docs/baselines/data/hand-contact.csv
python3 scripts/watch-csv.py --out docs/baselines/data/hand-contact.csv --steps 2008,2030,2050,2065,2075
```

CSV columns: `t_s,note,id,name,pos,speed,load,cur,moving,tq,v,temp`. One row per motor per sample. Not a database.

## Data we keep

| File | What |
| --- | --- |
| `docs/baselines/data/hand-move-watch.txt` | Raw host printout, first backdrive |
| `docs/baselines/data/hand-contact.csv` | Commanded close-nudge, `load` vs `pos` |
| `docs/baselines/hand-move-usb-battery-on.md` | Encoders live, `tq=0` |
| `docs/baselines/hand-open-close.md` | Hand walls + commanded run + contact probe |

## Not this folder

No MySQL. No BLE yet. No generic joint writer.
