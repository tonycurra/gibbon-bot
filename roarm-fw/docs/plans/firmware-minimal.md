# Minimal firmware (ESP owns metal, host owns math)

The chip is a **bus + radio**. It does not unwrap, does not do degrees, does not guess battery %, does not learn walls, does not FK. The browser / Django does that.

Flash this as a later sitting. Do not mix it into the current calibrate pass.

## Split

| Chip (must be fast, must work if USB dies mid-move) | Host (browser) |
| --- | --- |
| Read present block (raw) | Unwrap, turns, Δ, degrees |
| WritePos / SyncWrite | Calibrate, slider, named poses |
| **Never command 12 or 13 alone** — `raise G12 G13` only | Pair sign, jog Δ |
| Raw walls 0–4095 on chip (non-wrap). Wrap Δ host-only | Wrist roll / yaw Δ walls |
| `estop` → all `tq=0` | STOP button |
| While `tq=1` and moving: poll **that** ID, cut on `load` / `cur` / miss | Display load, decide next goal |
| VIN as tenths of a volt | 3S % table |
| USB + NimBLE, **same bytes** | Parse, queue, UI |

Servo inner PID stays in the Feetech. We do not retune it on the ESP.

## Strip from the chip

Drop or never call from the hot path:

- `un` / `turns` / `here` unwrap
- `batt` rest-% (`lipo3s_pct`)
- compact ASCII field-omission, hex dumps, `listen`, `pingall`, `who` on every connect
- `twist mid`, `hand open` / `close` as named policy
- hop-walk across 0/4095 (host walks, or later one wrap-step command)
- `stream` of all 7 IDs as fat text

Keep a **debug** text line (`help`, `ping`, `raw N`) behind `debug 1`. Default boot is quiet + binary stream.

## Actuation (small command set)

Text is fine for a sitting. Binary later if BLE is still tight.

```
estop
speed N
acc N
tq N 0|1          # 12 and 13 refused — use raise
to N G            # 11,14,15,16,17. clamp chip wall if set. hop abort if |G-pos| too big until walls exist
raise G12 G13     # INST_SYNC_WRITE only
raise tq 0|1
lim N min|max|home G | lim N clear
stream N          # period ms, 0 = off. default ~50–80 on one moving ID, else all still IDs skipped
```

No `id 12 to`. No `id 13 nudge`. Host already learned that.

## Sensors (raw notify)

Chip timer: read present @56, pack, notify. Host does not `get` in a loop.

One frame, little-endian, no math:

```
g1  t_ms:u32  n:u8
repeat n:
  id:u8  pos:u16  speed:i16  load:i16  cur:i16  v:u8  temp:u8  flags:u8
```

`flags`: tq, moving, miss. `v` is VIN 0.1 V LSB as the servo reports it.

Skip an ID that did not change (pos Δ < 2, load/cur still). **Never skip the ID we are driving.**

If NimBLE notify buffer is full: **drop the frame**, do not block the bus/guard task. USB CDC can take the same blob (or a one-line hex for bring-up).

ASCII fallback while we debug: `p id=16 pos=3884 spd=0 load=0 cur=0 v=112 tq=0` — still no `un`.

## Limits + 12+13

- Non-wrap: chip min/max/home in NVS, clamp `to`.
- Wrap (11, 16): chip does **not** store Δ as min/max. Host clamps Δ, sends a legal `to` on this side of 0/4095.
- 12+13: one SyncWrite. Dual move-guard (either motor trips → both `tq=0`).
- Voltage: refuse WritePos under 10.5 V. No % on chip.

## Guard (not “smart”, just a fuse)

While a goal is in flight, chip polls **only the moving ID(s)** (~20–40 ms):

| Trip | Action |
| --- | --- |
| `\|load\|` ≥ cruise + dload, or hard load | `estop` that move / all |
| `\|cur\|` ≥ threshold | same |
| 3 present misses | same |
| VIN < 10.5 V | same |

Thresholds stay setpoints (`guard dload N`), not computed SoC or angles.

## BLE / serial from the browser

- **One protocol**, two pipes: USB 115200 and NimBLE NUS.
- Browser → Django (or later direct Web Bluetooth) → ACM0 / NUS. Django does not own math either beyond queue + persist walls.
- No WiFi on this WROOM while BLE is on.
- Host **send queue** (already): one command at a time; `estop` jumps. Never spam unknown text (`raise hold` until this firmware is actually on the chip).

Cadence target: notify 20–50 Hz **for the joint in motion**, 5–10 Hz idle for the rest. BLE MTU: one frame ≲ 80 bytes.

## Order to flash

1. Binary (or skinny ASCII) present stream + `estop` / `to` / `tq`. No unwrap.
2. `raise G12 G13` + refuse solo 12/13. Dual guard.
3. NVS walls + clamp on `to`.
4. Stream skip-still + notify drop-not-block.
5. Host: unwrap, calibrate, Δ slider, 3S %, degrees — all already headed that way in the console.

Until (2) is on the metal, the console must not send `raise …`. **(2) flashed 2026-09-12** — console still waits for `info raise=1` so an old image cannot sneak a SyncWrite.
