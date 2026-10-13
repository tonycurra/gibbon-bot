# Hand limits — open / close (ID 17)

USB + XT60, `tq=0`. You moved only the jaws. Host `scripts/watch-hand.py`.

Stand behind the base. Words: `docs/hardware/motors/names.md`.

## Sign

| Say | `pos` |
| --- | --- |
| **open hand** | decreases (−) |
| **close hand** | increases (+) |

## Stops (you set these)

| | `pos` | Notes |
| --- | --- | --- |
| Close, hard | **2088** | Hit four times. You parked again at **2085–2086**. Jaws meet. |
| **Open limit** | **609** | This pose. Held. **Do not go below.** More open = cable cut / parts collide. |

Safe span 2088−609 = **1479 counts ≈ 130°**.

**Over-limit (do not repeat):** `pos` **569** at t=625.9, and 593 / 596. Those are past the cable. Treat as a scrape, not a target.

## Other `get` fields (ID 17, this pass)

| Field | While moving | At a stop |
| --- | --- | --- |
| `speed` | ±250 to −950 / +800 | 0 |
| `load` | 0 | 0 |
| `cur` | 0 | 0 |
| `moving` | 0 | 0 |
| `tq` | 0 | 0 |
| `v` | pack ~11.4 | pack ~11.4 |

No force signal. Limits are **`pos` only**.

## Soft numbers (on the chip now)

`hand open` → **689**. `hand close` → **2008**. Speed 300, acc 20. ID 17 only. Pack must be ≥ 10.5 V. `hand free` drops torque.

Never command ≤ 609 or ≥ 2088. **609 is a wall, not a goal.**

## Commanded run

Start 2085 (`tq=0`). Open: 2085 → 1838 → 1458 → 1121 → 740 → **691** (goal 689). Close: 691 → 974 → 1316 → 1650 → **2003** (goal 2008). IDs 11–16 did not move.

With torque on, `moving=1` and `load` live (open ~56, close hold ~1056). Then `hand free`: `tq=0`, `load=0`, parked at 2003.

## Contact probe (nudge past 2008)

`scripts/watch-csv.py` → `docs/baselines/data/hand-contact.csv`. Steps 2008 → 2030 → 2050 → 2065 → 2075, then free. Chip clamp max 2075 (wall 2088).

| Goal | Settled `pos` | `load` | Read |
| --- | --- | --- | --- |
| rest | 2003 | 0 (`tq=0`) | free |
| 2008 | 2003 | **1056** | 5 counts short, holding hard (stiction / leftover squeeze) |
| 2030 | 2026 | **1048** | still high |
| 2050 | 2049 | **0** | arrived. jaws not hard-contacted |
| 2065 | 2064 | **0** | arrived. still free |
| 2075 | 2073 | **1032** | 2 short. first clean “touching” after a free hold |
| free | 2073 | 0 | `tq=0` |

`cur` stayed 0. `temp` 33–34. IDs 11–16 still. So conservative 2008 is **not** jaw contact. Contact effort shows up near **2073**. `load` is controller effort, not a torque cell — it also spikes on a tiny position error.

## Trace (hand only)

```
2086 → 1811 → 1012          first open, held ~1012
1012 → 1245 → 1998 → 2088   close, stop 2088
2088 → 2057 → 1586 → 1103 → 681 → 593   open
593  → 701 → 1686 → 2088    close, stop 2088
2088 → 1604 → 756 → 569     open, farthest
569  → 579 → 761 → 1567 → 2088   close, stop 2088
2088 → 1777 → 899 → 596 → 609    open, held 609
```

Raw session: `docs/baselines/data/hand-move-watch.txt`.
