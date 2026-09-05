# Where we are

You write here; this page is the pile sorted. Words: `docs/hardware/motors/names.md`.

## Not mixed — three jobs

| Job | Question | Status |
| --- | --- | --- |
| **Safety / bring-up** | Chip up? Pack ok? `tq=0` unless we asked? No boot move? | **Done.** `usb-start.md`, `usb-battery-on.md` |
| **Limits** | Where does a cable cut or plastic hit? Soft band inside that. | **Hand done.** Twist → turn still open |
| **Calibration** | What does a count *mean*? Home, degrees, later FK / sim. | **After walls.** Signs for the hand already fell out of limits (open −, close +) |

Limits and calibration **share a sitting** (you move, we record `pos` and the word), but they are not the same write-up. Limits = do not break. Calibration = map to math. We do not calibrate a joint we have not walled.

Commanded motion is a **check** on a finished wall (hand: `hand open` / `close` / `to`), not a way to find the wall. Find the wall by hand, `tq=0`. Then I move inside the soft band.

## Hand (ID 17) — done enough to leave

- Open wall **609** (cable/collide). Do not go below. 569 was past it.
- Close wall **2088** (jaws meet).
- Soft command: **689–2008**. Probe max **2075**.
- Contact `load` showed up ~**2073**, not at 2008. Conservative close is fine.
- Gold: `docs/baselines/hand-open-close.md`. CSV: `docs/baselines/data/hand-contact.csv`.

## Next sitting

1. **Twist** (16) — you, one phrase, CW then CCW, stop at taut/hit. I record `pos` and sign.
2. Same for wrist, bend, raise (12+13 together), turn.
3. Only after a wall exists: a slow `to` inside that band (firmware does not have those writers yet).
4. **Calibration** when all six moves have walls: one table of home + min/max + deg/count. Then FK. BLE after USB is boring.

Pack-off hand-move (silence) can wait. It does not protect cables.

## How we save as we go

- Walls and signs → that joint’s baseline (`hand-open-close.md`, later `twist.md`, …) and the table in `joint-limits.md`.
- Raw stream → `docs/baselines/data/*.csv` or `*.txt`. CSV for anything with `load` / commanded steps.
- Software changes → `docs/software.md` plus the script header.
- Git: one commit when a test has a result (this hand block). Not a commit per chat line.
