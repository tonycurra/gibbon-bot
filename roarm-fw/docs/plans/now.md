# Where we are

You write here; this page is the pile sorted. Words: `docs/hardware/motors/names.md`.

## Not mixed — three jobs

| Job | Question | Status |
| --- | --- | --- |
| **Safety / bring-up** | Chip up? Pack ok? `tq=0` unless we asked? No boot move? | **Done.** `usb-start.md`, `usb-battery-on.md` |
| **Limits** | Where does a cable cut or plastic hit? Soft band inside that. | **Hand done. Twist walls known; commanded check failed (cable).** Flash ESTOP, then wrist → turn |
| **Calibration** | What does a count *mean*? Home, degrees, later FK / sim. | **After walls.** Signs for the hand already fell out of limits (open −, close +) |

Limits and calibration **share a sitting** (you move, we record `pos` and the word), but they are not the same write-up. Limits = do not break. Calibration = map to math. We do not calibrate a joint we have not walled. Collision (config hits, later FK) is a fourth job: `docs/plans/collision.md`.

Commanded motion is a **check** on a finished wall (hand: `hand open` / `close` / `to`), not a way to find the wall. Find the wall by hand, `tq=0`. Then I move inside the soft band. How we move (inner servo PID, speed/acc, `load` stop): `docs/plans/control.md`.

## Hand (ID 17) — done enough to leave

- Open wall **609** (cable/collide). Do not go below. 569 was past it.
- Close wall **2088** (jaws meet).
- Soft command: **689–2008**. Probe max **2075**.
- Contact `load` showed up ~**2073**, not at 2008. Conservative close is fine.
- Gold: `docs/baselines/hand-open-close.md`. CSV: `docs/baselines/data/hand-contact.csv`.

## Twist (ID 16) — walls yes, commanded check failed

Home **4068** slack (you). CW **978**. CCW **`un` 7126**. Soft `un` ~1058–7046. `docs/baselines/twist.md`.

Sitting log (every trial + error): `docs/baselines/twist-sitting-2026-09-05.md`.

`scripts/live.py --check twist` first run **failed** and wound the cable: host ignored `load` 1137→1300 (abort was 1600), chip did **not** watch during wheel, bus died, motor kept spinning. You cut XT60.

**No twist command until the chip guard is flashed** (`ESTOP` on `Δload` / `cur` / bus miss). Sensors: `docs/hardware/motors/sensors.md`. Then `get` must work, look at the loom, slack **4068** only.

## Next sitting

1. **Flash** twist `ESTOP` (`Δload` / `cur` / bus miss). No twist command before that.
2. Loom slack at **4068**, `get` works, then retry `--check twist` if we still want it.
3. **Wrist** up / down (15). Park, 1D walls, then one coupling check: `docs/plans/collision.md`.
4. Then bend, raise (12+13), turn. Only after a wall exists: a slow `to` inside that band.
5. **Calibration** when all six moves have walls: one table of home + min/max + deg/count. Then FK. BLE after USB is boring.

Pack-off hand-move (silence) can wait. It does not protect cables.

## How we save as we go

- Walls and signs → that joint’s baseline (`hand-open-close.md`, later `twist.md`, …) and the table in `joint-limits.md`.
- Raw stream → `docs/baselines/data/*.csv` or `*.txt`. CSV for anything with `load` / commanded steps.
- Software changes → `docs/software.md` plus the script header.
- Git: one commit when a test has a result (this hand block). Not a commit per chat line.
