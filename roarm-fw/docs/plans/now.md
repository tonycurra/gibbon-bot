# Where we are

You write here; this page is the pile sorted. Words: `docs/hardware/motors/names.md` (yaw / pitch / roll + room words). Console: `../../gibbon-console/docs/README.md`.

## Not mixed — three jobs

| Job | Question | Status |
| --- | --- | --- |
| **Safety / bring-up** | Chip up? Pack ok? `tq=0` unless we asked? No boot move? | **Done.** `usb-start.md`, `usb-battery-on.md` |
| **Limits** | Where does a cable cut or plastic hit? Soft band inside that. | **Hand done. Twist walls known; commanded check failed (cable).** Flash ESTOP, then wrist → turn |
| **Calibration** | What does a count *mean*? Home, degrees, later FK / sim. | **After walls.** Console `/calibrate/` can mark home/min/max from live pos. Signs for the hand already fell out of limits (open −, close +) |

Limits and calibration **share a sitting** (you move, we record `pos` and the word), but they are not the same write-up. Limits = do not break. Calibration = map to math. We do not calibrate a joint we have not walled. Collision (config hits, later FK) is a fourth job: `docs/plans/collision.md`.

Commanded motion is a **check** on a finished wall (hand: `hand open` / `close` / `to`), not a way to find the wall. Find the wall by hand, `tq=0`. Then I move inside the soft band. How we move (inner servo PID, speed/acc, `load` stop): `docs/plans/control.md`. Console `/info/` is the same story for sitting.

## 2026-09-09 sitting (USB + console)

- Chip = text API (USB/BLE). Console = brain. No fake HAT sensors.
- Six joints in the UI: shoulder **yaw** 11, shoulder **pitch** 12+13 (coupled), elbow 14, wrist pitch 15, wrist roll 16, hand 17.
- Compact `get` / `stream` in source: omit default zeros; stream skips still IDs. **Not flashed yet** — old fat lines until `idf.py flash`. Host parsers already accept both.
- Console footer logs deltas (`14  pos 2811→2835`), not the full `get` line. Backdrive: watch **pos**; `moving`/`load` stay 0 with `tq=0`.

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

1. **Flash** this firmware: twist `ESTOP` **and** compact `stream`. Stop Django first (`ACM0`). Confirm `stream` is quiet until you move a joint.
2. Loom slack at **4068**, `get` works, then retry `--check twist` if we still want it.
3. **Wrist pitch** (15) walls by hand, `tq=0`, mark on `/calibrate/?j=wrist-pitch`. Then a slow `to` inside the band.
4. Elbow, shoulder pitch (12+13 **together**), yaw. Same: wall by hand, then check. SyncWrite for 12+13 still owed (`docs/plans/control.md`).
5. Fill `/calibrate/` homes + min/max for all six. Then deg/count table → FK. BLE after USB is boring.

Pack-off hand-move (silence) can wait. It does not protect cables.

## How we save as we go

- Walls and signs → that joint’s baseline (`hand-open-close.md`, later `twist.md`, …) and the table in `joint-limits.md`.
- Raw stream → `docs/baselines/data/*.csv` or `*.txt`. CSV for anything with `load` / commanded steps.
- Software changes → `docs/software.md` plus the script header. Console pages → `gibbon-console/docs/`.
- Git: `roarm-fw` and `gibbon-console` are **two repos**. One commit when a sitting has a result.
