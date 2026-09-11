# Where we are

You write here; this page is the pile sorted. Words: `docs/hardware/motors/names.md` (yaw / pitch / roll + room words). Console: `../../gibbon-console/docs/README.md`.

## Not mixed — three jobs

| Job | Question | Status |
| --- | --- | --- |
| **Safety / bring-up** | Chip up? Pack ok? `tq=0` unless we asked? No boot move? | **Done.** `usb-start.md`, `usb-battery-on.md` |
| **Limits** | Where does a cable cut or plastic hit? Soft band inside that. | **Hand gold. Twist Sep-5 walls known; 11 Sep slack was 960.** Wrist/elbow host marks; 15 ESTOP near 3134. Flash ESTOP + raise |
| **Calibration** | What does a count *mean*? Home, degrees, later FK / sim. | **In progress.** Wrap joints: Δ / degrees from **here**. `/calibrate/<slug>/`. Signs for the hand already fell out of limits (open −, close +) |

Limits and calibration **share a sitting** (you move, we record `pos` and the word), but they are not the same write-up. Limits = do not break. Calibration = map to math. We do not calibrate a joint we have not walled. Collision (config hits, later FK) is a fourth job: `docs/plans/collision.md`.

Commanded motion is a **check** on a finished wall (hand: `hand open` / `close` / `to`), not a way to find the wall. Find the wall by hand, `tq=0`. Then I move inside the soft band. How we move (inner servo PID, speed/acc, `load` stop): `docs/plans/control.md`. **Next firmware shape:** `docs/plans/firmware-minimal.md` (chip = raw + walls + raise + estop; host = math). Console `/info/` is the same story for sitting.

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

Chip guard is in source (`ESTOP` on `Δload` / `cur` / bus miss) — **flash owed**. Sensors: `docs/hardware/motors/sensors.md`. 2026-09-11 slack **here** was **960**, not 4068. Look at the loom; do not mix the two origins.

## 2026-09-11 sitting (calibrate + wrap Δ)

Sitting log: `development-logs/2026-09-11-calibrate.md`. Console: `/calibrate/` index + `/calibrate/<slug>/`. Wrap joints (11, 16): degrees (`360°/4096`), **here** → **calibrate** → move → **done** (span ≥80). Host unwraps so Δ is monotonic through 0/4095. `send()` is a queue; `estop` jumps. **Do not send `raise …` until flash.**

Twist slack this sitting **960** (Sep-5 gold was **4068** — do not mix). Leftover raw 3108/3111 cleared. Wrist pitch toward 3134 tripped `ESTOP id=15 dload`. Shoulder pitch still two packets on the chip.

Firmware in tree, **unflashed**: `raise to` / `raise hold` / `raise free` SyncWrite, solo 12/13 abort, `id N here`. Plan: `docs/plans/firmware-minimal.md`.

## Next sitting

1. **Flash** when asked (stop Django, ACM0 free): SyncWrite raise + move guard + compact `stream`. Then the console may send `raise to` / `raise hold` / `raise free`.
2. Finish wrap Δ walls on **16** (and **11**) from a known slack **here**. Do not reuse Sep-5 `un` 1058…7046 against home **960**.
3. Soft-band wrist pitch inside 1126 … well short of 3134. Then elbow check (host 995…2997, hop ≤80).
4. After flash: pair sign on `/calibrate/shoulder-pitch/`, then one SyncWrite nudge. If they fight: STOP.
5. Deg/count table → FK only after walls are a sitting you trust. BLE after USB is boring.

Pack-off hand-move (silence) can wait. It does not protect cables.

## How we save as we go

- Walls and signs → that joint’s baseline (`hand-open-close.md`, later `twist.md`, …) and the table in `joint-limits.md`.
- Raw stream → `docs/baselines/data/*.csv` or `*.txt`. CSV for anything with `load` / commanded steps.
- Software changes → `docs/software.md` plus the script header. Console pages → `gibbon-console/docs/`.
- Git: `roarm-fw` and `gibbon-console` are **two repos**. One commit when a sitting has a result.
- 2026-09-11: `development-logs/2026-09-11-calibrate.md` + console `docs/calibrate.md`.
