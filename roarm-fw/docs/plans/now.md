# Where we are

You write here; this page is the pile sorted. Words: `docs/hardware/motors/names.md` (yaw / pitch / roll + room words). Console: `../../gibbon-console/docs/README.md`.

## Not mixed — three jobs

| Job | Question | Status |
| --- | --- | --- |
| **Safety / bring-up** | Chip up? Pack ok? `tq=0` unless we asked? No boot move? | **Done.** `usb-start.md`, `usb-battery-on.md` |
| **Limits** | Where does a cable cut or plastic hit? Soft band inside that. | **Hand gold. Twist Sep-5 walls known; 11 Sep slack was 960.** Wrist/elbow host marks; 15 ESTOP near 3134. **Flashed 2026-09-12:** ESTOP + raise SyncWrite + compact stream + `info fw=20260912 raise=1` |
| **Calibration** | What does a count *mean*? Home, degrees, later FK / sim. | **In progress.** Wrap joints: Δ / degrees from **here**. `/calibrate/<slug>/`. Signs for the hand already fell out of limits (open −, close +) |

Limits and calibration **share a sitting** (you move, we record `pos` and the word), but they are not the same write-up. Limits = do not break. Calibration = map to math. We do not calibrate a joint we have not walled. Collision (config hits, later FK) is a fourth job: `docs/plans/collision.md`.

Commanded motion is a **check** on a finished wall (hand: `hand open` / `close` / `to`), not a way to find the wall. Find the wall by hand, `tq=0`. Then I move inside the soft band. How we move (inner servo PID, speed/acc, `load` stop): `docs/plans/control.md`. **Next firmware shape:** `docs/plans/firmware-minimal.md` (chip = raw + walls + raise + estop; host = math). Addresses, what’s in the servo, BLE cadence, energy, language: `docs/plans/near-future.md`. Console `/info/` is the same story for sitting.

## 2026-09-09 sitting (USB + console)

- Chip = text API (USB/BLE). Console = brain. No fake HAT sensors.
- Six joints in the UI: shoulder **yaw** 11, shoulder **pitch** 12+13 (coupled), elbow 14, wrist pitch 15, wrist roll 16, hand 17.
- Compact `get` / `stream`: omit default zeros; stream skips still IDs. **On the chip 2026-09-12.** Host parsers already accept both.
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

Chip guard is **on the metal** (`ESTOP` on `Δload` / `cur` / bus miss). Sensors: `docs/hardware/motors/sensors.md`. 2026-09-11 slack **here** was **960**, not 4068. Look at the loom; do not mix the two origins.

## 2026-09-11 sitting (calibrate + wrap Δ)

Sitting log: `development-logs/2026-09-11-calibrate.md`. Console: `/calibrate/` index + `/calibrate/<slug>/`. Wrap joints (11, 16): degrees (`360°/4096`), **here** → **calibrate** → move → **done** (span ≥80). Host unwraps so Δ is monotonic through 0/4095. `send()` is a queue; `estop` jumps.

Twist slack this sitting **960** (Sep-5 gold was **4068** — do not mix). Leftover raw 3108/3111 cleared. Wrist pitch toward 3134 tripped `ESTOP id=15 dload`. Raise SyncWrite is on the chip as of 2026-09-12.

Firmware in tree, **flashed 2026-09-12** (`info fw=20260912 raise=1`): `raise to` / `raise hold` / `raise free` SyncWrite, solo 12/13 abort, `id N here`, compact `stream`, move `ESTOP`. Plan: `docs/plans/firmware-minimal.md`.

## 2026-09-12 later (console)

Navbar pack % is **linear 3S**: 12.6 V = 100%, 10.0 V = 0% (9.9 V LVC + 0.1 V pad). 11.1 V → **42%**, not the old rest-curve 12%. Chip `batt est=` still the old table until the next flash; the header is JS from volts.

`/raw/` is a **hand** page: open / close / hold / free, effort trip in load counts (left = light touch), live sensor graphs. `who` / `raw 17` are under servo dump. Footer is one line until **log**; on that page it only shows ID 17. Connect no longer `who`s all seven.

## Next sitting — try this

1. Hard-refresh. Confirm **fw 20260912** and pack **~42% at 11.1 V**.
2. `/raw/`: open and close. Watch pos / effort / mA. Trip STOP on purpose once (chip `ESTOP` should print).
3. Wrap Δ on **16** then **11** from a slack **here** this sitting. Do not reuse Sep-5 `un` 1058…7046 against home **960**.
4. Soft-band wrist pitch well inside 1126 … short of 3134. Then elbow (host 995…2997, hop ≤80).
5. Pair on `/calibrate/shoulder-pitch/`, **one** SyncWrite nudge. If they fight: STOP.
6. Deg/count → FK only after walls you trust. BLE after USB is boring. Optional flash so `batt est=` matches the navbar.

Pack-off hand-move (silence) can wait. It does not protect cables.

Four limbs, ROS2 vs ours, BLE/guard rates, bus is serial: `docs/plans/gibbon-platform.md`. Business + operational evaluation (keep our firmware/console): `docs/plans/ros2-vs-our-stack.md`.

## How we save as we go

- Walls and signs → that joint’s baseline (`hand-open-close.md`, later `twist.md`, …) and the table in `joint-limits.md`.
- Raw stream → `docs/baselines/data/*.csv` or `*.txt`. CSV for anything with `load` / commanded steps.
- Software changes → `docs/software.md` plus the script header. Console pages → `gibbon-console/docs/`.
- Git: `roarm-fw` and `gibbon-console` are **two repos**. One commit when a sitting has a result.
- 2026-09-11: `development-logs/2026-09-11-calibrate.md` + console `docs/calibrate.md`.
