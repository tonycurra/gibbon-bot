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

Firmware **on metal 2026-09-12** (`info fw=20260912 raise=1`): `raise to` / `raise hold` / `raise free` SyncWrite, solo 12/13 abort, `id N here`, compact `stream`, move `ESTOP`.

**In tree, not yet flashed** (`info fw=20260915 mat=1`): `mat` 7×8 dump, `lim ID min|max|home` marks live pos, `sweep ID` min→max→min (raise = `sweep 12`). Serial sitting: `scripts/mon.sh`. Plan: `docs/plans/firmware-minimal.md`.

## 2026-09-12 later (console)

Navbar pack % is the **3S rest table**: 12.6 V = 100%, 10.8 V = 20%, 10.5 V = 10%, 9.0 V = 0%. 11.1 V → **40%**. Chip `batt est=` matches after the next flash; the header is JS from volts.

`/raw/` is a **hand** page: open / close / hold / free, effort trip in load counts (left = light touch), live sensor graphs. `who` / `raw 17` are under servo dump. Footer is one line until **log**; on that page it only shows ID 17. Connect no longer `who`s all seven.

## 2026-09-15 — four limbs, two packs

Chassis is assembled. **Two healthy 3S** for the lab. Leftovers **5.6 V** and **8.2 V** — recycle; do not plug. 3S on XT60 is still the pack: `docs/hardware/lipo.md`. Log: `development-logs/2026-09-15-packs.md`.

### Pack plan

1. **Now:** test with the **two** good 3S. One pack = one limb. Body **supported**. Do not hang the chassis on dead limbs. Do not Y-split. Do not buy the other two yet.
2. **Good enough** (then buy two more 3S XT60): ESTOP trips on purpose; each of two limbs does free → pose → hold → one nudge; VIN ~11 V; no sag, no puff, STOP works. Same image on both HATs.
3. **Then:** four packs on the chassis, **southern Italy** — all four limbs talking, operator present. Not harvest MVP. Not a gait.

Do **not** write standing firmware. Do **not** start a four-bridge rewrite. Console is still one USB / one BLE until Italy needs four pipes.

**This sitting (teleop, two packs):**

1. Pack on one HAT, UART **ESP32**. Header **usb**, pack %, **fw 20260912** (metal). In-tree sitting stamp is **20260915** — only after you flash `main/`. Django `:8080` **or** `./scripts/mon.sh` (one ACM0).
2. `/raw/` or `hand open` / `close`. Squeeze until chip prints `ESTOP`. If that fails, stay in the lab.
3. Per joint: **free** → pose by hand → **hold**. Then one **nudge** (11–15 hop ≤80). Wrist short of **3134**. Never 12 or 13 alone — `raise`. No twist **wheel**.
4. After this tree is flashed: `lim N home` / `min` / `max` at live pos, then `sweep N` (raise = `sweep 12`). `estop` cuts. Hand 17 can sweep on the built-in soft band.
5. Phone: header **LOCK**, per-joint free, **STOP**. Support the limb; STOP drops it.
6. Second healthy pack → second limb the same way, body **supported**. Do not hang the chassis on two dead limbs.

Bring-up `bringup/02-ble-monitor`: flash USB, **unplug USB**, pack on. Chromium `/monitor/` paints packed `g1`. Listen-only. Sitting chip `mat` is ASCII — not `g1`. Flash **sitting** `main/main.c` when you need motion, not 02.

**After two-pack tests are good:** buy **two** more 3S XT60 (plus keep storing the working pair at ~11.4 V). Four-limb metal in **southern Italy**. Wrap Δ on 16 then 11 from *this* slack (not Sep-5 4068). Elbow host 995…2997. One `raise to` nudge. Chip `lock` / `free` / `rec` / `play` as five words. Limb picker when four HATs talk. Tree is operator-present taught replay, not harvest MVP.

Metal is still one-limb teleop on a physical prototype. A second limb is the same image again, after two packs look boring.

Four limbs, rates, and why this stack stays on the chip: `docs/plans/gibbon-platform.md` and `docs/plans/ros2-vs-our-stack.md`.

## How we save as we go

- Walls and signs → that joint’s baseline (`hand-open-close.md`, later `twist.md`, …) and the table in `joint-limits.md`.
- Raw stream → `docs/baselines/data/*.csv` or `*.txt`. CSV for anything with `load` / commanded steps.
- Software changes → `docs/software.md` plus the script header. Console pages → `gibbon-console/docs/`.
- Git: `roarm-fw` and `gibbon-console` are **two repos**. One commit when a sitting has a result.
- 2026-09-11: `development-logs/2026-09-11-calibrate.md` + console `docs/calibrate.md`.
