# Motor control, then robot control

What we can do with **these** Feetech ST32xx servos and the raw `get` block. Not a textbook, not sim, not four-limb gait. One arm, IDs 11–17, HAT + ESP32.

Sensors we actually have: `docs/hardware/motors/sensors.md`. Walls first: `docs/plans/joint-limits.md`. Collision later: `docs/plans/collision.md`. Where we are: `docs/plans/now.md`.

**Do not command a joint that has no wall.** Control is a check on a finished band, not a way to find the band.

## What we have (and do not)

Each joint is one bus servo with a **magnetic encoder inside the motor**. The HAT has no extra encoder, IMU, current chip, or force/torque cell.

| We can read | We can write (already or soon) | We do not have |
| --- | --- | --- |
| `pos` 0–4095 (wraps; host `un`) | Torque enable `tq` 0/1 | Joint torque cell |
| `speed` steps/s | Goal position (`WritePos`) | IMU / gravity vector |
| `load` controller effort, ~±1000 | Goal **speed** + **acc** on that write | Gripper force sensor |
| `cur` mA (coarse; often 0 in our traces) | Wheel / motor mode (speed loop) — twist only, experimental | Pack current (no INA219) |
| `v`, `temp`, `moving`, `tq` | Mode 0 servo vs 1 wheel (EEPROM) | True torque / current **command** |

`load` and `cur` are **0 with `tq=0`**. Backdrive does not show a tight cable. Contact only exists while holding.

Hand gold (`docs/baselines/hand-open-close.md`):

| Situation | Typical `load` | `cur` |
| --- | --- | --- |
| `tq=0`, you turn it | 0 | 0 |
| Free commanded open (`tq=1`) | ~50–64 | 0–2 |
| Holding against a small pos error / squeeze | **~1000–1056** | still ~0 |
| First clean jaw touch after a free hold | **~1032** at pos 2073 | 0 |

So `load` is the contact proxy. `cur` is not, yet. Watcher already prints `LOAD` at `|load| > 400`. Chip does **not** abort a move on that spike — that is the first control job.

Servo modes (ST32xx):

| Mode | What you command | Inner loop (in the motor) | Use on this arm |
| --- | --- | --- | --- |
| **Servo** (default) | Goal `pos` + speed + acc | Position PID | Almost everything |
| **Wheel / motor** | Goal speed (signed) | Speed closed-loop | Twist unwrap / jog only. EEPROM. Always put back to servo |
| **Step / multi-loop** | Relative steps, ±7 turns | Position, not saved across power-off | Avoid until unwrap is boring |

There is **no torque-control mode**. `tq=1` means “hold / follow goal.” It is not a newton-metre setpoint. Stock UI “Torque ON/OFF” is this bit. Stock **DEFA** is high-impedance hold (spring back after a shove). We want the opposite on a hit: **stop pushing**.

Chip today (`docs/software.md`): `WritePos` with speed **300**, acc **20** on hand and twist. Clamp + 10.5 V cutoff. Twist hop abort `>900`. No live `load` abort during the move.

## Layers (do not mix them)

A PID on the laptop fighting the PID in the servo is worse than either alone.

```
  you / later planner
        |  "open hand", then "go here", later XYZ
  host (laptop)
        |  unwrap, walls, sync of 12+13, later FK
  ESP32
        |  clamp, voltage, load-stop, WritePos / SyncWrite
  servo MCU
        |  inner P / PI / PID  (we do not retune first)
  motor + encoder
```

| Layer | Job | Cadence |
| --- | --- | --- |
| **Inner (servo)** | Track the goal we sent. Already a position (or speed) loop. | kHz, inside the case |
| **Chip supervisor** | Never command a wall. Cut on pack. **Stop on unexpected `load`.** One joint or SyncWrite. | every `get` while `moving` |
| **Host motion** | Time the goal (trapezoid / hops). Pair 12+13. Unwrap twist. | ~10–50 Hz USB |
| **Arm / later body** | FK, named bad poses, then IK. Four limbs = another repo | slower |

P / PI / PID are **already running in the servo** in servo mode. Our job is: pick a legal goal, feed it smoothly, watch `load`, stop if the world disagrees.

## P, PI, PID — what they mean here

Classic loop: `error = goal − measured`, then effort = some mix of error.

| | Formula in words | What it does | On this bus |
| --- | --- | --- | --- |
| **P** | effort ∝ error | Stiff. Big error → hard push. Can overshoot / buzz | Default inner feel |
| **PI** | P + running sum of error | Kills leftover sag (gravity, stiction) | Inner; we see it as `load` while holding |
| **PID** | PI + rate of error | D damps the buzz | Inner; acc register is the cheap outer damper |

We **do not** write a host PID that outputs `WritePos` from `pos` error. The servo already closed that loop. Host PID only makes sense later if we switch a joint to **wheel** (command speed from pos error) or if we ever get a real torque command. Neither is the default path.

What we **do** set, without touching inner gains:

| Knob | Register / write | Effect |
| --- | --- | --- |
| Goal position | `WritePos` | Where |
| Max speed | speed field, steps/s | How fast. Hand 300 is slow on purpose. Max ~3073 |
| Acceleration | acc 0–150 | Soft start/stop. Hand **20**. Higher = snappier, more slam |
| Torque enable | `tq` | Hold vs free. Free = no `load` |
| (later) max torque | unused so far | Cap effort. Softness without leaving servo mode |

Tuning inner P/D/I in EEPROM is a last resort (oscillation after everything else is right). Not this sitting.

## Smooth and efficient motion

Smooth = no step in speed. Efficient = do not fight the world or each other.

**One joint (what hand already does):**

1. Read `pos`. Abort if pack `< 10.5 V`.
2. Clamp goal to the **soft** band (inside the wall).
3. Refuse a wrap hop (twist: `|goal−cur| > 900`).
4. `tq=1`, `WritePos(goal, speed=300, acc=20)`.
5. Poll `get` until `moving=0` **or** a stop rule fires.
6. `free` if we are not holding (gripper after a test; arm later may need hold against gravity).

**Better than one big `WritePos`:** a **trapezoid** on the host — many small goals along the same line, same speed/acc. The servo still does inner PID; we just do not ask for a 130° jump in one packet. Hand open 2085→689 already took seconds at speed 300; that is the right feel.

**Several joints:** `SyncWritePosEx` (same packet, seven IDs). Needed for **raise** (12 and 13, often opposite sign, one horn). If they disagree, they heat and `load` both climb — that is waste and a strip risk, not a feature.

**Efficiency (battery, gears, cables):**

- Do not hold a **squeeze**. Hand close at 2008 sat at `load≈1056` for a 5-count error. That is wasted current and heat. Arrive, then either `free` or a looser goal.
- Expected gravity `load` on a raised arm is not a collision. Measure a **moving-free** band per joint (hand free-open was ~50–64) before using a stop threshold.
- Dual-drive: one motion, two writes, same time. Never jog 12 alone.
- Acc 20 beats acc 0 (step) for peak current.
- Voltage cutoff already exists. `temp` is the slow fuse.

Stock “very low speed shakes near singularities” — wait until FK. Until then, slow + support the arm.

## Unexpected force → stop

A wall in software is a **known** stop. A hand, a table, a taut loom we missed, a stall — that is **unknown**. `load` is how we see it. `docs/plans/collision.md`: a spike past ~400 with `tq=1` is a **stop**, not a map.

### Detect (need at least two)

| Signal | Rule (start here, then measure) | False friend |
| --- | --- | --- |
| **`load` spike** | `\|load\| > 400` already prints. Hand contact was ~1000. Use **two** numbers later: moving-free ceiling vs hit | Gravity hold on a big joint may sit high. Calibrate per ID |
| **Position lag** | `\|goal − pos\|` stuck while `moving=1` (hand: 2008 vs 2003) | Slow speed looks like lag. Require *time* + *no Δpos* |
| **`cur` rise** | Stall is 2.4–2.7 A on the wiki. Our traces showed ~0. Keep reading; do not trust yet | Scale / sign unknown until a real stall |
| **`temp`** | Climb while sitting | Slow. Backup only |
| **`v` sag** | Pack dip on a stall | Also happens on any hard move |

Do not stop on `load` with `tq=0`. There is no signal.

### Act (chip, while the move is live)

Order when a hit fires:

1. **Stop pushing.** Write **current `pos`** as the new goal (same speed/acc, or speed 0). Do not keep the old goal. Do not command “back 80 counts” yet — that can wrap or yank.
2. Log `id`, `pos`, `un`, `load`, `cur`, `goal`.
3. Human: `free` that ID (or all) if it is a bind / cable. Leave `tq=1` only if dropping would smash (raised arm — not there yet).
4. Do **not** auto-retry the same goal.

This is **not** stock DEFA (spring back to the pre-shove pose). Spring-back fights the obstacle. We freeze.

Chip must poll during `hand to` / later `joint to`, not only when the host types `get`. Host `watch-csv.py` can do the same as a second line until the chip owns it.

### After a stop

- If `pos` is inside the soft band: treat as contact. Re-plan or ask.
- If `pos` is at a wall: the wall is the story; tighten the clamp.
- Twist: freeze in **`un`**, not raw `pos` (same count, different loom).

## Compliance vs rigid

| Feel | How | When |
| --- | --- | --- |
| **Rigid** | Servo mode, hold goal, high inner P | Park, repeatable `to` |
| **Soft stop** | Rigid move + `load` abort | Default for every commanded move |
| **Squeeze to contact** | Slow `to` until `load` rises, then freeze (hand 2073) | Gripper close, later grasp |
| **Backdrive / teach** | `tq=0` | Limits sitting. No `load` |
| **Cap torque** | Max-torque register (not wired) | Later softness without wheel mode |
| **Wheel + outer P** | Command speed from pos error | Only if servo mode cannot go slow enough. Twist unwrap maybe |

Gripper: close until `load` says jaws met, then **hold a little** or `free`. Do not wind to 2088.

Proximal joints (raise, bend): gravity is a constant `load`. Stop rule must be **Δload** or **load above the moving-free ceiling**, not “any load > 400” blindly copied from the hand.

## Per joint (this arm)

| Say | ID | Control note |
| --- | --- | --- |
| turn | 11 | Wrap possible. Same hop rule as twist once walled |
| raise / lower | **12+13** | One motion. SyncWrite. Opposite `pos` often. Never one ID |
| bend | 14 | Gravity `load` when holding a fold |
| wrist | 15 | First **config** hit. 1D clamp is not enough; host refuses some 7-tuples later |
| twist | 16 | Servo hops ≤900. Wheel only to walk a wrap, then **servo**. Host `un` |
| hand | 17 | Done as the template: clamp, slow, `load` contact |

Words: `docs/hardware/motors/names.md`.

## Ladder (build in this order)

Walls (`joint-limits.md`) stay ahead of writers. Control fills in **behind** a finished row.

1. **Live load-stop on the chip** — **twist wheel: written, not flashed yet.** `twist_guard_tick` ~25 ms while `twist speed ≠ 0`. Cut on `Δload≥100`, `|load|≥1350`, `|cur|≥80`, or 3 bus misses → speed 0 + `tq=0` + servo (`ESTOP`). `estop` / `get 16`. Hand `WritePos` still has no in-move poll. **Do not command twist until this is on the chip.**
2. **Moving-free `load` band** on the next commanded joint (wrist once walled). Hand ~50–64 vs hit ~1000. Write the ceiling in that joint’s baseline.
3. **Same `joint_to` for each new wall** — one ID, speed 300, acc 20, clamp, voltage, hop, load-stop. No generic writer until the table has min/max.
4. **12+13 SyncWrite** when raise is walled. Prove they do not fight (`load` both low while moving).
5. **Host trapezoid / hops** so a long move is many legal goals, not one. USB is enough.
6. **Hold vs free policy:** gripper often `free` after a test; a raised arm must hold. Explicit per command, never implicit on boot. **No move on boot** stays.
7. **Calibration table** (home, sign, deg/count) → FK stick → named bad poses (`collision.md`). Reactive `load` stays forever on any `tq=1` move.
8. **IK / Cartesian** last, on the laptop. Stock RoArm IK assumes factory homes and stock loom.
9. **Inner gain / max-torque / wheel-as-servo** only if 1–5 still buzz or cannot go gentle.

Not this folder: four-limb coordination, RL on metal, sim PID, BLE motion (USB first: `docs/plans/usb-then-ble.md`).

## Firmware rules that stay true

- No move on boot. `tq=0` until a named command.
- Soft clamp inside the wall. Never command the wall.
- Pack `v_raw < 105` → no torque on.
- `|load|` hit → stop that move (implement on chip; host is backup).
- Dual-drive is one joint.
- Twist: `un` / turns; WritePos is linear 0–4095, not shortest-arc.
- Wheel mode: experimental, EEPROM, restore servo on `free`.

## Do not do yet

- Host PID in parallel with servo PID.
- Wheel mode on 11–15, 17.
- Treating `load` as newton-metres or as a collision map.
- DEFA spring-back as the safety default.
- IK, Cartesian jog, or “keep the tip still.”
- Tuning inner P/I/D because a cable was taut.
- A generic joint writer before the wall table is filled.
