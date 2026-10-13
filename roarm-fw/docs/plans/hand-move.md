# Hand-move: encoder live or not

Next experiment. Power-cycle gold is done (`usb-start.md`, `usb-battery-on.md`). Still **no `WritePos` / `torque 1`**.

Question: when you backdrive a joint by hand, what does the bus actually report?

ST3215s have a magnetic encoder **inside the motor**. It needs servo VIN. The HAT has no extra joint sensor.

## A. USB on, XT60 **off**

Expect **silence**. Unpowered servos do not answer `ping` / `get`. Turning the output horn by hand should **not** create `bus_rx` hex or a position. If anything appears, that is a finding (noise, wrong switch, or a powered path we do not know).

1. Match `usb-start.md` (`state`, heartbeats `bus_rx=0`).
2. Move joints by hand (slow). Watch heartbeats. Do **not** ping yet.
3. Then `ping` / `get` / `batt` while still moving one joint.
4. Save `docs/baselines/hand-move-usb-no-battery.md`.

Gold guess: `found=0/7`, `get` all miss, `bus_rx` stays 0.

## B. USB on, XT60 **on**, torque stays **off**

Expect **position to change** on `get`. Voltage stays ~pack. `tq` must stay `0`. `moving` / `load` / `cur` may stay 0 while backdriving with torque off — or they may tick. Write down whatever we get; do not invent a load signal.

1. Match `usb-battery-on.md` (`found=7/7`, `batt`, `get` once at rest).
2. Support the arm. Dual-drive shoulder (12+13) is two motors on one joint — move slowly, do not force.
3. Repeat `get` (or a later `watch`) while moving **one** joint at a time. Name the joint.
4. Save rest vs moved lines in `docs/baselines/hand-move-usb-battery-on.md`. **Done 2026-09-05** (`scripts/watch-hand.py`). `pos` + `speed` live; `load` / `moving` / `cur` stay 0.

Look at: `pos` (must move), `speed`, `load`, `cur`, `moving`, `tq`, `v`. Pack % is still a rest-voltage guess.

If `pos` does not change: UART switch not ESP32, or we are reading a different ID than the joint we moved.

## Safety

- `tq=0` the whole time. If a `get` shows `tq=1`, stop and unplug XT60.
- USB does not hold the motors. Pack VIN does.
- Conservative pack click is still USB-out then XT60; this test needs USB in so we can `get`.

## After this

One-joint soft stops, human words: `docs/plans/joint-limits.md`, `docs/hardware/motors/names.md`. Then BLE. VIN-only ESP32 still open. Commanded nudge only after limits. `docs/plans/usb-then-ble.md`.
