# Inside the servos, then what to build next

Addresses, signals, hardware in the case, PID, language, BLE cadence, energy. For sitting review. Bus packets: `../hardware/motors/esp32-bus.md`. Sensors: `../hardware/motors/sensors.md`. Control ladder: `control.md`. Chip vs host: `firmware-minimal.md`. Now: `now.md`.

Two different “addresses.” Do not mix them.

| Kind | What it is | This arm |
| --- | --- | --- |
| **Bus ID** | Which *motor* on the daisy chain | **11–17**, unique, stored in that servo’s EEPROM |
| **Register address** | Which *byte* inside that motor’s memory table | Same map on every ID. Present block starts at **56** |

Choosing a motor = put its **ID** in the packet. Choosing a signal = READ/WRITE a **register** (56 = position, 62 = voltage, …). One READ of 15 bytes at 56 returns every present signal at once. We do not pick a different 56 per motor.

---

## 1. Bus ID (which motor)

RoArm factory set these. Permanent until you unlock EEPROM and write a new ID. Default from Waveshare is 1 — already changed.

| Bus ID | Model | Joint | Why that model |
| --- | --- | --- | --- |
| 11 | ST3235 | turn / yaw | 30 kg·cm, 45 RPM |
| 12 | ST3235 | raise A | dual-drive with 13 |
| 13 | ST3235 | raise B | same horn, opposite counts |
| 14 | ST3235 | bend / elbow | |
| 15 | ST3235 | wrist pitch | |
| 16 | ST3215 | twist / roll | same protocol; we use wheel mode here |
| 17 | ST3215-HS | hand | faster (106 RPM), slightly less stall current |

The ESP32 does not “assign” 56 to a motor. It sends `FF FF <ID> … READ 56 15`. ID 11 and ID 17 both have a position at 56. The reply’s ID byte tells you whose encoder you got.

---

## 2. Register 56 (which signal)

Each servo has a **memory table**: EEPROM (survives power-off: ID, baud, PID, mode) and RAM (live: pos, speed, load, VIN, temp, current).

**56 is RAM**, not EEPROM. Docs sometimes say “EEPROM 56” loosely. Firmware name: `REG_PRESENT 56`. One 15-byte window:

| Register | Bytes | Signal | What it actually is |
| --- | --- | --- | --- |
| **56–57** | 2 | `pos` | 12-bit **magnetic encoder** on the output horn. 0–4095 = one magnet turn (~0.088°). Wraps. Not cable turns. |
| **58–59** | 2 | `speed` | Servo MCU derivative of that encoder. steps/s. |
| **60–61** | 2 | `load` | **Inner PID effort** (PWM demand), ~±1000. Not a torque cell. **0 if torque off.** |
| **62** | 1 | `v` | VIN at *this* servo, 0.1 V LSB. Same rail as XT60. |
| **63** | 1 | `temp` | Driver / case °C. |
| 64 | 1 | — | read, unused |
| **65** | 1 | `st` | status flags |
| **66** | 1 | `moving` | 1 = tracking a *commanded* goal. Not “you turned it.” |
| 67–68 | 2 | — | read, unused |
| **69–70** | 2 | `cur` | winding current **estimate**, mA. Wiki stall 2.4–2.7 A. Our free moves often read **0**. |

Also (separate READs, not in the 15-byte block):

| Register | Signal |
| --- | --- |
| **40** | `tq` — torque enable 0/1 |
| **33** | `mode` — 0 servo / 1 wheel (EEPROM) |
| **41–47** | acc, goal pos, time, goal speed (what we WRITE to move) |
| **21–23** | Kp, Kd, Ki (inner PID, EEPROM) |
| **16** | max torque cap (EEPROM, **not wired in our firmware yet**) |
| **9–12** | amin/amax (often 0–4095 = no wall) |

`un` / `turns` / pack % are **not** in the servo. Chip or host compute them.

We “choose” signals by **which register we READ**. `get` always takes the whole present window plus `tq`. `batt` reads only 62. `who` reads EEPROM PID/mode/limits. There is no other sensor on the HAT.

---

## 3. What is inside the servo (not “just an encoder”)

It is a **small robot joint in a box**:

```
  3-pin TTL: VIN, GND, DATA
           │
           ▼
  servo MCU  ← UART slave, memory table, PID, mode
       │
       ├── 12-bit magnetic encoder  (magnet on horn, chip on PCB — no rubbing pot)
       ├── MOSFET / driver  (phase drive; we never PWM this from the ESP32)
       ├── DC motor + metal gears
       └── EEPROM  (ID, baud, Kp/Kd/Ki, mode)
```

Waveshare: magnetic encoder, not a carbon-film pot. Gap, no friction, 360°, 4096 counts.

The **encoder is dumb**. The **MCU is the closed-loop controller**. The ESP32 is a **supervisor + radio**: send a legal goal, poll present, cut on pack / load / miss. It does not commutate the motor.

You cannot run a custom algorithm *inside* the servo MCU. No user firmware on that chip. Capabilities we get:

| Can | Cannot |
| --- | --- |
| Servo mode: goal pos + speed + acc | Torque-control mode (Nm setpoint) |
| Wheel mode: signed speed (EEPROM) | User code on the servo MCU |
| Read pos/speed/load/V/temp/cur | Extra IMU / force cell in the case |
| Write Kp/Kd/Ki, max torque, ID, baud | Change the inner loop structure |
| Daisy-chain 253 IDs (power limited) | Broadcast telemetry without a READ |

Stall ~2.4–2.7 A, no-load tens to a few hundred mA. Hand traces: `cur` often 0 until a real fight; `load` is the useful contact proxy.

---

## 4. Inner PID — can we control it?

**Yes, the registers exist. We should almost never touch them first.**

`who` already READs `kp` `kd` `ki` at 21–23. WRITE needs EEPROM unlock (`REG_LOCK` 55), then lock again. Survives power-off. Bad gains → buzz, heat, stripped gears.

Layers (do not stack a second PID on the first):

```
phone / grove UI     1–5 Hz  pose, energy, alarms
host (laptop)        unwrap, walls, later FK
ESP32                clamp, VIN, load-stop, WritePos / SyncWrite, BLE
servo MCU            position (or speed) PID at kHz
motor + encoder
```

Knobs we **should** use, in order:

1. Goal, speed (default 300), acc (default 20)
2. `tq` on/off
3. Chip **load / Δload / cur / miss** cut (**on chip 2026-09-12**)
4. **Max-torque** register 16 — softness without leaving servo mode (not wired yet)
5. Inner Kp/Kd/Ki — last resort if it still oscillates after 1–4

A laptop or ESP32 PID that outputs `WritePos` from `pos` error **fights** the servo PID. Wheel + outer P is only a maybe for twist unwrap.

---

## 5. Language: this firmware is C, not C++

`main/main.c`, `ble_nus.c`. ESP-IDF C. Console is Python + JS. There is no C++ in the chip tree.

**Rust for this robot, now: no.** Safety of the arm is walls, `tq=0` on boot, hop limits, VIN cutoff, and a **chip guard that polls while pushing**. The 2026-09-05 twist wound a cable because the chip did not watch `load` during wheel — that is a missing *loop*, not a missing borrow checker.

Rust on ESP32 (esp-hal / embassy) is real, but:

- Rewrites the IDF UART + NimBLE stack we already have working
- Does not make the servo MCU safer (closed binary)
- Costs sittings we still owe: NVS walls, wrap Δ, BLE bind

Revisit **after** USB/BLE control is boring and the C file wants a second owner. If we ever split: keep bus + guard in C (or a tiny `unsafe` Rust HAL), put host math in Python/Rust on the laptop. Do not put FK on the WROOM.

Security in the grove: BLE has **no pairing** today (`ble.md`). Fine for a lab arm. Field: at least a bind / passkey before motion writes; `estop` and `to` must not be open to any phone in range. That is protocol + GAP, not Rust.

---

## 6. Cadence: grove vs millisecond

Millisecond closed-loop **already lives in the servo**. BLE cannot and should not replace it.

| Path | Sensible rate | Why |
| --- | --- | --- |
| Servo inner PID | kHz, in the case | Track goal |
| ESP32 guard poll (moving ID only) | **20–40 ms** | Cut on load/cur/miss. Must work if the phone dies |
| USB debug `stream` | 50–180 ms | Sitting |
| BLE to phone in an olive grove | **1–5 Hz** telemetry; commands on demand | Trees, distance, 3S pack. Not a haptic bus |
| Alarms (`ESTOP`, pack low) | **immediate notify** | One small packet, not a stream |

NimBLE today: Nordic UART text, advertise 160–320 ms (flash owed), conn interval 24–40 (≈30–50 ms), latency 2. Phone does not need that for a table of volts.

Grove-shaped radio:

- **Do not** `get` all seven IDs as fat ASCII at 50 Hz over BLE. That is what starved the bus on twist.
- Chip owns motion + guard. Phone sends “open”, “hold”, “estop”, named poses.
- Phone receives a **small binary (or skinny) frame**: pack V, ΣmA or energy, each ID pos + flags, maybe one `load` for the moving joint.
- If notify buffer is full: **drop the frame**, never block the guard (`firmware-minimal.md`).
- No WiFi while BLE is on this WROOM-32.

1 Hz is enough to watch energy and pose while walking. 5 Hz if a joint is moving and you want the slider to feel alive. Sub-20 ms BLE is wasted current and a worse radio in the grove.

---

## 7. Energy on the chip (yes — this is the right next “smart” bit)

HAT has **no shunt**. Per-servo `v` (62) and `cur` (69) are what we have. `cur` is coarse; still worth integrating *on the ESP32*, not on the phone.

Lightweight, fits this MCU:

| Estimator | How | Use |
| --- | --- | --- |
| Pack V | median of seven addr-62 bytes, already `batt` | cutoff 10.5 V; “recharge” band ~11.2–11.4 V rest |
| Σ mA | sum of signed `|cur|` on IDs that answered | header already shows this in the console; often 0 until fight |
| Power | `pack_v × Σ|cur|` | instantaneous, noisy |
| **Energy** | `E += V * I * dt` on the **guard/stream tick** (20–50 ms), not BLE | Wh this sitting; grove “how much did that climb cost” |
| **Averages** | exponential moving average of V and \|I\| (one multiply per tick) | smooth phone plot; ignore BLE jitter |
| Hold waste | if `tq=1` and `|load|` high and `speed≈0` for N ticks → log / optional `free` | hand squeeze at `load≈1056` was burning pack for 5 counts of error |

Do **not** put 3S % guess on the hot path (`firmware-minimal.md` already wants it off the chip). Phone can keep the rest-voltage table. Chip reports tenths of a volt + accumulated mWh.

No neural net on the WROOM for this. The “algorithm” is EMA + integral + load fuse.

---

## 8. Near-future order (piece of control, not a rewrite)

Art here is **the right loop on the right chip**, quiet radio, no wasted hold current.

**On the metal 2026-09-12:** compact `stream`, move **ESTOP**, `raise` SyncWrite, `info fw=20260912 raise=1`.

**Still owed:**

1. BLE ghost terminate + slow advertise (`ble.md`).
2. Finish wrap Δ walls (11, 16) from a known slack **here**. Soft-band 15, then 14.

**Then, still small firmware:**

3. Binary (or skinny) present notify; skip still IDs; **never skip the ID we are driving**.
4. NVS walls on chip for non-wrap IDs; host still owns wrap Δ.
5. EMA + energy accumulator on the stream/guard tick; one `energy mWh=` / binary field to the phone.
6. Wire **max-torque** (reg 16) as a softness cap. Still no inner-gain party.
7. BLE: bind before writes; telemetry 1–5 Hz; `estop` high priority.

**Later, host / grove:**

8. Trapezoid hops on the host (or a tiny chip sequencer): many legal goals, inner PID stays.
9. Hold vs free policy per joint (gripper often free after grasp; raised arm holds).
10. FK on the laptop after walls you trust. IK last.
11. Phone UI over the same NUS bytes — not a second protocol.
12. Language: stay C on the chip until this list is dull. Rust is optional on a *new* node, not a rescue of `main.c`.

**Do not:** host PID vs servo PID, ms BLE pose streaming, WiFi+BLE together, custom code inside the servo, averaging energy on the phone from dropped notifies, rewrite for “memory safety” before NVS walls and BLE bind are boring.

Olive grove: the arm must **survive a dropped phone**. That is chip guard + local `estop` + slow BLE, not a faster link.
