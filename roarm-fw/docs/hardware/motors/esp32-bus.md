# ESP32 ↔ ST bus (how these motors talk)

Check this against `main/main.c` (`bus_init`, `mk_pkt`, `bus_tx_rx`, `bus_read`, `bus_write`, `bus_sync_write`, `print_get_id`) and the HAT wiki dump `../SBSD-HAT.md`. Gold serial: `../../baselines/usb-start.md`, `usb-no-battery.md`, `usb-battery-on.md`.

Waveshare name on the box: **ST3215 / ST3235 / ST3215-HS**, **Bus Servo Driver HAT (A)**. Protocol family: **ST/SC serial bus** (vendor lib `SMS_STS` / `SCServo`). Packets are the same shape as Dynamixel Protocol 1.0. Default baud **1 Mbps**. Signal is **half-duplex TTL**, not PWM, not CAN, not I2C.

```
laptop  --USB 115200-->  ESP32 UART0     (text: get, ping, id N …)
                         ESP32 UART1     GPIO19 TX / GPIO18 RX @ 1 Mbps 8N1
                              |  HAT UART switch must be ESP32
                              v
                    TTL 3-pin daisy chain  (VIN, GND, DATA)
                         id 11 … 17
```

BLE Nordic UART is a **second copy of the same text pipe**. It does not talk to the servos. Only UART1 does.

---

## 1. Two power domains (this is why pack-off is silence)

| Domain | Source | What lives |
| --- | --- | --- |
| 3.3 V chip | USB VBUS (debug Type-C) or HAT 5 V buck from XT60 | ESP32, USB-CDC, BLE radio |
| Servo VIN | **XT60 / DC / screw terminal only** (9–25 V on the HAT; our 3S pack ~11.4 V) | Each ST node’s MCU, encoder, driver |

USB does **not** power the motors. A servo with no VIN cannot answer ping, cannot report `pos` / `v` / `temp`. Firmware then prints `miss`. Gold: `found=0/7` `guess=usb-no-battery`.

Pack click with USB already up does **not** reset the ESP32 (2026-09-04). Unplug pack: chip stays, bus goes silent again.

---

## 2. Electrical signal (what is on the wire)

Waveshare FAQ (ST3215): **half duplex**. One data wire, both directions, not at the same time.

| Property | This arm |
| --- | --- |
| PHY we use | **TTL** headers on the HAT (3-pin 5264 / 2.54). Daisy-chain: each servo has two 3-pin ports, in and out. |
| RS485 header | Same UART1, different PHY. We are not on RS485. |
| Idle | **high** (UART mark). `gpio` gold: `18(rx)=1 19(tx)=1` |
| Levels | ESP32 3.3 V UART. The HAT’s bus driver is the thing that is allowed to sit on the servo DATA line. Do not assume you can wire GPIO 19 straight to a 12 V servo pin. |
| Connector | VIN, GND, DATA. Power and signal share the loom. |
| Connector type | ST3215: **5264 / 2.54 3P** |

Half-duplex consequences:

1. Master (ESP32) talks, then **must stop driving TX** and listen. Firmware: `uart_write_bytes` → `uart_wait_tx_done` (5 ms cap) → `uart_read_bytes` with a 10–12 ms timeout (`bus_tx_rx`).
2. The HAT often **echoes TX into RX**. Firmware strips that: if the first `txn` RX bytes equal the TX packet, they are discarded. Remaining bytes are the servo reply.
3. Servos **never broadcast**. Sitting still with pack on still shows `bus_rx=0` until we PING or READ. Turning a horn by hand does not put bytes on the wire. You only see `pos` if something sends `get`.

Many nodes, one wire: each servo has a unique **ID 1–253**. Two nodes with the same ID collide (factory default is 1; this arm is already 11–17). Only the addressed ID replies, except sync-write (broadcast ID `0xFE`, no reply).

---

## 3. HAT as a UART switch (who owns the bus)

Physical slide switch on Bus Servo Driver HAT (A):

| Switch | Master of servo DATA |
| --- | --- |
| **ESP32** | UART1 GPIO 18/19. **This is us.** |
| USB | Bus-servo Type-C → servos directly. ESP32 isolated. Laptop could run Waveshare `ping.py`. |
| Pi | 40-pin UART. No Pi on this limb. |

If the switch is not ESP32, our firmware still prints heartbeats on USB, but **every ping misses** — same symptom as pack off. Do not flip the switch mid-test.

Two Type-C ports:

| Port | What |
| --- | --- |
| ESP32 Type-C | USB-CDC → UART0 @ 115200. `/dev/ttyACM0`. Firmware console. |
| Serial bus servo Type-C | Direct to the bus **only if the switch is USB**. Isolated when switch = ESP32. |

---

## 4. ESP32 UARTs (firmware)

```c
USB_UART  UART_NUM_0    115200 8N1     laptop text
BUS_UART  UART_NUM_1    1000000 8N1    GPIO19 TX, GPIO18 RX
```

`bus_init()`: 1 Mbps, 8 data, no parity, 1 stop, 1024-byte TX and RX rings. No RTS/CTS. No invert.

Laptop line is **newline text**. Bus line is **binary packets**. The chip is the translator.

Timing at 1 Mbps, 8N1 = **10 bit-times / byte = 10 µs / byte**.

| Packet | Bytes on the wire | ≈ time |
| --- | --- | --- |
| Ping TX | 6 | 60 µs |
| Ping RX | 6 | 60 µs |
| Present READ TX | 8 | 80 µs |
| Present READ RX | 5 header + 15 data + 1 CS ≈ 21 | 210 µs |

Firmware timeouts (10–12 ms) are ~50× slower than the wire. The wait is for the servo MCU to wake and answer, plus half-duplex turnaround, not for bit time. `get` of seven IDs is **14 bus transactions** (present + torque-en each). Host `watch-hand.py` saw ~1.2 s per full sample (drain + pause) — host pacing, not 1 Mbps.

---

## 5. Packet format

Every command the ESP32 puts on UART1:

```
FF FF  ID  LEN  INST  [params…]  CHECKSUM
```

| Byte | Meaning |
| --- | --- |
| `FF FF` | header |
| `ID` | 11–17, or `FE` broadcast (sync write only) |
| `LEN` | number of bytes after LEN, **including INST and CHECKSUM** (`nparams + 2`) |
| `INST` | instruction |
| params | instruction-specific |
| CHECKSUM | `~(sum of bytes from ID through last param)`  (8-bit, inverted) |

Status reply (PING / READ / WRITE):

```
FF FF  ID  LEN  ERR  [DATA…]  CHECKSUM
```

`ERR` 0 = ok. Firmware `bus_read` copies DATA starting at offset 5. It checks header `FF FF` and that `ID` matches. It does **not** currently verify checksum or `ERR`.

Instructions we use (`main.c`):

| INST | Hex | Name | Reply? |
| --- | --- | --- | --- |
| 1 | `01` | PING | yes, 6 bytes |
| 2 | `02` | READ | yes, ERR + N data bytes |
| 3 | `03` | WRITE | yes (status) |
| 0x83 | `83` | SYNC WRITE | **no** — fire and forget |

### Worked packets (ID 11 = `0x0B`)

**Ping**

```
TX  FF FF 0B 02 01 F1
         ID LEN INST  CS = ~(0x0B+0x02+0x01) = ~0x0E = F1

RX  FF FF 0B 02 00 F2     (gold-style; ERR=0)
```

`bus_ping` only checks `n >= 6` and `FF FF`. Characterize-hat example: `ff ff 0b 02 00 f2`.

**Read 15 bytes at address 56** (`REG_PRESENT`, `PRESENT_LEN=15`)

```
TX  FF FF 0B 04 02 38 0F  CS
              LEN INST addr=56  n=15
```

`LEN=4` because INST + addr + n + CS = 4 bytes after LEN? Wait: `nparams=2` (addr, nwant), `LEN = nparams+2 = 4`. Bytes after LEN: INST, 0x38, 0x0F, CS. Yes.

**Write torque enable** (addr 40, one byte)

```
TX  FF FF ID 04 03 28  00-or-01  CS
              INST addr=40  tq
```

**Sync write** (raise 12+13): ID=`FE`, INST=`83`, then `addr`, `nlen`, then for each servo `id, data…`. No status packet. Firmware uses this for `raise to` (ACC+goal+speed, 7 bytes @41).

---

## 6. Memory map the firmware actually touches

ST nodes have an internal table (EEPROM + RAM). Unlock EEPROM (`REG_LOCK` 55) before mode writes. RAM present block is always readable when VIN is up.

### Present / RAM (polled every `get`)

Start `REG_PRESENT = 56`, 15 bytes:

| Addr | Bytes | `get` key | Encoding |
| --- | --- | --- | --- |
| 56–57 | 2 | `pos` | little-endian, **signed 15-bit** (`sts_s15`: bit15 set → negative magnitude) |
| 58–59 | 2 | `speed` | same; steps/s |
| 60–61 | 2 | `load` | same; PWM effort, ~±1000. **0 if `tq=0`** |
| 62 | 1 | `v` | unsigned, **0.1 V**. `batt` reads this byte alone (`REG_VOLTAGE`) |
| 63 | 1 | `temp` | unsigned °C |
| 64 | 1 | — | read, not printed |
| 65 | 1 | `st` | status |
| 66 | 1 | `moving` | 0/1 = “tracking a goal”, not “horn turning” |
| 67–68 | 2 | — | read, not printed |
| 69–70 | 2 | `cur` | signed 15-bit, mA |

Second READ: `REG_TORQUE_EN` **40**, 1 byte → `tq`.

`un` / `turns` are **ESP32 RAM**, not servo registers. Wrap: `|Δpos| > 2048` → `turns ±1`. Lost on chip reset.

Wiki “feedback”: Position, Load, Speed, Input Voltage. Temperature and current are in the same present block; firmware reads them.

### Writes (motion)

`WritePosEx` in this firmware is **one WRITE of 7 bytes at `REG_ACC` 41**:

| Addr | Bytes | Field |
| --- | --- | --- |
| 41 | 1 | acc (`move acc`, default 20) |
| 42–43 | 2 | goal position |
| 44–45 | 2 | time (firmware writes 0, 0) |
| 46–47 | 2 | goal speed (`move speed`, default 300, max 3073) |

Then the **servo’s own PID** tracks that goal. The ESP32 does not PWM phases. Host PID on top of this fights the inner loop — we do not do that.

Torque: WRITE `1` or `0` at addr 40 **before** the goal write (`joint_to`).

### EEPROM (infrequent)

| Addr | `who` / use |
| --- | --- |
| 0 | firmware version |
| 9–12 | amin / amax (often 0–4095 = no wall) |
| 16 | max torque limit |
| 21–23 | Kp Kd Ki |
| 33 | mode: **0 servo**, **1 wheel** (twist only; EEPROM — restore servo) |
| 55 | lock (0 unlock / 1 lock) before mode write |

---

## 7. Master / slave rules

- **One master.** ESP32. Servos are slaves. They do not talk to each other.
- **Poll.** Want telemetry? Send READ. Period.
- **Addressed.** `get 16` is one node. `get` is a for-loop 11–17. A long `get` of seven can starve a moving twist — that is why the guard wants `get 16` while pushing.
- **IDs on this arm**

| ID | Model | Joint |
| --- | --- | --- |
| 11–15 | ST3235 | turn, raise A/B, bend, wrist pitch |
| 16 | ST3215 | twist |
| 17 | ST3215HS | hand |

Same packet, same baud, same present map. Model only changes torque/speed ratings and (16) wheel mode.

Baud on the servo is EEPROM, default **1 Mbps**, range 38400–1 Mbps. We never change it. If a node were 115200, it would look like `miss` at 1 Mbps.

---

## 8. Servo communication specs (Waveshare)

| | ST3235 (11–15) | ST3215 (16) | ST3215-HS (17) |
| --- | --- | --- | --- |
| Bus | TTL UART, half duplex | same | same |
| ID range | 1–253 | 0–253 (wiki) / 1–253 | 1–253 |
| Baud | 38400–1 Mbps, **1 Mbps default** | same | same |
| Encoder | 12-bit magnetic, 0–4095 = 360°, **0.088°/step** | same | same |
| VIN | 6–12 V typ. 12 V | 6–12.6 V | 6–12.6 V |
| Stall current | 2.7 A | 2.7 A | 2.4 A |
| No-load speed @12 V | 45 RPM | 45 RPM | 106 RPM |

Protocol is identical. Do not mix RS485 RSBL servos onto this TTL chain.

---

## 9. What a transaction looks like in time

`get id=11` in firmware:

1. Flush UART1 RX (drop leftover echo / junk).
2. TX READ present @56, 15 bytes.
3. Wait TX done.
4. RX up to 40 bytes, 12 ms.
5. Strip echo of the TX packet if present.
6. Parse `FF FF 0B …`, copy 15 data bytes.
7. Repeat for torque-en @40 (another full round-trip).
8. `printf` a text line on UART0 / BLE.

If step 5–6 fail: `get id=11 miss`. Three misses while a guarded move is running → `ESTOP` (speed 0, `tq=0`).

`bench` is seven PINGs, then `found=N/7` and `guess=`.

---

## 10. What this HAT is not

No INA219, no IMU, no OLED, no user LED, no BMS. `batt` is the seven VIN bytes at addr 62, median, then linear 3S % (**10.0 V = 0%, 12.6 V = 100%**). Navbar uses the same math (host, no flash needed). Chip `est=` matches after the next flash. Pack cutoff for writes: **10.5 V** (`V_CUTOFF 105`).

TTL activity LEDs by USB-C are PHY blinks, not GPIO.

---

## 11. Quick checks (on metal)

UART switch **ESP32**. One process on ACM0.

| You type | Bus packets | Healthy |
| --- | --- | --- |
| `gpio` | none | 18=1 19=1 |
| `bench` pack off | 7× PING, no RX | `found=0/7` |
| `bench` pack on | 7× PING + 7 replies | `found=7/7` |
| `raw 11` | PING + READ hex dump | `n=` ≥ 6, starts `ff ff 0b` |
| `get` | 7× (READ 15 + READ 1) | seven `pos=` `v=` `temp=` lines, or `miss` |
| `batt` | 7× READ addr 62 | `pack=11.xV` or `pack=miss` |

If pack is on and you still get all miss: switch not ESP32, wrong Type-C, or bus wiring.

Console gold tables of those dumps: `../../gibbon-console` page `/hat/`.
