# Characterize HAT: USB listen, then attach battery

USB-only listen is **done** (2026-09-04). Gold: `docs/baselines/usb-start.md`. Ping gold still: `docs/baselines/usb-no-battery.md`. Next: plug XT60 while the same session is printing, still no move.

Goal: sit on serial with **minimal, no-move** firmware, dump **raw** what the chip and the bus are doing, first with XT60 unplugged, then plug the pack **while still listening**. See what actually wakes. Save whatever we get as the next baseline.

This HAT is a UART switch + bus driver + ESP32. It is **not** a sensor board. Nothing like INA219 / IMU / OLED will appear when the battery clicks in. Those lived on the old General Driver Board (`docs/hardware.md`).

## Honest map of “ports”

| Path | What it is | USB-only | After XT60 |
| --- | --- | --- | --- |
| ESP32 USB-C → `/dev/ttyACM0` 115200 | Our console (UART0). Chip is alive because USB powers it | `info` / task list / heartbeats | Same chip. Heap/tasks should barely change |
| UART1 GPIO 18 RX / 19 TX @ 1 Mbps | Servo bus (TTL headers + RS485 header are the same UART, different PHY) | RX silence (servos unpowered) | Still likely **RX silence until we ping**. Feetech ST servos do not broadcast on boot |
| Bus-servo USB-C | Direct USB↔bus bridge when the **UART switch is USB**, not ESP32 | Isolated if switch = ESP32 | Still isolated if switch stays ESP32. Do **not** flip the switch mid-test |
| Pi 40-pin TX/RX | Host UART when switch = Pi | Unused (no Pi on this limb) | 5 V buck may come up for a Pi we are not using |
| Pi 40-pin 5 V | From the HAT’s 9–25 V buck | Off (no pack) | Should come alive. **Do not** wire 5 V into an ESP32 GPIO |
| I2C / ADC / “HAT sensors” | Not populated on this PCB | Noise if we scan anyway | Same |
| WiFi AP `ESP32_DEV` | Stock Waveshare firmware only | Ours is off (good) | Stay off |

So “listen to all ports” here means: **USB console + raw UART1 RX + ESP32 process/heap**. That is the whole live surface.

## Safety (battery click)

- Firmware must **not** send `WritePos` / torque-on / speed. Listen and optional ping/read only.
- Servos can enable **last EEPROM torque** the moment they get VIN. Support the arm. Clear the workspace. Be ready to unplug XT60.
- UART switch stays **ESP32** the whole time.
- One monitor on `/dev/ttyACM0`. Stop extras: `pkill -f esp_idf_monitor`.

USB does **not** hold the motors. Twitch risk is XT60 VIN, with or without USB.

Two supplies at once (USB VBUS + XT60): this HAT is meant to do that (5 V buck can feed the ESP32 with no USB). Hot-plug of the pack **while USB is enumerated** is a different question (ground bounce / VBUS). 2026-09-04: two USB-connected pack clicks and two unplugs — ESP32 did not reset, ACM0 stayed, `tq=0`, no twitch. That is one machine, not a guarantee.

Conservative ops if you do not need to listen: **USB out → pack in/out → USB in**. You go blind during the click. Bring-up listen needs USB in.

Not done yet: pack **on**, then unplug USB (does the chip stay up on VIN?).

## Firmware (`listen v1`, on the chip)

No ping on boot, no WiFi, no BLE, no motion. USB prints immediately so we know the chip is running. `get` is READ-only (present + torque-enable); do not type it until the XT60 listen window is done.

Boot banner (once):

```
listen v1
mac=…
reset=…
tasks=   (vTaskList: name, state, stack)
bus=uart1 gpio18/19 1Mbps rx-only until ping
note=no move. plug XT60 while this is printing.
```

Then a slow heartbeat so “is anything going on” is visible even when the bus is dead:

```
t=12345 heap=298000 bus_rx=0
```

Every UART1 byte (if any) as raw hex with time, **not** parsed until we ask:

```
bus t=12400 n=6 hex=ff ff 0b 02 00 f2
```

USB commands (type in monitor):

| | |
| --- | --- |
| (nothing) | keep heartbeats + raw RX dump |
| `state` | task list, heap, uptime, reset — HAT = this ESP32, nothing else is running |
| `listen` | RX dump on / off |
| `gpio` | digital snapshot of safe pins only (18/19 as UART, not a fishing trip on USB/strapping pins) |
| `ping` | IDs 11–17 (only when we decide the listen window is done) |
| `raw 11` | ping + dump the status packet hex |
| `get` | READ pos/speed/load/V/temp/current/moving/torque-en. **After** the listen window |
| `bench` | same comparable block as `usb-no-battery.md` |
| `help` | list |

Do **not** I2C-scan or ADC-scan floating pins as “sensors”. If we ever do it, label it `noise=` so we do not confuse it with HAT data.

## Procedure

### 0. Host (laptop), before flash

Second terminal, leave running:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
dmesg -w
```

Note which USB serial exists. Battery should **not** create a new COM port (ESP32 is already on USB). If a second ACM appears, that is the other Type-C or a reconnect — write it down.

### 1. USB only, pack off — **done 2026-09-04**

Flash listen firmware. UART switch ESP32. Confirm heartbeats. Run `state`. Confirm `bus_rx` stays 0 for ~30 s.

Gold: `docs/baselines/usb-start.md`. Short paste: `docs/baselines/listen-usb-no-battery.md`. MAC matches `usb-no-battery.md`.

Got: chip alive, 5 FreeRTOS tasks (`main` + IDLEs + IPC), bus hex **none**. Did not ping.

### 2. Keep listening, plug XT60

Do not reset the ESP32. Watch:

- USB heartbeats — should continue (`t=` increasing). If they stop, USB brownout / port reset. That is a finding.
- `bus_rx` / hex lines — **expect still zero**. Servos waking is not a broadcast.
- Board LEDs — 3.3 V power LED already on from USB; bus TX/RX LEDs only if bytes move; pack-side LED if the 9–25 V rail comes up.
- `dmesg` — expect no new USB device.
- Arm — any twitch = servo power + stored torque. Unplug if unsafe. Still a finding (power reached the motors).

Mark the log: `--- xt60 on ---` (type that as a comment in the notebook; the chip will not know).

Wait ~20 s of listen-only.

### 3. Then poke the bus (still no move)

`state` then `ping` then `bench`.

| Result | Meaning |
| --- | --- |
| `found=0/7` still | no VIN to servos, switch not ESP32, or wiring |
| `found=7/7` | pack + bus + IDs 11–17. Save `docs/baselines/usb-battery-on.md` |
| `found=1..6` | partial chain. Save IDs. Do not move. |
| hex on RX only after ping | confirms “silent until addressed” — expected ST protocol |

Then `raw 11` … `raw 17` if ping hits, then `get`. That data lives **in the servos**, not on the HAT. Save `docs/baselines/usb-battery-on.md`.

### 4. What “sensors came alive”

Write the answer in the baseline file, even if it is “none on the HAT”.

Likely:

- **HAT:** same ESP32 process. 5 V buck for Pi header (we are not measuring it in firmware).
- **Bus:** ST nodes answer ping; voltage/temp/current are servo ADC inside each motor.

Not coming alive: INA219, IMU, OLED, user LED, WiFi AP.

## Already known (do not redo)

USB hello works. GPIO 2 is not an LED. `printf` in a 2 KB task stack-overflows. Battery off: `bench found=0/7`, listen `bus_rx=0`, MAC `44:1d:64:86:3f:54`. Day logs: `development-logs/2026-09-03.md`, `2026-09-04.md`.

## After this experiment

1. USB-only listen baseline — **saved** (`usb-start.md`).
2. Plug XT60, listen, `ping` / `get` / `batt` — **saved** (`usb-battery-on.md`). Power cycle too. Still no auto-move.
3. Next: hand-move, `tq=0`: `docs/plans/hand-move.md`.
4. Then BLE HTML: `docs/plans/ble-html.md`. VIN-only ESP32 still open.
5. Only then a tiny commanded nudge on one joint.

Order: `docs/plans/usb-then-ble.md`.
