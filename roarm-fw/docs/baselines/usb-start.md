# USB start — healthy board (XT60 off)

Gold for **every bring-up**. USB in, pack still unplugged, UART switch **ESP32**. Compare serial to this page before you trust anything else.

If this page does not match, stop. Do not `ping`, `get`, or move.

Captured 2026-09-04 on this HAT, firmware `listen v1`. Chip identity also matches 2026-09-03 `usb-no-battery.md`.

## Setup

| | Must be |
| --- | --- |
| USB-C | ESP32 debug port, **not** the bus-servo Type-C |
| Host node | `/dev/ttyACM0` 115200 |
| UART switch | **ESP32** |
| XT60 | **unplugged** |
| Firmware | no auto-move, no WiFi, no BLE, no ping on boot |

Heartbeats print by themselves. Type `state`, `gpio`, `info`.

## Identity (this physical chip)

These must not change unless you swapped the HAT.

| Field | Gold | Notes |
| --- | --- | --- |
| USB | `/dev/ttyACM0` | One ACM. A second ACM is the other Type-C or a reconnect |
| esptool chip | `ESP32-D0WD-V3` rev `v3.1` | Dual core, 240 MHz, 40 MHz crystal |
| `info chip=` | `ESP32 cores=2 rev=301 flash=4096KB` | rev 301 = v3.1 |
| `mac=` | `44:1d:64:86:3f:54` | STA MAC. Motors have no unique serial |
| `idf=` | `v5.4.2` | Until we bump IDF |
| `usb=` | `115200` | UART0 |
| `bus=` | `gpio18/19@1Mbps ids=11-17` | UART1, switch must be ESP32 |

Heap ~298000 (today `297940`). Uptime and `reset=` **may change** — ignore unless `brownout` / `panic` / reboot loop.

## Gold serial (2026-09-04)

```
info chip=ESP32 cores=2 rev=301 flash=4096KB reset=poweron
info mac=44:1d:64:86:3f:54 heap=297940 uptime_ms=123760
info idf=v5.4.2 usb=115200 bus=gpio18/19@1Mbps ids=11-17
info note=listen-only until ping/get. no move. servos need XT60

state t=37780 heap=297940 reset=poweron tasks=5 bus_rx=0 dump=1
state mac=44:1d:64:86:3f:54
state bus=uart1 gpio18/19 1Mbps rx-only until ping
tasks:
main           	X	1	1880	3
IDLE1          	R	0	1024	5
IDLE0          	R	0	1028	4
ipc0           	S	24	504	1
ipc1           	S	24	496	2

gpio 18(rx)=1 19(tx)=1 (uart1, not remuxed)

t=24010 heap=297940 bus_rx=0
t=26010 heap=297940 bus_rx=0
… every 2 s, t= increases, heap flat, bus_rx stays 0 …
```

Expected boot banner (once per reset):

```
listen v1
mac=44:1d:64:86:3f:54
reset=poweron
tasks=5
bus=uart1 gpio18/19 1Mbps rx-only until ping
note=no move. plug XT60 while this is printing.
```

## What this state is allowed to show

This HAT is a UART switch + bus driver + ESP32. USB powers **only the chip**.

| Observable | Normal now | Meaning |
| --- | --- | --- |
| Heartbeat `t=` | increases ~2000 ms | main loop alive |
| `heap=` | ~298 kB, not collapsing | no leak / panic |
| `bus_rx=` | **0** | UART1 RX empty |
| `bus t=… hex=` lines | **none** | ST servos do not broadcast, and they have no VIN |
| `tasks=` | **5**: `main` `IDLE0` `IDLE1` `ipc0` `ipc1` | no WiFi/BLE/motion task |
| `main` state `X` | running | listen loop |
| `ipc*` state `S` | blocked | dual-core IPC idle |
| stack watermark `main` | ~1880 | `printf` lives here; do not add a 2 KB print task |
| GPIO 18 / 19 | **1 / 1** | UART idle-high |
| 3.3 V power LED | on | USB VBUS |
| Bus TX/RX LEDs | off / idle | no bytes |
| User LED / GPIO 2 lamp | **none** | this PCB has no user LED |
| INA219, IMU, OLED | **none** | old General Driver Board only |
| WiFi AP `ESP32_DEV` | **absent** | stock Waveshare firmware only |
| Servo pos / speed / load / V / temp / current / moving / torque | **absent** | live inside ST3215s; need XT60 + a READ |

Optional, only if you already know this page matched: `bench` → `found=0/7` `guess=usb-no-battery`. Gold block: `usb-no-battery.md`. Do not treat `found=0` as a bus fault in this state.

## Abnormal — stop

| Symptom | Likely |
| --- | --- |
| MAC ≠ `44:1d:64:86:3f:54` | wrong board |
| No `/dev/ttyACM0` | wrong USB-C, cable, or chip dead |
| Extra `/dev/ttyACM1` | other Type-C or USB reset; write it down |
| Heartbeats stop, `reset=brownout` / `usb` | USB brownout or port reset |
| Reboot loop, `stack overflow` | firmware bug (seen before with `printf` on a 2 KB task) |
| `bus_rx` climbing, hex lines, pack still off | noise, echo, or something else TX on the bus — unexpected |
| `tasks` > 5, names like `wifi` / `bt` | radio came up; should be off |
| Heap crashing toward zero | leak / overflow |
| GPIO 18/19 not UART / switch not ESP32 | bus isolated; later ping will miss even with battery |

## What we still cannot know here

Plug XT60 **after** this page matches. Keep USB. Do not move.

Then, still listen-only at first: heartbeats should continue, `bus_rx` still likely **0** (servos stay quiet until addressed). Then `ping` / `bench` / `get` (READ only).

Pack-on gold: `usb-battery-on.md` (`found=7/7`, `batt pack=11.4V`).

## Unplug XT60, USB stays (2026-09-04)

Chip did **not** reset. Same ACM0. Heartbeats kept going (`bus_rx=0`). Then:

```
ping none 11-17 — expected if battery is off
found=0/7
guess=usb-no-battery
batt pack=miss est=? note=servos need XT60
```

Back to this page. USB powers the ESP32; pack only powers the servos.

Repeat 2026-09-04 (USB dead → USB in → pack in → pack out): same. USB replug is `reset=poweron` and `t=` restarts. Pack click with USB up: chip stays, `found=7/7` then `0/7`. Safety note: `docs/plans/characterize-hat.md` (USB does not hold motors; USB-out then pack click is more isolated but you cannot listen).

## How to re-run

Monitor already open: watch two heartbeats (`bus_rx=0`), type `state`, `gpio`, `info`.

Every-time flash: `docs/framework/espidf.md`.

```bash
cd ~/code/gibbon-bot/roarm-fw
. scripts/idf-env.sh
idf.py -p /dev/ttyACM0 flash monitor
```

Stop monitor from a second terminal: `pkill -f esp_idf_monitor` or `scripts/stop-monitor.sh`.
