# 01 — USB listen

Chip speaks. Host listens. Pack **off**. No bus, no BLE, no move.

Firmware: `01-usb-listen.c` (~160 lines). Host: `listen.py`.

## What it proves

USB-C on the ESP32 debug port enumerates, the WROOM runs, UART0 prints, the chip stays up. That is the whole test.

| | Must be |
| --- | --- |
| USB-C | ESP32 debug, **not** the bus Type-C |
| Host | `/dev/ttyACM0` 115200 |
| UART switch | **ESP32** |
| XT60 | unplugged |
| Radio / bus | off (`ble=0 bus=0`) |

## Run

Stop anything on ACM0 (`../scripts/stop-monitor.sh`, Django console, `listen.py` already running).

```bash
cd ~/code/gibbon-bot/roarm-fw/bringup
python3 listen.py --log /tmp/01-usb-listen.txt
```

Then plug the board (or unplug and plug again). Opening serial resets the ESP32; you should see boot ROM noise, then:

```
bringup 01-usb-listen
mac=44:1d:64:86:3f:54
reset=poweron
tasks=5 heap=…
note=usb only. no ble. no bus. no move.
state t=… heap=… reset=poweron tasks=5
state mac=44:1d:64:86:3f:54 fw=01-usb-listen
t=… heap=… reset=poweron
```

Heartbeats every **2 s**. `t=` goes up, `heap=` stays flat. Unplug: `# status=gone`, then `# status=waiting`.

Type `info` or `state` in a second serial tool only if you need a one-shot dump. `listen.py` is read-only.

Flash (replaces whatever is on the chip — sitting `roarm-fw` is gone until you flash that tree again):

```bash
./flash.sh 01-usb-listen
python3 listen.py
```

## Gold (this HAT)

Same chip as `docs/baselines/usb-start.md`.

| Field | Gold |
| --- | --- |
| USB | `/dev/ttyACM0` |
| `mac=` | `44:1d:64:86:3f:54` |
| `info chip=` | `ESP32 cores=2 rev=301 flash=4096KB` |
| `idf=` | `v5.4.2` until we bump |
| `tasks=` | **5** (`main` + 2×IDLE + 2×ipc). No `wifi` / `bt` |
| `ble=` / `bus=` | `0` |
| heartbeat | ~2 s, heap not collapsing |

`reset=` may be `poweron` (plug) or `usb` (serial open toggled DTR). Ignore unless `brownout` / `panic` / loop.

## Abnormal — stop

| Symptom | Likely |
| --- | --- |
| No `/dev/ttyACM0` | wrong USB-C, cable, or chip dead |
| MAC ≠ gold | wrong board |
| Heartbeats stop, `reset=brownout` | USB starve / port reset |
| Reboot loop, `stack overflow` | do not add a printf task (already burned) |
| `tasks` > 5 | radio or extra task crept in |
| Heap falling | leak |
| `bringup` line never appears | not this image, or baud/port wrong |

No servo `pos` / `V` here. USB powers the ESP32 only. Next test talks to the bus.

## After this sitting

Reflash production from `../` when you need the arm again:

```bash
cd ~/code/gibbon-bot/roarm-fw
. scripts/idf-env.sh
idf.py -p /dev/ttyACM0 flash
```
