# ESP-IDF — every time

Chip: ESP32-WROOM-32 on the HAT. Port: **`/dev/ttyACM0`**. UART switch: **ESP32**.

## Stop the monitor

Ctrl+] does **not** work (Italian keyboard / Cursor). Open a **second** terminal and run:

```bash
pkill -f esp_idf_monitor
```

No IDF env needed. The monitor tab should drop back to a prompt. Then you can flash again.

Do not start a second `flash monitor` until that process is dead.

## Start

New terminal = source the env again (`idf.py: command not found` means you skipped this).

```bash
cd ~/code/gibbon-bot/roarm-fw
. scripts/idf-env.sh
sudo chmod 666 /dev/ttyACM0          # until a full desktop logout
idf.py -p /dev/ttyACM0 flash monitor
```

| | |
| --- | --- |
| Stop monitor | Second tab: `pkill -f esp_idf_monitor` |
| `idf.py: command not found` | `. scripts/idf-env.sh` in **this** shell |
| Path not readable | `sudo chmod 666 /dev/ttyACM0` |

Edit → `pkill -f esp_idf_monitor` → same start commands.

## Now on the chip

No auto-move. Writes: **hand** (17) and **twist** (16). Twist wheel guard (`ESTOP`) is in source — flash before commanding twist. Details: `docs/software.md`.

**This HAT has no user LED.** USB still powers the ESP32 with the battery unplugged. Servos do **not** — position / load / voltage come from the motors, so those reads are silence until XT60 is on.

The RoArm kit’s INA219 / IMU / LED were on the **old** General Driver Board. This HAT has none of that.

Every USB plug, pack off: match `docs/baselines/usb-start.md` (`state`, heartbeats `bus_rx=0`) before anything else.

Type in the monitor, Enter:

| | |
| --- | --- |
| (nothing) | heartbeats + raw UART1 RX. no move |
| `state` | chip / tasks / heap / `bus_rx` |
| `gpio` | levels on 18/19 (still UART) |
| `info` | chip, MAC, heap, reset, IDF |
| `listen` | RX hex dump on/off |
| `ping` | Feetech ping IDs 11–17 (UART switch = **ESP32**) |
| `pingall` | IDs 1–253, only prints hits |
| `raw ID` | ping + READ present, dump hex |
| `get` | READ pos/speed/load/V/temp/current/moving/torque-en. **After** XT60 listen |
| Host `watch-hand.py` | Laptop loops `get` while you backdrive. `scripts/watch-hand.py`. Do not run monitor at the same time |
| `hand open` / `hand close` / `hand to N` / `hand free` | ID 17 only. Soft 689–2008, `to` max 2075. Walls 609 / 2088 |
| `batt` | pack V from servo VIN + 3S rest-% guess. Not a BMS. USB does not charge XT60 |
| `bench` | one-shot baseline: chip + ping 11–17 — compare `docs/baselines/usb-no-battery.md` |
| `help` | list |

Battery **off**: heartbeats `bus_rx=0`; `bench` would print `found=0/7` `guess=usb-no-battery`. Do not ping until `usb-start.md` matches.

Battery **on** + switch ESP32: expect `found=7/7`, then `get`. Procedure: `docs/plans/characterize-hat.md`.

UART switch must be **ESP32**. USB-C next to the bus headers is a different path (direct to servos); this firmware uses the ESP32 port.

## Bring-up on this board (keep)

Tried in order. Dead ends stay here so we do not repeat them.

| Step | What we did | Result |
| --- | --- | --- |
| 1. USB hello | Print `roarm-fw usb hello` / `alive` on UART0 | **Worked.** Chip ESP32-D0WD-V3 rev 3.1, MAC `44:1d:64:86:3f:54`, 4 MB flash |
| 2. Blink LED | GPIO 2 (DevKit habit) + serial `hz` / `tick` | **No lamp.** HAT has no user LED. Schematic LED3 is 3.3 V power. RoArm PH2.0 LED was on the old General Driver Board |
| 3. Blink task | `printf` from a 2 KB FreeRTOS task | **Reboot loop** (`stack overflow in task blink`). Console I/O stays on the main task |
| 4. Serial ticks | `hz` / `off` / `on` / `tick 1` `tick 0` | Worked as a USB test; dropped once we knew there is no LED |
| 5. `info` | Chip, MAC, heap, reset, IDF, uptime | **Works on USB alone** (battery unplugged) |
| 6. `ping` 11–17 | Feetech ping UART1 GPIO 18/19 @ 1 Mbps | Battery **off**: all miss. Expected — servos unpowered |
| 7. `listen v1` | Heartbeats + raw UART1 RX, no ping on boot | USB-only: `bus_rx=0`, 5 tasks, MAC match. Gold `docs/baselines/usb-start.md`. Next: XT60 |

Host traps (Italian keyboard + this Ubuntu):

- Ctrl+] does not quit the monitor. Process name is `esp_idf_monitor`, not `idf_monitor.py`. Second terminal: `pkill -f esp_idf_monitor` or `scripts/stop-monitor.sh`.
- `t` is in `dialout`. 2026-09-04 session could open ACM0 without `chmod`. If Permission denied: `sudo chmod 666 /dev/ttyACM0` (`newgrp` is not installed).
- Two monitors on ACM0 fight. Stop the first before flash.

## Once (already done on this machine)

IDF **v5.4.2** at `$HOME/esp/esp-idf` (not in git). Python **3.11** via `scripts/idf-env.sh`. `usermod -aG dialout t` done; group applies after logout.

If you ever reinstall: `development-logs/espidf-start.md`. Why IDF: `docs/framework/framework.md`.
