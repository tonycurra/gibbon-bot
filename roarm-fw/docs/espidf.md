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

## Now on the chip: `info` + `ping`

**This HAT has no user LED.** USB still powers the ESP32 with the battery unplugged. Servos do **not** — position / load / voltage come from the motors, so those reads are silence until XT60 is on.

The RoArm kit’s INA219 / IMU / LED were on the **old** General Driver Board. This HAT has none of that.

Type in the monitor, Enter:

| | |
| --- | --- |
| `info` | chip, MAC, heap, reset, IDF — works on USB alone |
| `ping` | Feetech ping IDs 11–17 (UART switch = **ESP32**) |
| `pingall` | IDs 1–253, only prints hits |
| `bench` | one-shot baseline (chip + ping 11–17) — compare `docs/baselines/usb-no-battery.md` |
| `help` | list |

Battery **off**: `bench` should print `found=0/7` `guess=usb-no-battery`. Gold copy: `docs/baselines/usb-no-battery.md`.

Battery **on** + switch ESP32: expect `found=7/7`. Then we add `get` (pos/load/V).

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
| 6. `ping` 11–17 | Feetech ping UART1 GPIO 18/19 @ 1 Mbps | Battery **off**: all miss. That is expected — servos are unpowered. Next: battery on, same command |

Host traps (Italian keyboard + this Ubuntu):

- Ctrl+] does not quit the monitor. Process name is `esp_idf_monitor`, not `idf_monitor.py`. Second terminal: `pkill -f esp_idf_monitor` or `scripts/stop-monitor.sh`.
- `t` is in `dialout` but this session still needs `sudo chmod 666 /dev/ttyACM0` until a full desktop logout (`newgrp` is not installed).
- Two monitors on ACM0 fight. Stop the first before flash.

## Once (already done on this machine)

IDF **v5.4.2** at `$HOME/esp/esp-idf` (not in git). Python **3.11** via `scripts/idf-env.sh`. `usermod -aG dialout t` done; group applies after logout.

If you ever reinstall: `development-logs/espidf-start.md`. Why IDF: `docs/framework.md`.
