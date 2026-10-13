# 02 — BLE monitor

USB + pack together is a fight (5 V rail vs XT60, the flicker you heard). This image is **listen-only** over BLE. Flash on USB, then **unplug USB**. Pack powers the ESP32 (5 V buck) and the servos.

Firmware: `02-ble-monitor.c`. Console: `/monitor/` in `gibbon-console`.

## Wire

One BLE notify per 50 ms. Page owns labels/units. Chip only sends:

```
g1 <82 bytes hex>\n
```

`g1` payload (little-endian): `t_ms:u32` `miss:u8` then 7 × (`pos:u16` `spd:i16` `load:i16` `cur:i16` `v10:u8` `temp:u8` `flags:u8`). `flags` bit0 = tq, bit1 = moving. `v10` = tenths of a volt. Subscribe starts the stream. First successful present read writes **`tq=0` all**. No commands from the page.

## Flash / run

Stop Django and `listen.py` (they own ACM0).

```bash
cd ~/code/gibbon-bot/roarm-fw/bringup
./flash.sh 02-ble-monitor
```

Then **unplug USB**. HAT switch **ESP32**. Pack on. Laptop:

```bash
cd ~/code/gibbon-bot/gibbon-console
.venv/bin/python manage.py runserver 0.0.0.0:8080
```

Chrome / Chromium `http://127.0.0.1:8080/monitor/` (or open `bringup/monitor.html` via localhost). **Connect** uses Web Bluetooth. The page sends **no commands** — subscribe starts the matrix. Labels and units live in the table header, not on the chip.

A 3S rest is 11.1–12.6 V. 5.1 V is USB 5 V on servo VIN — not the LiPo.

## Not this image

No raise, no hand/twist commands, no walls, no ESTOP guard (only `estop` → tq=0). Sitting firmware is still `../main/main.c`.
