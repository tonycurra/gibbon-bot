Laptop UI + USB/BLE bridge + trial log. Chip: `../roarm-fw`.

**Docs:** [`docs/README.md`](docs/README.md) (run, architecture, link, data, UI, calibrate, next).

```bash
cd /home/t/code/gibbon-bot/gibbon-console
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver 0.0.0.0:8080
```

Firefox `http://127.0.0.1:8080` — **home · calibration · imitation**. Header: usb / ble / pack / **LOCK** / **STOP**. Phone: same WiFi. One process owns ACM0.

Chromium `/monitor/` is BLE listen-only (packed `g1`). Unplug USB. Flash `../roarm-fw/bringup/02-ble-monitor` first.
