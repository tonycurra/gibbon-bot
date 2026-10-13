# gibbon-console

The laptop screen for [Gibbon Bot](../README.md), Olivabot’s laboratory robot in Amsterdam. The robot does not walk. This program owns the USB cable and Bluetooth to the arm. A phone on the same Wi-Fi is just a browser. Tony wrote it. Jade Leurs’s simulation work is in [`archive/`](../archive/README.md).

Firmware: [`roarm-fw`](../roarm-fw/readme.md). One-arm model: [`digital-twin`](../digital-twin/readme.md).

Firefox has no Web Serial and no Web Bluetooth, so this process is the bridge. It logs a trial as commands, motor samples, notes, and limits. The chip still owns `estop`. This page cannot save the arm if the radio dies.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python manage.py migrate
.venv/bin/python manage.py test
.venv/bin/python manage.py runserver 0.0.0.0:8080
```

Open `http://127.0.0.1:8080`. Header: usb / ble / pack / **LOCK** / **STOP**. Escape is stop. One process owns `/dev/ttyACM0`. Stop this server before flashing the chip.

Bind it to localhost unless the phone on your own LAN needs it. Do not put this process on the public internet. It can move a limb.

Docs: [`docs/README.md`](docs/README.md). Code is [Apache-2.0](LICENSE). Report a stop-button or limit bug via [SECURITY](SECURITY.md).

Pages: home, calibration, imitation. Chromium `/monitor/` is BLE listen-only (packed `g1`). Unplug USB and flash `bringup/02-ble-monitor` from [`roarm-fw`](../roarm-fw/readme.md) first. Doc map: [`docs/README.md`](docs/README.md).
