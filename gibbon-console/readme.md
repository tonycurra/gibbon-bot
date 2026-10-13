# gibbon-console

Django sitting UI for the RoArm ESP32. The laptop owns USB and BLE. The phone is a browser on the same Wi-Fi. Part of [Gibbon Bot](https://olivabot.com/en/projects/gibbon-bot.html).

Firmware: [gibbon-roarm-fw](https://github.com/tonycurra/gibbon-roarm-fw). One-arm model: [gibbon-digital-twin](https://github.com/tonycurra/gibbon-digital-twin). Laboratory map: [gibbon-bot](https://github.com/tonycurra/gibbon-bot).

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

Pages: home, calibration, imitation. Chromium `/monitor/` is BLE listen-only (packed `g1`). Unplug USB and flash `bringup/02-ble-monitor` from [gibbon-roarm-fw](https://github.com/tonycurra/gibbon-roarm-fw) first. Doc map: [`docs/README.md`](docs/README.md).
