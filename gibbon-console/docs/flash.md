# Flash (USB from the browser)

Today: laptop `idf.py flash`. Console owns `/dev/ttyACM0` — stop it first.

Next: Chromium **Web Serial** on `/flash/`. USB only, pack can be off (USB powers the ESP32, not the servos). Phone: Android Chrome + USB-C OTG. Firefox and iOS Safari have no Web Serial.

## Identity

Chip prints `info board=gibbon-XXXX` (MAC last two bytes) and `info fw=`. Same id as BLE name. Calibration snapshots key off that.

## Signed images (closed source)

ESP-IDF Secure Boot v2 + signed app. Signing key stays off the repo. Console verifies the downloaded `.bin` against the public key / image hash before write. Unsigned bits do not go on production boards.

Not wired yet. This sitting still uses `idf.py`.
