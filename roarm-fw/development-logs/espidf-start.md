# ESP-IDF — this machine (once)

Living **every-time** loop: [`docs/framework/espidf.md`](../docs/framework/espidf.md). Update that file when the flash/serial ritual changes.

## Already done (2026-09-03)

- IDF v5.4.2 at `$HOME/esp/esp-idf` (not in git)
- `python3.11 ./install.sh` is wrong — `install.sh` is bash; use `scripts/idf-env.sh` so 3.11 wins over system 3.14
- cmake/ninja: `python3 "$IDF_PATH/tools/idf_tools.py" install cmake ninja`
- `usermod -aG dialout t` — **this session still needs** `sudo chmod 666 /dev/ttyACM0` (`newgrp` not installed)
- Firmware on the chip: USB `listen v1` (`docs/framework/espidf.md`). IDF venv python can die if `/tmp/py311bin` vanished — `scripts/idf-env.sh` recreates it.
