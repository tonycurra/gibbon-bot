# ESP-IDF (how we build)

Native Espressif framework. Why: `docs/framework.md`. Chip: ESP32-WROOM-32. This repo is an IDF project (`CMakeLists.txt` + `main/`).

Machine notes (Python 3.11, cmake via idf_tools, dialout): `development-logs/espidf-start.md`.

## Install (once)

IDF is **not** inside git. Clone to `$HOME/esp/esp-idf`. `install.sh` is bash (not `python3.11 ./install.sh`). Put Python 3.11 first if `python3` is 3.14:

```bash
mkdir -p "$HOME/esp"
cd "$HOME/esp"
git clone --branch v5.4.2 --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/espressif/esp-idf.git
cd esp-idf
mkdir -p /tmp/py311bin
ln -sfn "$HOME/.local/bin/python3.11" /tmp/py311bin/python3
PATH="/tmp/py311bin:$PATH" ./install.sh esp32
```

Then cmake/ninja if missing:

```bash
. /path/to/roarm-fw/scripts/idf-env.sh
python3 "$IDF_PATH/tools/idf_tools.py" install cmake ninja
```

Serial: `sudo usermod -aG dialout "$USER"` then a new login.

## Every shell

```bash
cd /path/to/roarm-fw
. scripts/idf-env.sh
idf.py -p /dev/ttyACM0 build flash monitor
```

First `build` is long. Later edits are incremental. Exit monitor: **Ctrl+]**.

This HAT enumerated as **`/dev/ttyACM0`**. UART switch = **ESP32**. Hold **BOOT** if download fails.

## First test

Prints `roarm-fw usb hello` then `alive` on USB. Servo bus is not started.

Build on 2026-09-03: **ok**. Flash: blocked until you are in `dialout`.

## Commands

| | |
| --- | --- |
| Env | `. scripts/idf-env.sh` |
| Target | `idf.py set-target esp32` |
| Build | `idf.py build` |
| Flash + serial | `idf.py -p /dev/ttyACM0 flash monitor` |
| Config | `idf.py menuconfig` |
| Size | `idf.py size` |
