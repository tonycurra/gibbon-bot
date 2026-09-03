# ESP-IDF — start and test (this machine)

Chip: ESP32-WROOM-32. USB serial seen as **`/dev/ttyACM0`**. Console **115200**.

System Python is 3.14; IDF wants 3.10–3.12. Use **`/home/t/.local/bin/python3.11`**. From this repo:

```bash
. scripts/idf-env.sh
```

That also loads `$HOME/esp/esp-idf/export.sh`. IDF lives **outside** git.

CMake/Ninja were missing on PATH; install them into Espressif’s tools (once):

```bash
. scripts/idf-env.sh
python3 "$IDF_PATH/tools/idf_tools.py" install cmake ninja
. scripts/idf-env.sh   # again so cmake is on PATH
```

## Once: install

```bash
mkdir -p "$HOME/esp"
cd "$HOME/esp"
git clone --branch v5.4.2 --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/espressif/esp-idf.git
cd esp-idf
# Do not run: python3.11 ./install.sh  — that is a bash script.
./install.sh esp32
```

Add serial access (then log out and in):

```bash
sudo usermod -aG dialout "$USER"
```

Until then, if flash says permission denied: `sudo chmod 666 /dev/ttyACM0`.

## Every terminal

```bash
cd /home/t/code/gibbon-bot/roarm-fw
. scripts/idf-env.sh
```

The Espressif VS Code/Cursor extension can hide `export.sh`.

## Loop (this project)

HAT UART switch = **ESP32**. BOOT button if the port will not enter download mode.

```bash
idf.py set-target esp32
idf.py -p /dev/ttyACM0 build flash monitor
```

Exit monitor: **Ctrl+]** . Edit, up-arrow, enter.

Expect `roarm-fw usb hello` on the serial log. That is the USB test — not the servo bus yet.

If the port is `/dev/ttyUSB0` instead, use that. List: `ls /dev/ttyACM* /dev/ttyUSB*`.

## 2026-09-03 result

IDF v5.4.2 installed. `idf.py build` produced `build/roarm-fw.bin`. Flash was **not** done here: `t` is not in `dialout`, `/dev/ttyACM0` is `root:dialout`, sudo needs your terminal. After `sudo usermod -aG dialout t` (new login) or `sudo chmod 666 /dev/ttyACM0`:

```bash
. scripts/idf-env.sh
idf.py -p /dev/ttyACM0 flash monitor
```

## Not yet

Servo ping/get/move, NimBLE, flash encryption. `menuconfig` later for NimBLE. Do not enable flash encryption while iterating.
