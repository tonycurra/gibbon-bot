# Framework pick

Chip: **ESP32-WROOM-32** on Waveshare Bus Servo Driver HAT (A).  
Wanted: USB iterate, low-level control, room to grow, BLE, performance, later security.

**Pick: native ESP-IDF (`idf.py`), NimBLE when BLE starts.**  
Not PlatformIO+Arduino. Not PlatformIO wrapping IDF unless `idf.py` becomes painful.

Bring-up order is unchanged: USB `ping` / `get` / `move` first, BLE as the same commands later. No auto-move. See `docs/plans/usb-then-ble.md`.

## Three things people mix up

The web comparisons treat “PlatformIO vs ESP-IDF” as two products. It is three stacks:

| Stack | What you write | Who owns the SDK |
| --- | --- | --- |
| PlatformIO + **Arduino** | `setup()` / `loop()`, `SCServo.h` | Arduino-ESP32 core |
| PlatformIO + **ESP-IDF** | IDF components, FreeRTOS | Espressif, but **PIO’s copy can lag** |
| Native **ESP-IDF** | same IDF code | Espressif, `idf.py`, current release |

Arduino vs IDF is the real architecture choice. PIO vs `idf.py` is only how you invoke the build.

PIO’s multi-chip story (STM32, AVR, …) does not matter here. This product is Espressif only.

## Why not PIO+Arduino for this use case

It wins the **shortest wait after an edit** (cached core/libs). That is why reupload felt fast.

It loses on everything else on the list:

- BLE on classic ESP32 Arduino is the flaky stack. NimBLE is first-class in IDF.
- WROOM-32 is RAM-tight. NimBLE vs Bluedroid is a real budget, not a slogan.
- Dual-core: servo bus on a high-priority task, BLE on another — FreeRTOS the IDF way, not `loop()`.
- Four identical limbs, safety, later Secure Boot — IDF `components/` + `menuconfig`, not a flat sketch.
- Rewriting Arduino → IDF later is the cost this pick avoids.

USB iteration still works: `idf.py -p /dev/ttyACM0 build flash monitor` (or `ttyUSB0`). First build is long. After that Ninja only rebuilds what changed — same *kind* of loop as PIO, not Arduino IDE. How-to: `docs/framework/espidf.md`.

## Why not PIO wrapping IDF

You would write IDF anyway. PIO gives `platformio.ini` and skips `export.sh`.

Cost: Espressif BLE / security fixes land in IDF first; PIO’s `espressif32` package often trails. For “BLE stably” and “grow for years”, lag is the actual risk. Secure Boot / flash encryption / custom partitions are also clumsier through the wrapper.

Use PIO+IDF only if native IDF install is a mess and you are willing to pin an older IDF.

## Native IDF — iterate

Espressif Cursor/VS Code extension **or** one shell with the env loaded:

```bash
. $HOME/esp/esp-idf/export.sh          # once per terminal (extension can hide this)
idf.py -p /dev/ttyACM0 build flash monitor
```

Exit monitor: **Ctrl+]** . Up-arrow, enter. That is the loop.

`menuconfig` for BLE/NimBLE, partitions, later security. Do **not** turn on flash encryption while still iterating USB tests — it slows bricks and flashes.

Agent and human: same `idf.py` commands, same as we wanted from `pio`.

## Security (honest)

IDF can do Secure Boot, flash encryption, encrypted NVS. That is production-device stuff. For this arm, v1 security is: slow moves, voltage/load cutoff, no auto-home slam, BLE pairing later. Hardware crypto when the protocol is stable, not on day one.

## Commands cheat sheet

| Action | IDF | PIO (Arduino, if you still want it) |
| --- | --- | --- |
| Build | `idf.py build` | `pio run` |
| Flash | `idf.py flash` | `pio run -t upload` |
| Serial | `idf.py monitor` | `pio device monitor` |
| All three | `idf.py flash monitor` | `pio run -t upload && pio device monitor` |
| Config | `idf.py menuconfig` | `pio run -t menuconfig` (IDF env only) |

## Verdict

| Goal | Winner |
| --- | --- |
| Fastest Arduino-style reupload | PlatformIO + Arduino |
| This product (WROOM-32, BLE, grow, low-level) | **ESP-IDF native** |
| PIO as a button over IDF | Optional later, not the source of truth |
