# Link (USB + BLE)

This chip: ESP32-WROOM-32 on Bus Servo Driver HAT. UART switch **ESP32**.

| Power | ESP32 | Motors (`get`) |
| --- | --- | --- |
| USB, XT60 off | on (VBUS) | all miss |
| USB + XT60 | on | live |
| XT60 on, USB out | on from buck | live if chip stayed up |
| nothing | dead | — |

USB powers the chip; **XT60 powers servos**. USB does not charge the pack. Limb pack is **3S** (HAT 9–25 V). **2S max 8.4 V — do not plug it in.** Meter / store / recycle: `../roarm-fw/docs/hardware/lipo.md`.

## USB

`/dev/ttyACM0`, WCH CH343 `1a86:55d3`, 115200. Prefer USB when the node exists. BLE thread sleeps while `serial_ok` so we do not hold two pipes (and NimBLE `MAX_CONNECTIONS=1` is BLE-only anyway).

## BLE

Nordic UART: svc `6e400001-…`, RX write `…0002`, TX notify `…0003`. Advertise name `gibbon` + UUID; scan response `gibbon-XXXX` (`gibbon-3f54` on this board).

| | |
| --- | --- |
| WiFi STA MAC | `44:1d:64:86:3f:54` (identity) |
| BLE address | `44:1D:64:86:3F:56` (`.ble-addr`, `GIBBON_BLE`) |

Laptop is central. Direct connect to that address — no 5 s name scan. Names on BlueZ are often empty; filtering `gibbon*` misses a live advertiser.

### Why “unplug USB / already on pack” needed a battery yank

BlueZ keeps a GATT session after Django/python dies (`fuser -k`, reloader). Firmware `MAX_CONNECTIONS=1` → no advertise → scan finds nothing. Power cycle cleared the radio.

**Host fix:** `bluetoothctl disconnect` that address, then `BleakClient(addr)`. Backoff 1–12 s if the pack is off. `atexit` disconnects.

**Firmware (flash when USB is in):** slower advertise (~160–320 ms, not 20–40 ms). If connected but never subscribed for 5 s, `ble_gap_terminate` (“ghost”). Connection: interval 24–40, latency 2, supervision 2 s. Code: `roarm-fw/main/ble_nus.c`, `ble_nus_tick()` from `app_main`.

Until that flash, host disconnect is enough to recover without pulling XT60.

## Energy

- Do not discover() in a 5 s loop. Address connect first.
- No BLE scan while USB is up.
- Advertise slow on the chip (after flash).
- `stream 180` is the bus poll. Stop with `stream 0` on drop.

## Page auto-connect

`GET /api/status` every 2 s. EventSource `/api/rx` reconnects on error. `ble up` / `serial up` retriggers `afterConnect`. Hard-refresh if an old service worker still caches the static PWA.
