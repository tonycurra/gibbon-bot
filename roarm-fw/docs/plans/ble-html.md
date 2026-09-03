# BLE HTML: same data, phone/laptop, all power states

After USB listen + `get` (`docs/plans/characterize-hat.md`). Not implemented yet.

One **static HTML + JS** page. Chrome **Web Bluetooth**. The chip speaks the **same text lines** as USB (`info`, `bench`, `ping`, `get`, later `raw`). BLE is only a wireless serial pipe (Nordic-UART-style, NimBLE). No WiFi on this WROOM-32 while BLE is on.

The page is a dashboard, not a new protocol. If a field is empty, the power state does not have that sensor — do not invent HAT IMU/INA219/LED.

## Power states (what can actually be alive)

ESP32 3.3 V comes from USB VBUS and, on this HAT, from the XT60→5 V buck (stock firmware runs a hotspot with no USB). Confirm the no-USB case by unplugging USB while BLE is connected.

| State | ESP32 | BLE page | Motors (`get`) |
| --- | --- | --- | --- |
| A. USB, XT60 **off** | On (USB) | Connect. `bench found=0/7`. Chip/MAC/heap/tasks | All miss. No pos/V/temp |
| B. USB + XT60 **on** | On | Same connect. `found=7/7` then `get` | Full servo table |
| C. XT60 **on**, USB **unplugged** | Should stay on from VIN | BLE should survive. If the page drops and never comes back, ESP32 was USB-only — that is a finding, write it down | Same as B if ESP32 stayed up |
| D. No USB, XT60 **off** | **Dead** | Impossible. Nothing to advertise | Off |

Do **A first** (USB still attached so we can flash and compare USB vs BLE on the same lines). Then B. Then unplug USB for C. Never expect D.

UART switch stays **ESP32**. No auto-move. Support the arm when XT60 goes on.

## What the page shows

Three blocks. Raw log always on, so we see bytes even when a parse fails.

**Board (ESP32 / HAT)** — works in A:

- BLE connected / RSSI if the browser gives it
- MAC `44:1d:64:86:3f:54` (this chip’s identity — motors have **no unique serial**)
- `bench` / `info`: chip, flash, heap, uptime, reset, `guess=`
- `state`: FreeRTOS tasks (is anything running besides listen + NimBLE)

**Bus** — A: silence / miss. B/C: ping hex.

**Motors 11–17** — only after XT60 and a successful ping. One row per ID:

| Field | Source | Notes |
| --- | --- | --- |
| ID | ping | 11–17 |
| Model / fw | servo EEPROM if readable | Not a factory serial; Feetech gives model + firmware rev |
| Position | `ReadPos` | 0–4095 |
| Speed | `ReadSpeed` | |
| Load | `ReadLoad` | stall / collision |
| Voltage | `ReadVoltage` | ÷10 = pack at that servo |
| Temp | `ReadTemper` | °C |
| Current | `ReadCurrent` | mA |
| Moving | `ReadMove` | 0/1 |
| Mode | EEPROM | servo vs wheel, if we read it |
| Raw | last status packet hex | always keep this |

HAT-side “sensors” that will stay blank in every state: INA219, IMU, OLED, user LED.

## Page (when we write it)

`docs/ble-ui/index.html` — no build, no npm. Serve it (Web Bluetooth is blocked on `file://`):

```bash
cd ~/code/gibbon-bot/roarm-fw/docs/ble-ui
python3 -m http.server 8080
```

Chrome → `http://localhost:8080`. Linux: user in `bluetooth` / Chrome allowed to use BLE.

Buttons: **Connect**, **info**, **bench**, **get** (disabled until `found>0`), **listen** (stream heartbeats). A `<pre>` of every TX/RX line. A table for IDs 11–17, empty cells = no data in this power state.

Phone: Chrome/Edge with Web Bluetooth, or nRF Connect as a debug pipe if the page is awkward.

## Firmware (when we write it)

NimBLE GATT: one RX char, one TX notify, UART-NUS UUIDs. Advertise `roarm-fw` + MAC suffix. USB UART0 still works at the same time for state A.

Same command parser as USB. **Do not** start BLE until USB `get` has been seen once with battery on — otherwise the page will look broken when it is only `found=0`. Exception: a tiny BLE `info` hello on USB-only is allowed to prove radio, still no fake motor numbers.

Keep WiFi off.

## Test pass (fill baselines)

1. A — USB, no pack. Page connects. Matches `docs/baselines/usb-no-battery.md`. Motor table empty. Save a screenshot / log as `docs/baselines/ble-usb-no-battery.md`.
2. B — plug XT60, still USB. `get` fills the table. Save `docs/baselines/ble-usb-battery-on.md` (same numbers as USB `get`).
3. C — unplug USB. Page stays up or reconnects. Same motor numbers. Save `docs/baselines/ble-battery-only.md`. If BLE dies, ESP32 dropped — USB was its only 3.3 V.

## Order vs USB listen

Do not skip `docs/plans/characterize-hat.md`. BLE cannot show motor fields USB never proved. HTML is the wireless view of that data, then the four-limb host can reuse the same pipe.
