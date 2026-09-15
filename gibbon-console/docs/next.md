# Next

Do not start a new protocol on the sitting chip. Parse the text it already prints (compact or fat `get`). Packed `g1` is **bring-up only** (`02-ble-monitor` + `/monitor/`).

## Do not invent

HAT IMU, INA219, OLED, user LED, unique motor serials, BMS. `batt` is servo VIN + **3S rest table** (12.6 V = 100%, 10.8 V = 20%, 10.5 V = 10%, 9.0 V = 0%). USB is not a charger. Navbar uses that math from volts (ignores old chip `est=`). Chip `est=` matches after the next flash.

## Firmware flash

**2026-09-12 on the metal:** compact `stream`, move `ESTOP`, `raise to` / `hold` / `free`, solo 12/13 abort, `id N here`, `info fw=20260912 raise=1`. Console header should be cyan `fw 20260912` after hard-refresh.

Still owed: BLE energy advertise + ghost terminate in `ble_nus.c`. Until then, `bluetoothctl disconnect` recovers a stuck radio. Optional: flash so `batt est=` uses the 3S rest table already in `main.c`.

Chip plan: `../roarm-fw/docs/plans/firmware-minimal.md`.

## Try next (this sitting)

1. Hard-refresh. Header: usb, pack %, **LOCK**, **STOP**.
2. `/calibration/`: watch **pos** and **un**. Wrap (yaw / wrist roll): turn through 0/4095 — pos jumps, un and now° keep going after home is set.
3. `/calibration/hand/`: close → click home (closed = 0°) → open → min/max → **test**. Then **next**.
4. Wrap slack is this sitting (not Sep-5 4068). Wrist pitch stay short of ~3134.
5. `/calibration/shoulder-pitch/`: **mirror −** if 12 and 13 count opposite. Click home / min / max as this pose. **test**. Rec/play is optional under details. If they fight: STOP.
6. Chromium `/monitor/`: flash `02-ble-monitor`, unplug USB, pack on. Connect. No commands. Confirm 7/7 VIN ~11 V.

## Later

- Push host `Limit` rows into chip clamps for **non-wrap** IDs (today only hand/twist are hardcoded on chip).
- Wrap-walk so test/sweep can cross 0/4095. Until then wrap test stays off.
- MySQL when mariadb is installed (`docs/run.md`).
- Delete stale root `index.html` / `console.js` / `console.css` / `sw.js` once nobody uses the old PWA.
- Trial graphs: overlay Command marks on Sample traces.
- Phone Chromium can Web Bluetooth later; Firefox stays on this WiFi bridge. Laptop Chromium `/monitor/` is the listen-only path today (`02-ble-monitor`, USB unplugged).

Arm-side order: `../roarm-fw/docs/plans/now.md`.

## Safety

Chip owns ESTOP (flashed). Host **STOP** is backup and queued so it is not starved. No `twist free` during wheel checks. Raise is SyncWrite both, or not at all. Hop: twist ≤900; 11–15 ≤80 until both NVS walls exist. Hand effort trip is page-side `|load|` plus chip `guard`.
