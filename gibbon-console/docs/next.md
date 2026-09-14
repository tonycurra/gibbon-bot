# Next

Do not start a new protocol. Parse the text the chip already prints (compact or fat `get`).

## Do not invent

HAT IMU, INA219, OLED, user LED, unique motor serials, BMS. `batt` is servo VIN + **linear 3S %** (10.0 V = 0, 12.6 V = 100). USB is not a charger. Navbar uses that math now; chip `est=` matches after the next flash.

## Firmware flash

**2026-09-12 on the metal:** compact `stream`, move `ESTOP`, `raise to` / `hold` / `free`, solo 12/13 abort, `id N here`, `info fw=20260912 raise=1`. Console header should be cyan `fw 20260912` after hard-refresh.

Still owed: BLE energy advertise + ghost terminate in `ble_nus.c`. Until then, `bluetoothctl disconnect` recovers a stuck radio. Optional: flash so `batt est=` uses the linear table already in `main.c`.

Chip plan: `../roarm-fw/docs/plans/firmware-minimal.md`.

## Try next (this sitting)

1. Hard-refresh. Header: usb, pack %, **FREE ALL**, **LOCK POSE**, **STOP**.
2. `/calibration/hand/`: close → **closed = 0°** → open → mark open → **test** (full travel). Then **next wrist roll**.
3. Wrap slack is this sitting (not Sep-5 4068). Wrist pitch stay short of ~3134.
4. Pair on `/calibration/shoulder-pitch/` before go/test. If they fight: STOP.

## Later

- Push host `Limit` rows into chip clamps for **non-wrap** IDs (today only hand/twist are hardcoded on chip).
- Wrap-walk so test/sweep can cross 0/4095. Until then wrap test stays off.
- MySQL when mariadb is installed (`docs/run.md`).
- Delete stale root `index.html` / `console.js` / `console.css` / `sw.js` once nobody uses the old PWA.
- Trial graphs: overlay Command marks on Sample traces.
- Phone Chromium can Web Bluetooth later; Firefox stays on this WiFi bridge.

Arm-side order: `../roarm-fw/docs/plans/now.md`.

## Safety

Chip owns ESTOP (flashed). Host **STOP** is backup and queued so it is not starved. No `twist free` during wheel checks. Raise is SyncWrite both, or not at all. Hop: twist ≤900; 11–15 ≤80 until both NVS walls exist. Hand effort trip is page-side `|load|` plus chip `guard`.
