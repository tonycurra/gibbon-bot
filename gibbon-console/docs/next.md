# Next

Do not start a new protocol on the sitting chip. Parse the text it already prints (compact or fat `get`). Packed `g1` is **bring-up only** (`02-ble-monitor` + `/monitor/`).

## Do not invent

HAT IMU, INA219, OLED, user LED, unique motor serials, BMS. `batt` is servo VIN + **3S rest table** (12.6 V = 100%, 10.8 V = 20%, 10.5 V = 10%, 9.0 V = 0%). USB is not a charger. Navbar uses that math from volts (ignores old chip `est=`). Chip `est=` matches after the next flash.

## Firmware flash

**2026-09-12 on the metal:** compact `stream`, move `ESTOP`, `raise to` / `hold` / `free`, solo 12/13 abort, `id N here`, `info fw=20260912 raise=1`. Console header should be cyan `fw 20260912` after hard-refresh.

Still owed: BLE energy advertise + ghost terminate in `ble_nus.c`. Until then, `bluetoothctl disconnect` recovers a stuck radio. Optional: flash so `batt est=` uses the 3S rest table already in `main.c`.

Chip plan: `../roarm-fw/docs/plans/firmware-minimal.md`.

## Try next (this sitting)

Last hours: **do not write firmware.** Four limbs are bolted; two 3S short. Teleop **one** powered limb from this console (phone on WiFi). LOCK / FREE / nudge / STOP. Trip ESTOP once on `/raw/`. Support the body. Arm-side: `../roarm-fw/docs/plans/now.md`.

When that is boring:

1. Hard-refresh. Header: usb, pack %, **LOCK**, **STOP**. Metal **fw 20260912** until you flash sitting `main/` (**20260915** in tree).
2. `/j/<slug>/` or home jog: free → pose → hold → one hop. Wrist short of ~3134. Never 12 or 13 alone.
3. `/calibration/` only if you have wall time: wrap slack is **this** sitting (not Sep-5 4068). Shoulder **mirror −** before any `raise`.
4. Chromium `/monitor/`: flash `02-ble-monitor`, unplug USB, pack on. Connect. No commands. Confirm 7/7 VIN ~11 V. Reflash sitting firmware before any motion.

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
