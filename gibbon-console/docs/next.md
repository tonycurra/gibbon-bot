# Next

Do not start a new protocol. Parse the text the chip already prints (compact or fat `get`).

## Do not invent

HAT IMU, INA219, OLED, user LED, unique motor serials, BMS. `batt` is servo VIN + 3S rest-% guess. USB is not a charger.

## Firmware flash still owed (ACM0)

Stop Django first. One flash should cover:

1. Compact `stream` (omit zeros, skip still IDs) — source in `roarm-fw/main/main.c`.
2. Twist ESTOP (`Δload` / cur / bus miss) if not already on the chip.
3. BLE energy advertise + ghost terminate in `ble_nus.c`. Until then, `bluetoothctl disconnect` recovers a stuck radio.

Confirm after flash: `stream` quiet until you move; `found=7/7`; pack V in the header.

## Productive next

1. Wrist pitch walls (15) by hand → `/calibrate/?j=wrist-pitch` → slow `to` inside the band. Then elbow, shoulder pitch (12+13 together), yaw. Dual-drive: never command 12 or 13 alone. SyncWrite still owed.
2. Push host `Limit` rows into chip clamps once a wall exists (today only hand/twist are hardcoded).
3. Shoulder pitch: opposite-sign / SyncWrite so nudge does not fight.
4. MySQL when mariadb is installed (`docs/run.md`).
5. Delete stale root `index.html` / `console.js` / `console.css` / `sw.js` once nobody uses the old PWA.
6. Trial graphs: overlay Command marks on Sample traces.
7. Phone Chromium can Web Bluetooth later; Firefox stays on this WiFi bridge.

Arm-side order: `../roarm-fw/docs/plans/now.md`.

## Safety

Chip owns twist ESTOP. Host **STOP** is backup. No `twist free` during wheel checks. Raise is 12+13 together or not at all. No twist command until ESTOP is on the chip.
