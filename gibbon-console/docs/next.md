# Next

Do not start a new protocol. Parse the text the chip already prints.

## Do not invent

HAT IMU, INA219, OLED, user LED, unique motor serials, BMS. `batt` is servo VIN + 3S rest-% guess. USB is not a charger.

## Firmware flash still owed

Energy advertise + ghost terminate in `ble_nus.c` — flash when ACM0 is in. Until then, host `bluetoothctl disconnect` recovers a stuck radio.

## Productive next

1. Flash BLE energy/ghost. Confirm state C: XT60 on, USB out, page `via ble`, `found=7/7`.
2. MySQL when mariadb is installed (`docs/run.md`).
3. Delete stale root `index.html` / `console.js` / `console.css` / `sw.js` once nobody uses the old PWA.
4. Wrist walls (15), then bend/raise/turn — `roarm-fw/docs/plans/now.md`. Dual-drive 12+13 only as a pair.
5. Graphs: overlay Command marks on Sample traces (trial page).
6. `who` PID/amin/amax already on live row select — use after walls exist.
7. Phone Chromium can Web Bluetooth later; Firefox stays on this WiFi bridge.

## Safety

Chip owns twist ESTOP (`Δload` / current / bus miss). Host **STOP** is backup. No `twist free` during wheel checks. Raise is sync 12+13 or not at all.
