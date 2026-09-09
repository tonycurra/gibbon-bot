# BLE (NimBLE Nordic UART)

Same text as USB. Advertise `gibbon` + NUS UUID; scan response `gibbon-XXXX`. This board: name `gibbon-3f54`, BLE addr `44:1D:64:86:3F:56`, STA MAC `44:1d:64:86:3f:54`.

Host (laptop central, Firefox): `../../gibbon-console/docs/link.md`.

## Energy / ghost (flash this)

`main/ble_nus.c`

- Advertise interval ~160–320 ms (not 20–40 ms).
- If GAP connected but notify never subscribed for 5 s → `ble_gap_terminate` (`ble_nus_tick` from `app_main`).
- Conn params: itvl 24–40, latency 2, supervision 2 s.
- `CONFIG_BT_NIMBLE_MAX_CONNECTIONS=1`.
- No pairing (`sm_io_cap` no-io, no bonding).

Unflashed: laptop still recovers by BlueZ disconnect + direct address connect. Power-cycling XT60 also works; do not rely on that.

Do not start WiFi on this WROOM-32 while BLE is on.
