# Baseline: listen, USB only, XT60 unplugged

**Every-start checklist (this capture + `info` + identity):** [`usb-start.md`](usb-start.md).

2026-09-04. UART switch **ESP32**. `/dev/ttyACM0`. Firmware `listen v1` (no ping on boot, no move).

Compare chip/MAC to `usb-no-battery.md`. Do not ping in this window.

## Must match

| Field | Gold |
| --- | --- |
| `mac=` | `44:1d:64:86:3f:54` |
| `reset=` | `poweron` (may change) |
| `bus_rx=` | `0` |
| `dump=` | `1` |
| bus hex lines | none |
| `gpio 18(rx)` / `19(tx)` | `1` / `1` (UART idle) |

Heap ~298000. Five tasks: `main`, `IDLE0`, `IDLE1`, `ipc0`, `ipc1`.

## Capture (`state` + heartbeats)

```
state t=37780 heap=297940 reset=poweron tasks=5 bus_rx=0 dump=1
state mac=44:1d:64:86:3f:54
state bus=uart1 gpio18/19 1Mbps rx-only until ping
tasks:
main           	X	1	1880	3
IDLE1          	R	0	1024	5
IDLE0          	R	0	1028	4
ipc0           	S	24	504	1
ipc1           	S	24	496	2
gpio 18(rx)=1 19(tx)=1 (uart1, not remuxed)
t=24010 … t=64020 heap=297940 bus_rx=0
```

No `bus t=… hex=` lines. Servos unpowered, and ST nodes do not broadcast anyway.
