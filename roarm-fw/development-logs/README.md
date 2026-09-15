# Development logs

Notes of what we tried, named by the test. Durable how-tos live in `docs/`.

| Log | What |
| --- | --- |
| [usb-hello.md](usb-hello.md) | Naming, IDF pick, USB hello, LED dead end, info+ping |
| [2026-09-09-console.md](2026-09-09-console.md) | Six-joint console, compact `get`/`stream` (then unflashed) |
| [2026-09-11-calibrate.md](2026-09-11-calibrate.md) | Wrap Δ calibrate, host unwrap, UART send queue (raise still unflashed that day) |
| [2026-09-12-flash.md](2026-09-12-flash.md) | Flashed ESTOP + raise SyncWrite + compact stream + `info fw=20260912` |
| [2026-09-15-packs.md](2026-09-15-packs.md) | Two 3S short, 2S in the pile, storage vs regen. Rules: `docs/hardware/lipo.md` |
| [listen-xt60.md](listen-xt60.md) | Listen, XT60, `get`/`batt`, power cycle |
| [hand-move.md](hand-move.md) | Backdrive, hand walls, commanded hand, contact CSV |
| [espidf-start.md](espidf-start.md) | Once-on-this-machine. **Every time:** `docs/framework/espidf.md` |

Every-start (pack off): `docs/baselines/usb-start.md`.  
Pack on: `docs/baselines/usb-battery-on.md`.  
Ping gold: `docs/baselines/usb-no-battery.md`.  
Hand-move: `docs/baselines/hand-move-usb-battery-on.md`.  
Hand open/close: `docs/baselines/hand-open-close.md`.

Stock RoArm UI / homes: `docs/old docs to review/controls.md`.  
Motor notes: `docs/hardware/motors/` and `docs/old docs to review/motors.md`.
