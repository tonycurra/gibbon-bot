# Data (trials)

A **trial** is one sitting: title, notes, commands you sent, `get` rows the motors sent back.

Start on `/trials/` if you want a named tape. Otherwise the first live `get` still auto-opens a `Trial` titled `sitting YYYY-MM-DD HH:MM`. That is a log, not calibration — the header does not show **rec sitting**. Stop the trial on `/trials/` when you care.

## Models (`console/models.py`)

| | |
| --- | --- |
| **Trial** | started, ended, title, note, via (`usb`/`ble`), pack_v |
| **Command** | t_ms, text, source=`ui` — skip auto `info`/`state`/`bench`/`batt`/`who`/`stream` |
| **Sample** | t_ms, servo 11–17, pos, un, turns, speed, load, v, temp, cur, moving, tq, mode |
| **Note** | t_ms, text |
| **Limit** | servo 11–17, home, amin, amax — live notebook |
| **Calibration** | dated snapshot: board, fw, via, joints JSON |

Ingest: `bridge.db_loop` from the same RX lines the page parses. Miss-spam with pack off is not stored. Pack V set from first `batt pack=…V`.

Index: `(trial, servo, t_ms)`. Cap read API 8000 samples.

## Graphs

Live traces: `/raw/` (hand) and `/j/<slug>/` (that joint), ring buffer from SSE `get` lines.  
Recorded `/trials/<id>/`: `GET /api/trials/<id>/samples?servo=`. `/graphs/` redirects to `/raw/`.

Traces: pos@56, un, speed@58, load@60, cur@69, V@62, temp@63. Load is PWM effort, 0 if `tq=0`.

## Files

- SQLite: `db.sqlite3` (gitignored)
- Old line dumps: `sittings/*.jsonl` (pre-Django)
- BLE addr cache: `.ble-addr`

Django admin can browse rows after `createsuperuser`.
