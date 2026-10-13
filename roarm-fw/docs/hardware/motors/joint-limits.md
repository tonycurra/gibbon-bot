# Joint limits

You pose. I read `pos`. I say it back. That is a **zero** or a **stop**.

Each motor: **zero**, how far **−**, how far **+**. Counts from that zero. Soft = 80 inside a stop. I do not command until both stops exist.

`pos` only.

| ID | Joint | zero | − | + |
| --- | --- | --- | --- | --- |
| 11 | turn | **2084** (host, 11 Sep) | wrap Δ not finished | wrap Δ not finished |
| 12+13 | raise | | | SyncWrite **on chip** (`raise to`, `info raise=1`) |
| 14 | bend | **2347** (host) | **995** | **2997** |
| 15 | wrist | **2091** (host) | **1126** | **3134** (ESTOP on the way — not soft) |
| 16 | twist | Sep-5 slack **4068** / 11 Sep **here 960** | Sep-5 CW **978** | Sep-5 CCW `un` **7126** |
| 17 | hand | **2088** close | **1479** open (wall 609) | **0** (jaws) |

Host notebook (Django `Limit`) is not gold. Gold hand: `docs/baselines/hand-open-close.md`. Gold twist: `docs/baselines/twist.md`. 11 Sep sitting: `development-logs/2026-09-11-calibrate.md`.

You move one joint. Say **zero** / **minus** / **plus**. I listen and write that cell.
