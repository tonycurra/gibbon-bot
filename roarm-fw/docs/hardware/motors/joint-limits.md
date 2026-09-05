# Joint limits

You pose. I read `pos`. I say it back. That is a **zero** or a **stop**.

Each motor: **zero**, how far **−**, how far **+**. Counts from that zero. Soft = 80 inside a stop. I do not command until both stops exist.

`pos` only.

| ID | Joint | zero | − | + |
| --- | --- | --- | --- | --- |
| 11 | turn | | | |
| 12+13 | raise | | | |
| 14 | bend | | | |
| 15 | wrist | | | |
| 16 | twist | | | |
| 17 | hand | **2088** close | **1479** open (wall 609) | **0** (jaws) |

You move one joint. Say **zero** / **minus** / **plus**. I listen and write that cell.
