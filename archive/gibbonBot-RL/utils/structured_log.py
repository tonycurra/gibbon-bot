"""
Structured per-run event log — one JSON object per line in events.jsonl,
written next to train.log under the run directory.

The training-artifacts indexer (`utils/training_index.py`) prefers this
file when present; otherwise it falls back to regex-parsing train.log.
Use this for any event you care about querying later: keeper saves,
brain consolidations, autopilot phase changes, scene/policy switches.

Schema per line:
    {
      "wall_time": 1781234567.89,
      "level": "KEEPER",       # INFO / METRIC / KEEPER / BRAIN / AUTOPILOT / STOP
      "message": "saved auto_step123.npz",
      "update_count": 42,      # optional
      "total_steps": 688128,   # optional
      "metrics": {...}         # optional dict
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def log_event(run_dir: str | Path, level: str, message: str = "", *,
              update_count: Optional[int] = None,
              total_steps: Optional[int] = None,
              metrics: Optional[Dict[str, Any]] = None,
              wall_time: Optional[float] = None) -> None:
    """Append one event to <run_dir>/events.jsonl. Never raises on I/O error."""
    rec: Dict[str, Any] = {
        "wall_time": wall_time if wall_time is not None else time.time(),
        "level": level,
        "message": message,
    }
    if update_count is not None:
        rec["update_count"] = int(update_count)
    if total_steps is not None:
        rec["total_steps"] = int(total_steps)
    if metrics is not None:
        rec["metrics"] = metrics
    try:
        path = Path(run_dir) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError:
        pass
