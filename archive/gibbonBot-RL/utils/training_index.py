"""
Training-artifacts indexer — walks runs/ape_*_mjx/ and upserts every
on-disk training artifact into runs/ape_policy_db.sqlite so all of it
becomes queryable.

Reads (read-only, never modifies the source files):
  - <run_dir>/lineage.json                 → lineage rows + runs.inherited_from
  - <run_dir>/latest.fingerprint.json      → fingerprints + runs.fingerprint_hash
  - <run_dir>/autopilot_state.json         → runs.autopilot_phase
  - <run_dir>/autopilot.log                → autopilot_events
  - <run_dir>/train.log                    → log_events (METRIC + KEEPER + STOP lines)
  - <run_dir>/mjx_*.npz (+ .safetensors)   → checkpoints
  - <run_dir>/keepers/auto_*.json          → keeper checkpoints (sidecar metrics)
  - <run_dir>/_history/<ts>_<hash>/...     → archived checkpoints (archived=1)

Writes (idempotent):
  - runs, checkpoints, lineage, fingerprints, log_events, autopilot_events
    in runs/ape_policy_db.sqlite (extended PolicyDatabase schema).

CLI: see scripts/reindex.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from brain.policy_database import PolicyDatabase

# Matches the tabular metric line emitted in train_mjx.py's training loop
# header: "{updates:>8}  {steps:>12}  {sps:>8}  {mean_rew:>10}  metrics"
# Followed by lines like:
#   "      42        688,128       1234  rew=+0.12  d=1.20  vx=+0.30  z=0.45  upr=0.92  ωz=+0.01"
# We only need: updates, total_steps, sps, mean_rew. Use the existing
# METRIC_RE shape from utils/autopilot.py for compatibility.
_METRIC_RE = re.compile(
    r"^\s*(\d+)\s+([\d,]+)\s+([\d,]+)\s+rew=([+-]?[\d.]+)\b"
)
_KEEPER_RE = re.compile(r"^\[keeper\]\s+saved\s+(\S+)")
_STOP_RE = re.compile(r"^\[stop\]\s+(\S+)\s+—\s+saving at\s+([\d,]+)\s+steps")
_BRAIN_RE = re.compile(r"^\[brain\]\s+(.+)$")
_AUTOPILOT_PHASE_RE = re.compile(r"\[autopilot\]\s+phase\s+→\s+([A-Z]):")
_AUTOPILOT_FALLBACK_RE = re.compile(r"\[autopilot\]\s+(fallback|stall)\b")


def task_from_dir(run_dir: Path) -> str:
    name = run_dir.name
    # ape_<task>_mjx
    if name.startswith("ape_") and name.endswith("_mjx"):
        return name[4:-4]
    if name.startswith("ape_"):
        return name[4:]
    return name


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _step_from_npz(p: Path) -> Optional[int]:
    # mjx_000000032768.npz → 32768
    try:
        return int(p.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _ts_to_epoch(ts: str) -> Optional[float]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def index_run(run_dir: Path, db: PolicyDatabase, scene_path: str = "data/scene.xml") -> int:
    """Index one runs/ape_<task>_mjx/ tree. Returns runs.id."""
    run_dir = Path(run_dir)
    task = task_from_dir(run_dir)

    # ---- lineage / inheritance ----
    lineage = _read_json(run_dir / "lineage.json") or []
    inherited = lineage[-1].get("inherited_from") if lineage else None
    started_at = _ts_to_epoch(lineage[0]["timestamp"]) if lineage else None
    ended_at = _ts_to_epoch(lineage[-1]["timestamp"]) if lineage else None

    # ---- fingerprint ----
    fp = _read_json(run_dir / "latest.fingerprint.json") or {}
    fp_hash = fp.get("hash")
    fp_parts = fp.get("parts") or {}
    fp_ts = _ts_to_epoch(fp.get("ts") or "") or ended_at or 0.0

    # ---- autopilot phase ----
    ap_state = _read_json(run_dir / "autopilot_state.json") or {}
    phase = ap_state.get("phase")

    # ---- scene_hash (best-effort) ----
    scene_hash = None
    try:
        if Path(scene_path).exists():
            scene_hash = PolicyDatabase.compute_scene_hash(scene_path)
    except Exception:
        pass

    run_id = db.upsert_run(
        run_dir=str(run_dir), task=task,
        scene_hash=scene_hash, fingerprint_hash=fp_hash,
        started_at=started_at, ended_at=ended_at,
        inherited_from=inherited, autopilot_phase=phase,
    )

    if fp_hash and fp_parts:
        db.upsert_fingerprint(fp_hash, fp_parts, fp_ts)

    for i, entry in enumerate(lineage):
        db.upsert_lineage_entry(run_id, task, i, entry)

    # ---- live checkpoints ----
    for npz in sorted(run_dir.glob("mjx_*.npz")):
        step = _step_from_npz(npz)
        if step is None:
            continue
        st = npz.with_suffix(".safetensors")
        db.upsert_checkpoint(
            path=str(npz), task=task, step_count=step,
            run_id=run_id, scene_hash=scene_hash,
            safetensors_path=str(st) if st.exists() else None,
            is_keeper=False, fingerprint_hash=fp_hash,
            wall_time=npz.stat().st_mtime,
        )

    # ---- keepers ----
    keepers_dir = run_dir / "keepers"
    if keepers_dir.is_dir():
        for sidecar in keepers_dir.glob("auto_*.json"):
            data = _read_json(sidecar) or {}
            npz = sidecar.with_suffix(".npz")
            step = int(data.get("step", _step_from_npz(npz) or 0))
            metrics = data.get("metrics") or {}
            db.upsert_checkpoint(
                path=str(npz), task=task, step_count=step,
                run_id=run_id, scene_hash=scene_hash,
                is_keeper=True, keeper_tag=data.get("tag"),
                mean_reward=metrics.get("mean_rew"),
                metrics=metrics, fingerprint_hash=fp_hash,
                wall_time=sidecar.stat().st_mtime,
            )

    # ---- archived checkpoints under _history/<ts>_<hash>/ ----
    history = run_dir / "_history"
    if history.is_dir():
        for sub in history.iterdir():
            if not sub.is_dir():
                continue
            for npz in sub.glob("mjx_*.npz"):
                step = _step_from_npz(npz)
                if step is None:
                    continue
                db.upsert_checkpoint(
                    path=str(npz), task=task, step_count=step,
                    run_id=run_id, scene_hash=scene_hash,
                    is_keeper=False, archived=True,
                    fingerprint_hash=sub.name.split("_", 1)[-1],
                    wall_time=npz.stat().st_mtime,
                )

    # ---- structured event log (preferred) ----
    events_jsonl = run_dir / "events.jsonl"
    if events_jsonl.exists():
        _index_events_jsonl(events_jsonl, db, run_id)
    else:
        # Fallback: parse train.log
        _index_train_log(run_dir / "train.log", db, run_id)

    # ---- autopilot.log ----
    _index_autopilot_log(run_dir / "autopilot.log", db, run_id, task)

    return run_id


def _index_events_jsonl(path: Path, db: PolicyDatabase, run_id: int) -> None:
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                db.insert_log_event(
                    run_id=run_id,
                    wall_time=float(rec.get("wall_time") or 0.0),
                    level=rec.get("level") or "INFO",
                    source="events.jsonl",
                    message=rec.get("message") or "",
                    update_count=rec.get("update_count"),
                    total_steps=rec.get("total_steps"),
                    metrics=rec.get("metrics"),
                )
    except OSError:
        pass


def _index_train_log(path: Path, db: PolicyDatabase, run_id: int) -> None:
    if not path.exists():
        return
    base_mtime = path.stat().st_mtime
    try:
        with path.open(errors="replace") as f:
            for offset, line in enumerate(f):
                line = line.rstrip("\n")
                # Synthetic monotonic wall_time so UNIQUE(run_id, source, wall_time, message)
                # accepts every line without colliding (real wall_time isn't in the log).
                wt = base_mtime - 1e-3 * (1_000_000 - offset)

                m = _METRIC_RE.match(line)
                if m:
                    upd = int(m.group(1))
                    steps = int(m.group(2).replace(",", ""))
                    sps = int(m.group(3).replace(",", ""))
                    mean_rew = float(m.group(4))
                    db.insert_log_event(
                        run_id=run_id, wall_time=wt, level="METRIC",
                        source="train.log", update_count=upd, total_steps=steps,
                        metrics={"sps": sps, "mean_rew": mean_rew}, message=line,
                    )
                    continue

                m = _KEEPER_RE.match(line)
                if m:
                    db.insert_log_event(
                        run_id=run_id, wall_time=wt, level="KEEPER",
                        source="train.log", message=line,
                    )
                    continue

                m = _STOP_RE.match(line)
                if m:
                    steps = int(m.group(2).replace(",", ""))
                    db.insert_log_event(
                        run_id=run_id, wall_time=wt, level="STOP",
                        source="train.log", total_steps=steps, message=line,
                    )
                    continue

                m = _BRAIN_RE.match(line)
                if m:
                    db.insert_log_event(
                        run_id=run_id, wall_time=wt, level="BRAIN",
                        source="train.log", message=line,
                    )
    except OSError:
        pass


def _index_autopilot_log(path: Path, db: PolicyDatabase, run_id: int, task: str) -> None:
    if not path.exists():
        return
    base_mtime = path.stat().st_mtime
    try:
        with path.open(errors="replace") as f:
            for offset, line in enumerate(f):
                line = line.rstrip("\n")
                wt = base_mtime - 1e-3 * (1_000_000 - offset)
                m = _AUTOPILOT_PHASE_RE.search(line)
                if m:
                    db.insert_autopilot_event(
                        run_id=run_id, task=task, wall_time=wt,
                        to_phase=m.group(1), reason=line,
                    )
                    db.insert_log_event(
                        run_id=run_id, wall_time=wt, level="AUTOPILOT",
                        source="autopilot.log", message=line,
                    )
                    continue
                m = _AUTOPILOT_FALLBACK_RE.search(line)
                if m:
                    # to_phase placeholder so the (run_id, wall_time, to_phase)
                    # UNIQUE key dedupes (NULLs are distinct in SQLite UNIQUE).
                    db.insert_autopilot_event(
                        run_id=run_id, task=task, wall_time=wt,
                        to_phase="_fallback", fallback=True, reason=line,
                    )
                    db.insert_log_event(
                        run_id=run_id, wall_time=wt, level="AUTOPILOT",
                        source="autopilot.log", message=line,
                    )
    except OSError:
        pass


def index_all(runs_root: str = "runs",
              db_path: str = "runs/ape_policy_db.sqlite",
              scene_path: str = "data/scene.xml") -> dict:
    """Walk every ape_*_mjx/ under runs_root and index it. Returns per-task counts."""
    root = Path(runs_root)
    db = PolicyDatabase(db_path)
    counts: dict = {}
    try:
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("ape_"):
                continue
            try:
                run_id = index_run(sub, db, scene_path=scene_path)
                counts[sub.name] = run_id
            except Exception as e:
                print(f"[index] {sub.name}: failed: {e}")
    finally:
        db.close()
    return counts


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "runs"
    if Path(target).is_dir() and not (Path(target) / "lineage.json").exists():
        result = index_all(target)
        print(f"indexed {len(result)} runs:", list(result.keys()))
    else:
        db = PolicyDatabase("runs/ape_policy_db.sqlite")
        rid = index_run(Path(target), db)
        db.close()
        print(f"indexed {target} → run_id={rid}")
