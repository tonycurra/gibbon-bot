#!/usr/bin/env python3
"""Single-page dashboard for gibbonBot training.

Composes existing scripts — does NOT replace them. Buttons shell out to
train_bg.sh, remote.sh, watch_remote.sh, cloud_push.sh. The page polls a
few JSON endpoints every few seconds and renders log tail + metrics.

Run from repo root:
    .venv/bin/python scripts/dashboard.py
    # → http://localhost:8765

Stop with Ctrl-C. Footprint: ~30 MB RAM, single uvicorn worker.
"""
from __future__ import annotations

import json
import re
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TRAIN_LOG = ROOT / "train.log"
REMOTE_LOG_GLOB = "ape_*_mjx/train.log"
WATCH_PID_FILE = ROOT / "runs" / ".watch_remote.pid"
CLOUD_PID_FILE = ROOT / "runs" / ".cloud_push.pid"
MLFLOW_UI_PID_FILE = ROOT / "runs" / ".mlflow_ui.pid"
MLFLOW_UI_PORT = int(os.environ.get("MLFLOW_UI_PORT", "5000"))

# Source of truth for "what skills exist + what's the success bar?" — kept in
# sync with scripts/curriculum.py TASKS table. (task, prereq, metric, op, threshold)
SKILLS: list[tuple[str, Optional[str], str, str, float]] = [
    ("stand",        None,           "mean_upright",  "gt",  0.90),
    ("walk",         None,           "mean_rew",      "gt",  5.0),     # legacy walk-to-tree
    ("walk_forward", "stand",        "mean_speed_x",  "gt",  0.20),
    ("turn_left",    "walk_forward", "mean_angvel_z", "gt",  0.40),
    ("turn_right",   "turn_left",    "mean_angvel_z", "lt", -0.40),
]
CKPT_RE = re.compile(r"mjx_(\d+)\.npz$")

# Matches train_mjx.py per-update lines: "  123  4,096  12,345   0.123   2.901"
METRIC_RE = re.compile(r"^\s*(\d+)\s+([\d,]+)\s+([\d,]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")

app = FastAPI()


def _tail(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        return data.decode("utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _parse_metrics(lines: list[str], limit: int = 60) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        m = METRIC_RE.match(line)
        if not m:
            continue
        out.append({
            "update": int(m.group(1)),
            "env_steps": int(m.group(2).replace(",", "")),
            "sps": int(m.group(3).replace(",", "")),
            "loss": float(m.group(4)),
            "reward": float(m.group(5)),
        })
    return out[-limit:]


def _newest_ckpt() -> Optional[dict]:
    candidates = list(RUNS.glob("**/mjx_*.npz")) + list(RUNS.glob("**/*.safetensors"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    st = newest.stat()
    return {
        "path": str(newest.relative_to(ROOT)),
        "size_mb": round(st.st_size / 1024 / 1024, 2),
        "mtime": st.st_mtime,
        "age_s": round(time.time() - st.st_mtime, 1),
    }


def _pid_alive(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def _trainer_pids() -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", "[t]rain_mjx.py"], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _mlflow_db_path() -> Optional[Path]:
    """Prefer the H100-synced DB; fall back to local."""
    for db_path in (ROOT / "mlflow.db.remote", ROOT / "mlflow.db"):
        if db_path.exists():
            return db_path
    return None


def _mlflow_runs(task: Optional[str] = None, limit: int = 40) -> list[dict]:
    """Return the most recent runs across all experiments (or filtered to one
    task's experiment), with the latest value of each interesting metric.
    Used by the runs-explorer UI for sorting + comparison."""
    db = _mlflow_db_path()
    if db is None:
        return []
    # Keys we want to surface in the runs table. Order matters — first column
    # after run_id is the "headline" sort metric.
    METRIC_KEYS = [
        "mean_rew", "min_dist", "mean_upright", "mean_speed_x",
        "mean_angvel_z", "success_frac", "gait_coherence", "checkpoint_steps",
    ]
    rows: list[dict] = []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        cur = con.cursor()
        if task is not None:
            cur.execute(
                "SELECT r.run_uuid, r.name, r.status, r.start_time, r.end_time, e.name "
                "FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
                "WHERE r.lifecycle_stage = 'active' AND e.name = ? "
                "ORDER BY r.start_time DESC LIMIT ?",
                (f"ape_{task}_mjx", limit),
            )
        else:
            cur.execute(
                "SELECT r.run_uuid, r.name, r.status, r.start_time, r.end_time, e.name "
                "FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
                "WHERE r.lifecycle_stage = 'active' AND e.name LIKE 'ape_%' "
                "ORDER BY r.start_time DESC LIMIT ?",
                (limit,),
            )
        run_rows = cur.fetchall()
        for run_uuid, name, status, start_ms, end_ms, exp_name in run_rows:
            entry = {
                "run_id": run_uuid,
                "short_id": run_uuid[:8],
                "name": name or "(unnamed)",
                "status": status,
                "experiment": exp_name,
                "start_time": (start_ms or 0) / 1000.0,
                "end_time": (end_ms or 0) / 1000.0 if end_ms else None,
                "metrics": {},
            }
            # Pull latest values for each interesting key in one query.
            qmarks = ",".join("?" for _ in METRIC_KEYS)
            cur.execute(
                f"SELECT key, value FROM latest_metrics WHERE run_uuid = ? AND key IN ({qmarks})",
                (run_uuid, *METRIC_KEYS),
            )
            for k, v in cur.fetchall():
                entry["metrics"][k] = float(v) if v is not None else None
            rows.append(entry)
        con.close()
    except sqlite3.Error as e:
        return [{"error": str(e)}]
    return rows


def _mlflow_run_history(run_id: str, metric: str, max_points: int = 400) -> list[list[float]]:
    """Return [[step, value], ...] for a metric within a run, downsampled to
    max_points by simple stride sampling. Cheap enough at 400 pts to render
    inline SVG curves for several runs at once."""
    db = _mlflow_db_path()
    if db is None:
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        cur = con.cursor()
        cur.execute(
            "SELECT step, value FROM metrics WHERE run_uuid = ? AND key = ? AND is_nan = 0 ORDER BY step ASC",
            (run_id, metric),
        )
        rows = cur.fetchall()
        con.close()
    except sqlite3.Error:
        return []
    if len(rows) <= max_points:
        return [[int(s), float(v)] for s, v in rows]
    stride = max(1, len(rows) // max_points)
    return [[int(rows[i][0]), float(rows[i][1])] for i in range(0, len(rows), stride)]


def _mlflow_latest_metric(task: str, metric: str) -> Optional[float]:
    """Most-recent value of `metric` for the experiment matching this task."""
    db = _mlflow_db_path()
    if db is None:
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        cur = con.cursor()
        cur.execute(
            "SELECT m.value FROM metrics m "
            "JOIN runs r ON r.run_uuid = m.run_uuid "
            "JOIN experiments e ON e.experiment_id = r.experiment_id "
            "WHERE e.name = ? AND m.key = ? "
            "ORDER BY m.timestamp DESC LIMIT 1",
            (f"ape_{task}_mjx", metric),
        )
        row = cur.fetchone()
        con.close()
        return float(row[0]) if row else None
    except sqlite3.Error:
        return None


def _skill_status(task: str, metric: str, op: str, threshold: float) -> dict:
    run_dir = RUNS / f"ape_{task}_mjx"
    if not run_dir.exists():
        return {"task": task, "state": "untrained", "metric": metric, "threshold": threshold}

    ckpts = list(run_dir.glob("mjx_*.npz"))
    if not ckpts:
        return {"task": task, "state": "started_no_ckpt", "metric": metric, "threshold": threshold}

    newest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
    m = CKPT_RE.search(newest_ckpt.name)
    steps = int(m.group(1)) if m else 0

    last_value: Optional[float] = None
    source: Optional[str] = None

    # Source priority: MLflow remote DB (richest, freshest) → keepers sidecar
    # → live_stats.json → train.log fallback.
    mlf = _mlflow_latest_metric(task, metric)
    if mlf is not None:
        last_value = mlf
        source = "mlflow"

    # Per-task sidecars in keepers/.
    if last_value is None:
        keepers = run_dir / "keepers"
        if keepers.exists():
            sidecars = sorted(keepers.glob("auto_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for sc in sidecars[:5]:
                try:
                    metrics = json.loads(sc.read_text()).get("metrics", {})
                    if metric in metrics:
                        last_value = float(metrics[metric])
                        source = "keeper"
                        break
                except (OSError, ValueError):
                    continue

    # live_stats.json (some tasks write this).
    if last_value is None:
        live = run_dir / "live_stats.json"
        if live.exists():
            try:
                data = json.loads(live.read_text())
                if metric in data:
                    last_value = float(data[metric])
                    source = "live_stats"
            except (OSError, ValueError):
                pass

    # Last resort: parse last reward from train.log.
    if last_value is None and metric == "mean_rew":
        for line in reversed(_tail(run_dir / "train.log", n=400)):
            mm = METRIC_RE.match(line)
            if mm:
                last_value = float(mm.group(5))
                source = "log"
                break

    pass_check: Optional[bool] = None
    confidence: Optional[float] = None
    if last_value is not None:
        if op == "gt":
            pass_check = last_value > threshold
            confidence = max(0.0, min(1.0, last_value / threshold)) if threshold > 0 else None
        else:  # lt — used for negative thresholds (e.g. turn_right angvel)
            pass_check = last_value < threshold
            confidence = max(0.0, min(1.0, last_value / threshold)) if threshold < 0 else None

    log_mtime = (run_dir / "train.log").stat().st_mtime if (run_dir / "train.log").exists() else newest_ckpt.stat().st_mtime
    return {
        "task": task,
        "state": "trained" if pass_check else ("training" if (time.time() - log_mtime) < 600 else "stalled"),
        "metric": metric,
        "threshold": threshold,
        "op": op,
        "value": last_value,
        "value_source": source,
        "confidence": confidence,
        "passes": pass_check,
        "steps": steps,
        "ckpts": len(ckpts),
        "ckpt": newest_ckpt.name,
        "age_s": round(time.time() - newest_ckpt.stat().st_mtime, 1),
    }


def _bash(cmd: list[str], timeout: int = 20) -> dict:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return {"ok": r.returncode == 0, "code": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": -1, "stdout": "", "stderr": f"timeout after {timeout}s"}


@app.get("/api/status")
def api_status() -> JSONResponse:
    # Prefer the most recently modified train.log under runs/, fall back to root train.log.
    log_candidates = sorted(RUNS.glob(REMOTE_LOG_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    log_path = log_candidates[0] if log_candidates else TRAIN_LOG
    lines = _tail(log_path, n=200)
    metrics = _parse_metrics(lines)

    return JSONResponse({
        "now": time.time(),
        "log_path": str(log_path.relative_to(ROOT)) if log_path.exists() else None,
        "log_tail": lines[-40:],
        "metrics": metrics,
        "ckpt": _newest_ckpt(),
        "trainer_pids": _trainer_pids(),
        "watch_remote_pid": _pid_alive(WATCH_PID_FILE),
        "cloud_push_pid": _pid_alive(CLOUD_PID_FILE),
    })


@app.get("/api/runs")
def api_runs(task: Optional[str] = None, limit: int = 40) -> JSONResponse:
    return JSONResponse({"runs": _mlflow_runs(task=task, limit=limit)})


@app.get("/api/run_history")
def api_run_history(run_id: str, metric: str = "mean_rew") -> JSONResponse:
    return JSONResponse({"run_id": run_id, "metric": metric, "points": _mlflow_run_history(run_id, metric)})


@app.get("/api/skills")
def api_skills() -> JSONResponse:
    out = []
    for task, prereq, metric, op, thr in SKILLS:
        s = _skill_status(task, metric, op, thr)
        s["prereq"] = prereq
        out.append(s)
    # "next to train" = first untrained skill whose prereq is trained (or has no prereq).
    trained = {s["task"] for s in out if s.get("passes")}
    next_task = None
    for s in out:
        if s.get("passes"):
            continue
        if s.get("prereq") is None or s["prereq"] in trained:
            next_task = s["task"]
            break
    return JSONResponse({"skills": out, "next_to_train": next_task})


def _remote_host() -> Optional[str]:
    """Parse the REMOTE=... line from scripts/remote.sh so the dashboard
    always pings the same host the operator scripts target."""
    try:
        for line in (ROOT / "scripts/remote.sh").read_text().splitlines():
            if line.startswith("REMOTE="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


@app.get("/api/remote/health")
def api_remote_health() -> JSONResponse:
    host = _remote_host()
    if not host:
        return JSONResponse({"ok": False, "error": "no REMOTE= in scripts/remote.sh"}, status_code=500)

    # One ssh round-trip: fetch health.json + nvidia-smi in a single shell.
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=4",
        "-o", "BatchMode=yes",                # never prompt for password
        "-o", "StrictHostKeyChecking=accept-new",
        host,
        "cat gibbonBot-RL/runs/ape_walk_mjx/health.json 2>/dev/null; "
        "echo '---SMI---'; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits 2>/dev/null",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "ssh timeout", "host": host}, status_code=504)

    if r.returncode != 0:
        return JSONResponse({"ok": False, "error": r.stderr.strip()[:300] or "ssh failed", "host": host}, status_code=502)

    parts = r.stdout.split("---SMI---", 1)
    health_raw = parts[0].strip()
    smi_raw = parts[1].strip() if len(parts) > 1 else ""

    health: dict = {}
    if health_raw:
        try:
            health = json.loads(health_raw)
        except ValueError:
            health = {"raw": health_raw[:200]}

    gpu: dict = {}
    if smi_raw:
        # First line is the GPU we care about (single-GPU box).
        first = smi_raw.splitlines()[0]
        bits = [b.strip() for b in first.split(",")]
        if len(bits) >= 5:
            try:
                gpu = {
                    "util_pct": float(bits[0]),
                    "mem_used_mb": float(bits[1]),
                    "mem_total_mb": float(bits[2]),
                    "temp_c": float(bits[3]),
                    "power_w": float(bits[4]),
                }
            except ValueError:
                gpu = {"raw": first}

    return JSONResponse({"ok": True, "host": host, "health": health, "gpu": gpu, "ts": time.time()})


@app.get("/api/processes")
def api_processes() -> JSONResponse:
    # Top processes by RSS — covers the "what's eating my RAM" question without
    # needing psutil. ps output is portable.
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,pmem,rss,comm,args", "--sort=-rss"],
            text=True, timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return JSONResponse({"error": str(e), "rows": []}, status_code=500)

    lines = out.splitlines()
    rows = []
    for ln in lines[1:60]:  # top 60 by RSS, header dropped
        parts = ln.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[0]); pcpu = float(parts[1]); pmem = float(parts[2]); rss_kb = int(parts[3])
        except ValueError:
            continue
        comm = parts[4]
        args = parts[5][:140]
        # Annotate gibbonbot-related rows so the UI can highlight them.
        tag = None
        if "train_mjx" in args or "train_walk_mjx" in args:
            tag = "trainer"
        elif "/claude" in args or "claude-agent" in args:
            tag = "claude"
        elif "uvicorn" in args or "dashboard.py" in args:
            tag = "dashboard"
        elif "mlflow" in args:
            tag = "mlflow"
        elif "ollama" in args:
            tag = "ollama"
        elif "dockerd" in args or "containerd" in args:
            tag = "docker"
        elif "rsync" in args or "watch_remote" in args:
            tag = "sync"
        rows.append({"pid": pid, "pcpu": pcpu, "pmem": pmem, "rss_mb": rss_kb // 1024, "comm": comm, "args": args, "tag": tag})

    return JSONResponse({"rows": rows[:25]})


@app.post("/api/train/start")
def api_train_start(task: str = "walk") -> dict:
    # Whitelist task names against SKILLS — never let the browser pass arbitrary args to bash.
    valid = {t for t, *_ in SKILLS}
    if task not in valid:
        raise HTTPException(400, f"unknown task '{task}' — valid: {sorted(valid)}")
    return _bash([str(ROOT / "scripts/train_bg.sh"), "start", task])


@app.post("/api/train/stop")
def api_train_stop() -> dict:
    return _bash([str(ROOT / "scripts/train_bg.sh"), "stop"])


@app.post("/api/remote/start")
def api_remote_start() -> dict:
    return _bash([str(ROOT / "scripts/remote.sh"), "start"], timeout=60)


@app.post("/api/remote/stop")
def api_remote_stop() -> dict:
    return _bash([str(ROOT / "scripts/remote.sh"), "stop"], timeout=30)


@app.post("/api/sync/start")
def api_sync_start() -> dict:
    if _pid_alive(WATCH_PID_FILE):
        return {"ok": False, "stderr": "watch_remote already running"}
    WATCH_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = ROOT / "runs" / "watch_remote.log"
    proc = subprocess.Popen(
        [str(ROOT / "scripts/watch_remote.sh")],
        cwd=ROOT,
        stdout=log.open("ab"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    WATCH_PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid, "log": str(log.relative_to(ROOT))}


@app.post("/api/sync/stop")
def api_sync_stop() -> dict:
    pid = _pid_alive(WATCH_PID_FILE)
    if not pid:
        return {"ok": False, "stderr": "watch_remote not running"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            return {"ok": False, "stderr": str(e)}
    WATCH_PID_FILE.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/mlflow_ui/status")
def api_mlflow_ui_status() -> dict:
    pid = _pid_alive(MLFLOW_UI_PID_FILE)
    return {"running": pid is not None, "pid": pid, "port": MLFLOW_UI_PORT, "url": f"http://127.0.0.1:{MLFLOW_UI_PORT}"}


@app.post("/api/mlflow_ui/start")
def api_mlflow_ui_start() -> dict:
    if _pid_alive(MLFLOW_UI_PID_FILE):
        return {"ok": True, "already_running": True, "url": f"http://127.0.0.1:{MLFLOW_UI_PORT}"}
    # Prefer the H100-synced DB so the UI shows the latest runs.
    db = _mlflow_db_path()
    if db is None:
        return {"ok": False, "error": "no mlflow.db found"}
    log = ROOT / "runs" / "mlflow_ui.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    venv_python = ROOT / ".venv" / "bin" / "python"
    cmd = [
        str(venv_python), "-m", "mlflow", "ui",
        "--backend-store-uri", f"sqlite:///{db.name}",
        "--host", "127.0.0.1", "--port", str(MLFLOW_UI_PORT),
    ]
    proc = subprocess.Popen(
        cmd, cwd=ROOT,
        stdout=log.open("ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    MLFLOW_UI_PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid, "url": f"http://127.0.0.1:{MLFLOW_UI_PORT}", "db": db.name}


@app.post("/api/mlflow_ui/stop")
def api_mlflow_ui_stop() -> dict:
    pid = _pid_alive(MLFLOW_UI_PID_FILE)
    if not pid:
        return {"ok": False, "error": "mlflow_ui not running"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try: os.kill(pid, signal.SIGTERM)
        except OSError as e: return {"ok": False, "error": str(e)}
    MLFLOW_UI_PID_FILE.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/cloud/push")
def api_cloud_push() -> dict:
    # Synchronous — small enough that it should complete in <60s for typical runs.
    return _bash([str(ROOT / "scripts/cloud_push.sh")], timeout=300)


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>gibbonBot dashboard</title>
<style>
:root { color-scheme: dark; }
body { font: 13px/1.4 ui-monospace, Menlo, Consolas, monospace; margin: 0; padding: 16px; background: #0e1116; color: #d0d6dd; }
h1 { margin: 0 0 12px; font-size: 16px; color: #fff; }
.row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.card { background: #161b22; border: 1px solid #2a313a; border-radius: 6px; padding: 10px 12px; min-width: 220px; flex: 1; }
.card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #8b949e; margin: 0 0 6px; font-weight: 600; }
.kv { display: flex; justify-content: space-between; gap: 8px; padding: 2px 0; }
.kv .v { color: #c9d1d9; font-variant-numeric: tabular-nums; }
.ok { color: #3fb950; }
.bad { color: #f85149; }
.warn { color: #d29922; }
button { background: #21262d; color: #c9d1d9; border: 1px solid #2a313a; padding: 6px 10px; border-radius: 4px; cursor: pointer; font: inherit; margin-right: 4px; }
button:hover { background: #30363d; }
button.danger { color: #f85149; }
button.go { color: #3fb950; }
pre { background: #0b0e13; border: 1px solid #2a313a; border-radius: 4px; padding: 8px; overflow: auto; max-height: 380px; margin: 0; white-space: pre-wrap; word-break: break-all; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 2px 6px; }
th { color: #8b949e; font-weight: 600; }
tbody tr:hover { background: #1a1f27; }
.muted { color: #6e7681; }
canvas { width: 100%; height: 140px; display: block; background: #0b0e13; border: 1px solid #2a313a; border-radius: 4px; }
</style></head><body>
<h1>gibbonBot — training dashboard</h1>

<div class="row">
  <div class="card">
    <h2>local trainer</h2>
    <div class="kv"><span>pids</span><span class="v" id="trainer-pids">—</span></div>
    <div style="margin-top: 8px;">
      <button class="go" onclick="post('/api/train/start')">start walk</button>
      <button class="danger" onclick="post('/api/train/stop')">stop</button>
    </div>
  </div>

  <div class="card">
    <h2>remote H100 <span class="muted" id="remote-host"></span></h2>
    <div class="kv"><span>watch_remote</span><span class="v" id="watch-pid">—</span></div>
    <div class="kv"><span>gpu util</span><span class="v" id="rh-util">—</span></div>
    <div class="kv"><span>vram</span><span class="v" id="rh-vram">—</span></div>
    <div class="kv"><span>temp</span><span class="v" id="rh-temp">—</span></div>
    <div class="kv"><span>power</span><span class="v" id="rh-power">—</span></div>
    <div class="kv"><span>sps</span><span class="v" id="rh-sps">—</span></div>
    <div class="kv"><span>last update</span><span class="v" id="rh-update">—</span></div>
    <div style="margin-top: 8px;">
      <button class="go" onclick="post('/api/remote/start')">start</button>
      <button class="danger" onclick="post('/api/remote/stop')">stop</button>
      <button class="go" onclick="post('/api/sync/start')">sync on</button>
      <button class="danger" onclick="post('/api/sync/stop')">sync off</button>
    </div>
  </div>

  <div class="card">
    <h2>checkpoint</h2>
    <div class="kv"><span>file</span><span class="v" id="ckpt-path">—</span></div>
    <div class="kv"><span>size</span><span class="v" id="ckpt-size">—</span></div>
    <div class="kv"><span>age</span><span class="v" id="ckpt-age">—</span></div>
    <div style="margin-top: 8px;">
      <button onclick="post('/api/cloud/push')">cloud push</button>
    </div>
  </div>

  <div class="card">
    <h2>latest metrics</h2>
    <div class="kv"><span>update</span><span class="v" id="m-update">—</span></div>
    <div class="kv"><span>sps</span><span class="v" id="m-sps">—</span></div>
    <div class="kv"><span>reward</span><span class="v" id="m-reward">—</span></div>
    <div class="kv"><span>loss</span><span class="v" id="m-loss">—</span></div>
  </div>
</div>

<div class="card">
  <h2>skills · what gibbon can do <span class="muted" id="next-hint"></span></h2>
  <table id="skills-table">
    <thead><tr><th style="text-align:left">task</th><th>state</th><th>steps</th><th>metric</th><th>value</th><th>target</th><th>confidence</th><th>age</th><th></th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="row">
  <div class="card" style="flex: 1; min-width: 360px;">
    <h2>reward (last 60 updates)</h2>
    <canvas id="reward-chart"></canvas>
  </div>
  <div class="card" style="flex: 1; min-width: 360px;">
    <h2>sps</h2>
    <canvas id="sps-chart"></canvas>
  </div>
</div>

<div class="card">
  <h2>runs explorer · pick runs to overlay
    <span class="muted" style="margin-left:8px">task:</span>
    <select id="runs-task" onchange="refreshRuns()">
      <option value="">all</option>
      <option value="walk">walk</option>
      <option value="stand">stand</option>
      <option value="walk_forward">walk_forward</option>
      <option value="turn_left">turn_left</option>
      <option value="turn_right">turn_right</option>
    </select>
    <span class="muted" style="margin-left:8px">sort:</span>
    <select id="runs-sort" onchange="renderRuns()">
      <option value="start_time">newest</option>
      <option value="mean_rew">mean_rew</option>
      <option value="min_dist">min_dist (asc)</option>
      <option value="mean_upright">mean_upright</option>
      <option value="mean_speed_x">mean_speed_x</option>
      <option value="checkpoint_steps">steps</option>
    </select>
    <button onclick="refreshRuns()" style="margin-left:8px">refresh</button>
  </h2>
  <table id="runs-table" style="font-size:12px">
    <thead><tr>
      <th></th>
      <th style="text-align:left">run</th>
      <th style="text-align:left">exp</th>
      <th>status</th>
      <th>mean_rew</th>
      <th>min_dist</th>
      <th>upright</th>
      <th>speed_x</th>
      <th>ωz</th>
      <th>steps</th>
      <th>age</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <div style="margin-top:10px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
    <span class="muted">overlay metric:</span>
    <select id="runs-metric" onchange="renderRunCharts()">
      <option value="mean_rew">mean_rew</option>
      <option value="min_dist">min_dist</option>
      <option value="mean_upright">mean_upright</option>
      <option value="mean_speed_x">mean_speed_x</option>
      <option value="mean_angvel_z">mean_angvel_z</option>
      <option value="success_frac">success_frac</option>
      <option value="gait_coherence">gait_coherence</option>
    </select>
    <span class="muted" id="runs-selected-count">0 selected</span>
  </div>
  <svg id="runs-chart" width="100%" height="260"
       style="margin-top:8px; background:#0b0e13; border:1px solid #2a313a; border-radius:4px; display:block;"></svg>
  <div id="runs-legend" style="margin-top:6px; font-size:11px;"></div>
</div>

<div class="card">
  <h2>mlflow ui · embedded deep-dive
    <span class="muted" id="mlflow-status" style="margin-left:8px">checking…</span>
    <button class="go" id="mlflow-start" onclick="mlflowStart()" style="margin-left:8px">launch</button>
    <button class="danger" id="mlflow-stop" onclick="mlflowStop()" style="display:none">stop</button>
    <button id="mlflow-toggle" onclick="mlflowToggle()" style="margin-left:4px">hide</button>
  </h2>
  <div id="mlflow-frame-wrap" style="display:none;">
    <iframe id="mlflow-frame" src="" style="width:100%; height:600px; border:1px solid #2a313a; border-radius:4px; background:#fff;"></iframe>
  </div>
</div>

<div class="card">
  <h2>system processes · top 25 by RAM</h2>
  <table id="proc-table">
    <thead><tr><th style="text-align:left">pid</th><th>tag</th><th>cpu%</th><th>mem%</th><th>rss</th><th style="text-align:left">cmd</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="card">
  <h2>train log tail · <span class="muted" id="log-path">—</span></h2>
  <pre id="log-tail">loading…</pre>
</div>

<div class="card" style="margin-top:12px;">
  <h2>last action</h2>
  <pre id="action-out" class="muted">(nothing yet)</pre>
</div>

<script>
async function post(url) {
  const out = document.getElementById('action-out');
  out.textContent = url + ' …';
  try {
    const r = await fetch(url, { method: 'POST' });
    const j = await r.json();
    out.textContent = url + '\\n' + JSON.stringify(j, null, 2);
  } catch (e) {
    out.textContent = url + '  FAILED: ' + e;
  }
  refresh();
}

function fmtAge(s) {
  if (s == null) return '—';
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(1) + 'm';
  if (s < 86400) return (s/3600).toFixed(1) + 'h';
  return (s/86400).toFixed(1) + 'd';
}

async function refreshSkills() {
  let s;
  try { s = await (await fetch('/api/skills')).json(); } catch (e) { return; }
  const tbody = document.querySelector('#skills-table tbody');
  tbody.innerHTML = '';
  for (const sk of s.skills) {
    const tr = document.createElement('tr');
    const stateClass = sk.state === 'trained' ? 'ok' :
                       sk.state === 'training' ? 'warn' :
                       sk.state === 'stalled' ? 'bad' : 'muted';
    const conf = sk.confidence == null ? '—' : (sk.confidence * 100).toFixed(0) + '%';
    const val = sk.value == null ? '—' : sk.value.toFixed(3);
    const thr = (sk.op === 'lt' ? '< ' : '> ') + sk.threshold;
    tr.innerHTML = `
      <td style="text-align:left"><strong>${sk.task}</strong>${sk.prereq ? ' <span class="muted">← '+sk.prereq+'</span>' : ''}</td>
      <td class="${stateClass}">${sk.state}</td>
      <td>${sk.steps ? sk.steps.toLocaleString() : '—'}</td>
      <td class="muted">${sk.metric}</td>
      <td>${val}</td>
      <td class="muted">${thr}</td>
      <td class="${sk.passes ? 'ok' : 'muted'}">${conf}</td>
      <td>${fmtAge(sk.age_s)}</td>
      <td><button class="go" onclick="trainTask('${sk.task}')">train</button></td>
    `;
    tbody.appendChild(tr);
  }
  document.getElementById('next-hint').textContent =
    s.next_to_train ? '· next to train: ' + s.next_to_train : '· all skills passing';
}

function trainTask(task) {
  post('/api/train/start?task=' + encodeURIComponent(task));
}

// Tracks last sps so the UI can render a delta arrow.
let lastRemoteSps = null;

async function refreshRemote() {
  let h;
  try { h = await (await fetch('/api/remote/health')).json(); } catch (e) { return; }
  document.getElementById('remote-host').textContent = h.host ? '· ' + h.host : '';
  if (!h.ok) {
    ['rh-util','rh-vram','rh-temp','rh-power','rh-sps','rh-update'].forEach(id => {
      const el = document.getElementById(id);
      el.textContent = 'unreachable';
      el.className = 'v bad';
    });
    return;
  }
  const g = h.gpu || {};
  const hh = h.health || {};

  function setKv(id, text, cls) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = 'v ' + (cls || '');
  }

  if (g.util_pct != null) {
    setKv('rh-util', g.util_pct.toFixed(0) + '%', g.util_pct > 80 ? 'ok' : g.util_pct > 30 ? 'warn' : 'bad');
  } else { setKv('rh-util', '—', 'muted'); }

  if (g.mem_used_mb != null && g.mem_total_mb) {
    const pct = (g.mem_used_mb / g.mem_total_mb) * 100;
    setKv('rh-vram',
      (g.mem_used_mb / 1024).toFixed(1) + ' / ' + (g.mem_total_mb / 1024).toFixed(0) + ' GB (' + pct.toFixed(0) + '%)',
      pct > 95 ? 'bad' : pct > 60 ? 'ok' : 'muted');
  } else { setKv('rh-vram', '—', 'muted'); }

  if (g.temp_c != null) {
    setKv('rh-temp', g.temp_c.toFixed(0) + '°C', g.temp_c > 85 ? 'bad' : g.temp_c > 70 ? 'warn' : 'ok');
  } else { setKv('rh-temp', '—', 'muted'); }

  if (g.power_w != null) setKv('rh-power', g.power_w.toFixed(0) + ' W', 'muted');

  if (hh.last_sps != null) {
    let arrow = '';
    if (lastRemoteSps != null) {
      const d = hh.last_sps - lastRemoteSps;
      const pct = lastRemoteSps ? (d / lastRemoteSps * 100) : 0;
      arrow = ' ' + (d > 0 ? '▲' : d < 0 ? '▼' : '·') + ' ' + (Math.abs(pct) < 10 ? '' : (pct > 0 ? '+' : '') + pct.toFixed(0) + '%');
    }
    setKv('rh-sps', hh.last_sps.toLocaleString() + arrow, 'ok');
    lastRemoteSps = hh.last_sps;
  }

  if (hh.log_idle_seconds != null) {
    setKv('rh-update', hh.log_idle_seconds.toFixed(0) + 's ago',
      hh.log_idle_seconds > 300 ? 'bad' : hh.log_idle_seconds > 60 ? 'warn' : 'ok');
  }
}

// --- Embedded mlflow ui controls ---
async function mlflowStatus() {
  try {
    const r = await (await fetch('/api/mlflow_ui/status')).json();
    const statusEl = document.getElementById('mlflow-status');
    const startBtn = document.getElementById('mlflow-start');
    const stopBtn = document.getElementById('mlflow-stop');
    const wrap = document.getElementById('mlflow-frame-wrap');
    const frame = document.getElementById('mlflow-frame');
    if (r.running) {
      statusEl.textContent = '· running pid=' + r.pid + ' on :' + r.port;
      statusEl.className = 'muted ok';
      startBtn.style.display = 'none';
      stopBtn.style.display = '';
      if (!frame.src) frame.src = r.url;
      if (wrap.dataset.userHidden !== '1') wrap.style.display = '';
    } else {
      statusEl.textContent = '· not running';
      statusEl.className = 'muted';
      startBtn.style.display = '';
      stopBtn.style.display = 'none';
      wrap.style.display = 'none';
      frame.src = '';
    }
  } catch (e) {}
}
async function mlflowStart() {
  await fetch('/api/mlflow_ui/start', { method: 'POST' });
  // mlflow ui takes ~3 s to bind its port; poll until status flips, then load iframe.
  setTimeout(mlflowStatus, 3500);
}
async function mlflowStop() {
  await fetch('/api/mlflow_ui/stop', { method: 'POST' });
  mlflowStatus();
}
function mlflowToggle() {
  const wrap = document.getElementById('mlflow-frame-wrap');
  const btn = document.getElementById('mlflow-toggle');
  if (wrap.style.display === 'none') {
    wrap.style.display = '';
    wrap.dataset.userHidden = '0';
    btn.textContent = 'hide';
  } else {
    wrap.style.display = 'none';
    wrap.dataset.userHidden = '1';
    btn.textContent = 'show';
  }
}

// --- Runs explorer state ---
let runsCache = [];
let selectedRuns = new Set();
// 8 categorical colours that read well on dark background.
const PALETTE = ['#3fb950','#58a6ff','#d29922','#f85149','#bc8cff','#39c5cf','#ff7b72','#a371f7'];

async function refreshRuns() {
  const task = document.getElementById('runs-task').value;
  const url = '/api/runs?limit=40' + (task ? '&task=' + encodeURIComponent(task) : '');
  let j;
  try { j = await (await fetch(url)).json(); } catch (e) { return; }
  runsCache = j.runs || [];
  renderRuns();
}

function renderRuns() {
  const sortKey = document.getElementById('runs-sort').value;
  const sorted = [...runsCache].sort((a, b) => {
    if (sortKey === 'start_time') return b.start_time - a.start_time;
    const av = a.metrics[sortKey];
    const bv = b.metrics[sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return sortKey === 'min_dist' ? av - bv : bv - av;  // min_dist is "lower is better"
  });
  const tbody = document.querySelector('#runs-table tbody');
  tbody.innerHTML = '';
  sorted.forEach(r => {
    const tr = document.createElement('tr');
    const m = r.metrics || {};
    const ageS = (Date.now() / 1000) - r.start_time;
    const fmt = v => v == null ? '<span class="muted">—</span>' : v.toFixed(3);
    const fmtSteps = v => v == null ? '<span class="muted">—</span>' : Math.round(v).toLocaleString();
    const checked = selectedRuns.has(r.run_id) ? 'checked' : '';
    tr.innerHTML = `
      <td><input type="checkbox" data-run="${r.run_id}" ${checked}></td>
      <td style="text-align:left"><code>${r.short_id}</code> <span class="muted">${r.name}</span></td>
      <td style="text-align:left" class="muted">${r.experiment.replace('ape_','').replace('_mjx','')}</td>
      <td class="${r.status === 'RUNNING' ? 'warn' : r.status === 'FINISHED' ? 'ok' : 'muted'}">${r.status}</td>
      <td>${fmt(m.mean_rew)}</td>
      <td>${fmt(m.min_dist)}</td>
      <td>${fmt(m.mean_upright)}</td>
      <td>${fmt(m.mean_speed_x)}</td>
      <td>${fmt(m.mean_angvel_z)}</td>
      <td>${fmtSteps(m.checkpoint_steps)}</td>
      <td class="muted">${fmtAge(ageS)}</td>
    `;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll('input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', e => {
      const id = e.target.dataset.run;
      if (e.target.checked) selectedRuns.add(id); else selectedRuns.delete(id);
      document.getElementById('runs-selected-count').textContent = selectedRuns.size + ' selected';
      renderRunCharts();
    });
  });
  document.getElementById('runs-selected-count').textContent = selectedRuns.size + ' selected';
}

async function renderRunCharts() {
  const metric = document.getElementById('runs-metric').value;
  const svg = document.getElementById('runs-chart');
  const legend = document.getElementById('runs-legend');
  svg.innerHTML = '';
  legend.innerHTML = '';
  if (selectedRuns.size === 0) {
    svg.innerHTML = '<text x="12" y="20" fill="#6e7681" font-size="11">select runs above to overlay</text>';
    return;
  }
  const W = svg.clientWidth || 800;
  const H = 260;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  const ids = [...selectedRuns];
  const series = await Promise.all(ids.map(async (id, i) => {
    const r = await fetch('/api/run_history?run_id=' + id + '&metric=' + encodeURIComponent(metric));
    const j = await r.json();
    return { run_id: id, points: j.points || [], colour: PALETTE[i % PALETTE.length] };
  }));

  const allPts = series.flatMap(s => s.points);
  if (allPts.length === 0) {
    svg.innerHTML = '<text x="12" y="20" fill="#6e7681" font-size="11">no data for metric "' + metric + '" in selected runs</text>';
    return;
  }
  const minX = Math.min(...allPts.map(p => p[0]));
  const maxX = Math.max(...allPts.map(p => p[0]));
  const minY = Math.min(...allPts.map(p => p[1]));
  const maxY = Math.max(...allPts.map(p => p[1]));
  const padL = 50, padR = 12, padT = 14, padB = 24;
  const spanX = Math.max(1, maxX - minX);
  const spanY = (maxY - minY) || 1;
  const sx = x => padL + (x - minX) / spanX * (W - padL - padR);
  const sy = y => H - padB - (y - minY) / spanY * (H - padT - padB);

  // Axes / grid
  const axisColour = '#2a313a';
  const tickColour = '#6e7681';
  let grid = '';
  for (let k = 0; k <= 4; k++) {
    const yPx = padT + k / 4 * (H - padT - padB);
    const yVal = maxY - k / 4 * spanY;
    grid += `<line x1="${padL}" y1="${yPx}" x2="${W-padR}" y2="${yPx}" stroke="${axisColour}" stroke-dasharray="2,3"/>`;
    grid += `<text x="${padL-6}" y="${yPx+3}" fill="${tickColour}" font-size="10" text-anchor="end">${yVal.toFixed(2)}</text>`;
  }
  for (let k = 0; k <= 4; k++) {
    const xPx = padL + k / 4 * (W - padL - padR);
    const xVal = minX + k / 4 * spanX;
    grid += `<text x="${xPx}" y="${H-padB+14}" fill="${tickColour}" font-size="10" text-anchor="middle">${(xVal/1000).toFixed(0)}k</text>`;
  }

  // Series lines
  let lines = '';
  for (const s of series) {
    if (s.points.length < 2) continue;
    const d = s.points.map((p, i) => (i === 0 ? 'M' : 'L') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1)).join(' ');
    lines += `<path d="${d}" stroke="${s.colour}" stroke-width="1.5" fill="none"/>`;
  }

  // Best-of indicator: highest-final-value run (or lowest for min_dist).
  const lower_is_better = metric === 'min_dist';
  const ranked = series
    .filter(s => s.points.length)
    .map(s => ({ id: s.run_id, last: s.points[s.points.length - 1][1], colour: s.colour }))
    .sort((a, b) => lower_is_better ? a.last - b.last : b.last - a.last);

  svg.innerHTML = grid + lines;

  // Legend
  legend.innerHTML = ranked.map((r, i) => {
    const tag = i === 0 ? '<span class="ok"> ★ best</span>' : '';
    return `<span style="display:inline-block; margin-right:14px;">
      <span style="display:inline-block; width:10px; height:10px; background:${r.colour}; vertical-align:middle; margin-right:4px;"></span>
      <code>${r.id.slice(0,8)}</code>
      <span class="muted">${metric}=</span>${r.last.toFixed(3)}${tag}
    </span>`;
  }).join('');
}

async function refreshProcesses() {
  let p;
  try { p = await (await fetch('/api/processes')).json(); } catch (e) { return; }
  const tbody = document.querySelector('#proc-table tbody');
  tbody.innerHTML = '';
  for (const r of p.rows) {
    const tr = document.createElement('tr');
    const tagClass = r.tag === 'trainer' ? 'ok' :
                     r.tag === 'claude' ? 'warn' :
                     r.tag === 'ollama' ? 'bad' :
                     r.tag ? 'muted' : '';
    tr.innerHTML = `
      <td style="text-align:left">${r.pid}</td>
      <td class="${tagClass}">${r.tag || ''}</td>
      <td>${r.pcpu.toFixed(1)}</td>
      <td>${r.pmem.toFixed(1)}</td>
      <td>${r.rss_mb}M</td>
      <td style="text-align:left" class="muted">${r.args.length > 90 ? r.args.slice(0, 90)+'…' : r.args}</td>
    `;
    tbody.appendChild(tr);
  }
}

function drawChart(canvas, points) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth;
  const H = canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (!points.length) return;
  const mn = Math.min(...points), mx = Math.max(...points);
  const span = (mx - mn) || 1;
  ctx.strokeStyle = '#3fb950';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((y, i) => {
    const px = (i / Math.max(1, points.length - 1)) * (W - 8) + 4;
    const py = H - 8 - ((y - mn) / span) * (H - 16);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px monospace';
  ctx.fillText(mx.toFixed(2), 4, 12);
  ctx.fillText(mn.toFixed(2), 4, H - 2);
}

async function refresh() {
  let s;
  try {
    s = await (await fetch('/api/status')).json();
  } catch (e) { return; }

  document.getElementById('trainer-pids').textContent =
    s.trainer_pids.length ? s.trainer_pids.join(',') : 'idle';
  document.getElementById('trainer-pids').className =
    'v ' + (s.trainer_pids.length ? 'ok' : 'muted');

  document.getElementById('watch-pid').textContent = s.watch_remote_pid || 'off';
  document.getElementById('watch-pid').className =
    'v ' + (s.watch_remote_pid ? 'ok' : 'muted');

  if (s.ckpt) {
    document.getElementById('ckpt-path').textContent = s.ckpt.path.split('/').pop();
    document.getElementById('ckpt-size').textContent = s.ckpt.size_mb + ' MB';
    const age = s.ckpt.age_s;
    document.getElementById('ckpt-age').textContent =
      age < 60 ? age.toFixed(0)+'s' : (age/60).toFixed(1)+'m';
    document.getElementById('ckpt-age').className =
      'v ' + (age < 120 ? 'ok' : age < 600 ? 'warn' : 'bad');
  } else {
    document.getElementById('ckpt-path').textContent = '(none)';
  }

  if (s.metrics.length) {
    const last = s.metrics[s.metrics.length - 1];
    document.getElementById('m-update').textContent = last.update;
    document.getElementById('m-sps').textContent = last.sps.toLocaleString();
    document.getElementById('m-reward').textContent = last.reward.toFixed(3);
    document.getElementById('m-loss').textContent = last.loss.toFixed(3);
    drawChart(document.getElementById('reward-chart'), s.metrics.map(m => m.reward));
    drawChart(document.getElementById('sps-chart'), s.metrics.map(m => m.sps));
  }

  document.getElementById('log-path').textContent = s.log_path || '(no log)';
  document.getElementById('log-tail').textContent = s.log_tail.join('\\n') || '(empty)';
}

function refreshAll() { refresh(); refreshSkills(); refreshProcesses(); refreshRemote(); refreshRuns(); mlflowStatus(); }
refreshAll();
setInterval(refresh, 4000);
setInterval(refreshSkills, 8000);
setInterval(refreshProcesses, 6000);
// Remote ssh is the slowest poll — every 15s. Skipped silently if box is down.
setInterval(refreshRemote, 15000);
// Runs list refreshes infrequently — MLflow data only changes on sync ticks.
setInterval(refreshRuns, 30000);
setInterval(mlflowStatus, 10000);
</script>
</body></html>
"""


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    print(f"[dashboard] http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
