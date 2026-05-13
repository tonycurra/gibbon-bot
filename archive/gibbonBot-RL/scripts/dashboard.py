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

    # Prefer per-task sidecars in keepers/ (richest data).
    keepers = run_dir / "keepers"
    if keepers.exists():
        sidecars = sorted(keepers.glob("auto_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for sc in sidecars[:5]:
            try:
                metrics = json.loads(sc.read_text()).get("metrics", {})
                if metric in metrics:
                    last_value = float(metrics[metric])
                    break
            except (OSError, ValueError):
                continue

    # Fall back to live_stats.json (some tasks write this).
    if last_value is None:
        live = run_dir / "live_stats.json"
        if live.exists():
            try:
                data = json.loads(live.read_text())
                if metric in data:
                    last_value = float(data[metric])
            except (OSError, ValueError):
                pass

    # Last resort: parse last reward from train.log.
    if last_value is None and metric == "mean_rew":
        for line in reversed(_tail(run_dir / "train.log", n=400)):
            mm = METRIC_RE.match(line)
            if mm:
                last_value = float(mm.group(5))
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
    <h2>remote H100</h2>
    <div class="kv"><span>watch_remote</span><span class="v" id="watch-pid">—</span></div>
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

function refreshAll() { refresh(); refreshSkills(); refreshProcesses(); }
refreshAll();
setInterval(refresh, 4000);
setInterval(refreshSkills, 8000);
setInterval(refreshProcesses, 6000);
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
