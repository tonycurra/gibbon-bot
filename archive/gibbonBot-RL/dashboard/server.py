"""
GibbonBot Dashboard Server
FastAPI backend for training control, brain state retrieval, and MuJoCo viewer frames.

Run: uvicorn dashboard.server:app --reload --port 8000
Access: http://localhost:8000/static/index.html

Note: For MJX/Warp rendering, frames are generated on-demand.
"""

import os

# Force headless EGL for MuJoCo rendering. Under WSL2 / no-X environments,
# the default glfw backend hits an X11 assertion and the server crashes.
# Must be set before mujoco is imported (anywhere — including via brain/).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import logging
import re
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path


# Silence the firehose of GET /api/{frame,metrics,logs,status,system/stats}
# access logs. Errors and warnings still print, plus state-changing POSTs.
class _AccessFilter(logging.Filter):
    SPAMMY = (
        "/api/viewer/frame", "/api/viewer/stream", "/api/viewer/info",
        "/api/train/metrics", "/api/train/logs", "/api/train/status",
        "/api/system/stats", "/api/params/get", "/api/sim/joints",
        "/api/tasks/overview", "/api/whats-happening", "/api/sim/mode",
        "/api/runs", "/api/trial/info", "/api/trial/status",
        "/api/library/list", "/api/env/params",
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self.SPAMMY)


logging.getLogger("uvicorn.access").addFilter(_AccessFilter())

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# Serialize MuJoCo model/data/renderer access — FastAPI dispatches sync
# handlers on a threadpool, and the renderer + MjData are not thread-safe.
_SIM_LOCK = threading.Lock()

# Last good JPEG frame produced by the stream loop. /api/viewer/frame returns
# this snapshot instead of rendering, so EGL context contention can't happen
# between the stream thread and request handlers.
_LAST_FRAME_JPEG: bytes | None = None
_LAST_FRAME_TS: float = 0.0
_RENDER_OWNER_THREAD: int | None = None  # tid that created the EGL context

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_BG = REPO_ROOT / "scripts" / "train_bg.sh"
ACTIVE_TASK_FILE = REPO_ROOT / "runs" / ".active_task"
ACTIVE_ENVS_FILE = REPO_ROOT / "runs" / ".active_envs"

# Add parent directory to path for brain imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="GibbonBot Dashboard")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# Global training process
training_process = None
training_log_file = "train.log"

# MJX Model cache
_model_cache = None
_data_cache = None
_renderer_cache = None


def get_model():
    """Load or cache MuJoCo model + data only. Does NOT create the renderer
    — that must be created on the rendering thread (EGL contexts are
    thread-bound). Safe to call from any handler."""
    global _model_cache, _data_cache
    if _model_cache is None:
        try:
            import mujoco
            scene_path = "data/scene.xml"
            if not os.path.exists(scene_path):
                scene_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "scene.xml"
                )
            _model_cache = mujoco.MjModel.from_xml_path(scene_path)
            _data_cache = mujoco.MjData(_model_cache)
            for i in range(_model_cache.ngeom):
                if _model_cache.geom_type[i] == 5:
                    _model_cache.geom_contype[i] = 2
                    _model_cache.geom_conaffinity[i] = 1
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return _model_cache, _data_cache


def get_renderer():
    """Create or recreate the renderer ON THE CURRENT THREAD. Only the
    stream generator should call this — other endpoints don't render."""
    global _renderer_cache, _RENDER_OWNER_THREAD
    import mujoco
    tid = threading.get_ident()
    if _renderer_cache is None or _RENDER_OWNER_THREAD != tid:
        # Drop the old renderer explicitly so its EGL context releases
        # before we make a new one. Otherwise EGL_BAD_ACCESS piles up.
        if _renderer_cache is not None:
            try: _renderer_cache.close()
            except Exception: pass
        model, _ = get_model()
        _renderer_cache = mujoco.Renderer(model, height=960, width=1280)
        _RENDER_OWNER_THREAD = tid
    return _renderer_cache


# ============================================================================
# TRAINING API
# ============================================================================


def _trainer_pids():
    try:
        out = subprocess.check_output(["pgrep", "-f", "[t]rain_mjx.py"], text=True)
        return [int(p) for p in out.split() if p.strip()]
    except subprocess.CalledProcessError:
        return []


def _read_active(path: Path, default: str) -> str:
    try:
        return path.read_text().strip() or default
    except FileNotFoundError:
        return default


def _bg(*args: str) -> dict:
    """Invoke scripts/train_bg.sh — returns {ok, stdout, stderr}.

    - ALLOW_GPU_SHARE=1: dashboard holds a small chunk of GPU for its
      MuJoCo renderer; without this flag train_bg.sh refuses to launch.
    - XLA_PYTHON_CLIENT_PREALLOCATE=false: JAX grabs only what it needs
      instead of 75% of VRAM. Otherwise the dashboard's EGL renderer
      stops producing frames once training starts.
    - XLA_PYTHON_CLIENT_MEM_FRACTION=0.6: cap JAX growth to leave headroom.
    """
    cache_dir = REPO_ROOT / "runs" / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "ALLOW_GPU_SHARE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
        # JAX/XLA persistent compilation cache — second start with same
        # scene+code is ~5s instead of 60s. Cache survives reboots.
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
    }
    res = subprocess.run(
        ["bash", str(TRAIN_BG), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60, env=env,
    )
    return {"ok": res.returncode == 0, "stdout": res.stdout, "stderr": res.stderr}


INFINITE_STEPS = 10_000_000_000  # 10B — effectively never reached; safety cap only.
GOAL_SUCCESS_FRAC = 0.90          # auto-stop when policy succeeds in 90% of envs
_DEFAULT_TRIAL_STEPS = 600        # ~20s per trial episode before forced reset


@app.post("/api/train/start")
def start_training(task: str = "walk", envs: int | None = None,
                   total_steps: int | None = None, resume: bool = True):
    if _trainer_pids():
        raise HTTPException(status_code=400, detail="Trainer already running — use /api/train/restart")
    # Task is now meaningful: env.read_params(task) zeros out tree-progress
    # rewards when task='stand'. Folder runs/ape_<task>_mjx/ still segregates
    # checkpoints per task.
    if task not in ("walk", "stand"):
        task = "walk"
    args = ["start", task]
    if envs:
        args += ["--envs", str(envs)]
    # No step cap unless user gave one — the trainer auto-stops on goal
    # (success_frac > 0.9 sustained for 20 updates) and on Ctrl-C/Stop.
    args += ["--total-steps", str(total_steps or INFINITE_STEPS)]
    args += ["--early-stop-metric", "success_frac",
             "--early-stop-op", "gt",
             "--early-stop-value", str(GOAL_SUCCESS_FRAC)]
    if resume:
        args += ["--resume"]
    res = _bg(*args)
    if not res["ok"]:
        raise HTTPException(status_code=500, detail=res["stderr"] or res["stdout"])
    ACTIVE_TASK_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_TASK_FILE.write_text(task)
    if envs:
        ACTIVE_ENVS_FILE.write_text(str(envs))
    # Auto-start looping trial so the viewer shows real-time playback
    # of the latest checkpoint while the trainer churns. Default to
    # 0.5x speed — more realistic / less jerky to watch.
    global _trial_proc
    if _trial_proc is None or _trial_proc.poll() is not None:
        try: _spawn_trial(task, _DEFAULT_TRIAL_STEPS, 0.5, loop=True)
        except Exception as e: sys.stderr.write(f"[liana] auto-trial failed to start: {e}\n")
    return {"status": "started", "task": task, "envs": envs, "detail": res["stdout"].strip()}


@app.post("/api/train/stop")
def stop_training():
    """Stops only the trainer. The trial subprocess keeps running so the
    viewer continues to show the last-known-best policy. Use /api/trial/stop
    explicitly to kill the trial."""
    res = _bg("stop")
    return {"status": "stopped" if res["ok"] else "error", "detail": (res["stdout"] or res["stderr"]).strip()}


@app.post("/api/train/restart")
def restart_training(task: str | None = None, envs: int | None = None, total_steps: int | None = None, resume: bool = True):
    """Stop + start with new task / num_envs. Required for changing batch size mid-run
    (MJX recompiles on batch-shape change)."""
    cur_task = _read_active(ACTIVE_TASK_FILE, "walk")
    cur_envs = _read_active(ACTIVE_ENVS_FILE, "")
    new_task = task or cur_task
    new_envs = envs if envs is not None else (int(cur_envs) if cur_envs else None)
    _bg("stop")
    time.sleep(2)
    args = ["start", new_task]
    if new_envs:
        args += ["--envs", str(new_envs)]
    args += ["--total-steps", str(total_steps or INFINITE_STEPS)]
    args += ["--early-stop-metric", "success_frac",
             "--early-stop-op", "gt",
             "--early-stop-value", str(GOAL_SUCCESS_FRAC)]
    if resume:
        args += ["--resume"]
    res = _bg(*args)
    if not res["ok"]:
        raise HTTPException(status_code=500, detail=res["stderr"] or res["stdout"])
    ACTIVE_TASK_FILE.write_text(new_task)
    if new_envs:
        ACTIVE_ENVS_FILE.write_text(str(new_envs))
    return {"status": "restarted", "task": new_task, "envs": new_envs}


@app.get("/api/train/status")
def get_status():
    pids = _trainer_pids()
    return {
        "running": bool(pids),
        "pid": pids[0] if pids else None,
        "task": _read_active(ACTIVE_TASK_FILE, "walk"),
        "envs": _read_active(ACTIVE_ENVS_FILE, "(auto)"),
    }


# ───── Params editor (hot-reloaded per-episode by trainer) ─────
# (lo, hi, sane_default) — tight ranges that produce sensible motion.
# Maxing any of these gives garbage (jerky motors, exploding rewards).
_PARAM_KEYS = {
    "MOTOR_SPEED_FRAC":    (0.15, 0.80, 0.40),
    "MOTOR_ACCEL_RATIO":   (0.8,  3.0,  1.5),
    "ACTION_SMOOTH_ALPHA": (0.05, 0.50, 0.15),  # lower = smoother (EMA pole)
    "W_UPRIGHT":           (0.0,  10.0, 3.0),
    "W_HEIGHT":            (0.0,  5.0,  0.5),
    "W_PROGRESS":          (10.0, 200.0, 80.0),
    "W_SMOOTH":            (0.0,  2.0,  0.30),
    "FLIP_PENALTY":        (0.0,  150.0, 60.0),
}


@app.post("/api/params/reset")
def params_reset(task: str = "walk"):
    """Reset every tuning slider to its known-good default."""
    out = {}
    for key, (lo, hi, default) in _PARAM_KEYS.items():
        try:
            params_set(task=task, key=key, value=default)
            out[key] = default
        except Exception as e:
            out[key] = f"error: {e}"
    return {"ok": True, "applied": out}


def _params_path(task: str = "walk") -> Path:
    """All tasks share envs/env.py since the refactor; the file holds
    every reward/motor knob as a module-level constant."""
    return REPO_ROOT / "envs" / "env.py"


@app.get("/api/env/params")
def env_params():
    """Read-only dump of every active constant in envs/env.py, grouped
    by purpose. Lets the user see exactly what the trainer is using
    without opening the file."""
    import importlib
    import envs.env as e
    importlib.reload(e)  # pick up edits without restarting the dashboard
    groups = {
        "Motor & action": ["MOTOR_SPEED_FRAC", "MOTOR_ACCEL_RATIO",
                            "ACTION_SMOOTH_ALPHA", "MOTOR_KP"],
        "Reward · progress to tree": ["W_PROGRESS", "W_FWD_VEL",
                            "W_PROXIMITY_TREE", "W_HEADING",
                            "SUCCESS_DIST", "W_SUCCESS"],
        "Reward · posture": ["TARGET_Z", "TARGET_TOL", "W_HEIGHT", "W_UPRIGHT"],
        "Reward · penalties": ["GROUND_CLEARANCE", "W_GROUND",
                            "FLIP_PENALTY", "W_ALIVE", "W_SMOOTH",
                            "W_JVEL", "W_SYMMETRY", "W_ENERGY"],
        "PPO": ["LEARNING_RATE", "ENT_COEF", "CLIP_EPS"],
        "CPG (disabled)": ["W_CPG_PITCH", "W_CPG_ELBOW"],
    }
    out = []
    for gname, keys in groups.items():
        rows = []
        for k in keys:
            if hasattr(e, k):
                v = getattr(e, k)
                if isinstance(v, (int, float, str, bool)):
                    rows.append({"key": k, "value": v})
        out.append({"group": gname, "rows": rows})
    return {"groups": out, "source_file": "envs/env.py"}


LIBRARY_DIR = REPO_ROOT / "runs" / "library"


@app.get("/api/library/list")
def library_list():
    """List all named policy snapshots.

    Each entry has the form runs/library/<name>.npz + <name>.json
    (sidecar with metadata copied from the source keeper).
    """
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for npz in sorted(LIBRARY_DIR.glob("*.npz")):
        meta = {}
        side = npz.with_suffix(".json")
        if side.exists():
            try: meta = __import__("json").loads(side.read_text())
            except Exception: pass
        out.append({
            "name": npz.stem,
            "saved_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(npz.stat().st_mtime)),
            "size_kb": int(npz.stat().st_size / 1024),
            "task": meta.get("task"),
            "succ": meta.get("best_succ"),
            "min_dist": meta.get("min_dist"),
            "source": meta.get("source"),
        })
    return {"entries": out}


def _slug(name: str) -> str:
    import re as _re
    name = (name or "").strip().lower()
    name = _re.sub(r"[^a-z0-9_-]+", "_", name)[:60]
    return name or "untitled"


@app.post("/api/library/save")
def library_save(name: str, task: str = "walk"):
    """Snapshot the current BEST keeper into the library under `name`.

    Copies the .npz (and JSON sidecar if present) so the library entry
    is decoupled from the live runs/ tree being overwritten later.
    """
    import json, shutil
    runs = REPO_ROOT / "runs" / f"ape_{task}_mjx"
    keepers = sorted((runs / "keepers").glob("*.npz")) if (runs / "keepers").exists() else []
    if not keepers:
        raise HTTPException(status_code=404, detail=f"no keepers in {runs}/keepers — train more first")
    # Best by filename: highest succ then lowest min_d.
    import re as _re
    def score(p):
        md = _re.search(r"min_d([\d.]+)", p.name)
        sc = _re.search(r"succ([\d.]+)", p.name)
        return (float(sc.group(1)) if sc else 0.0, -(float(md.group(1)) if md else 99.0))
    best = max(keepers, key=score)
    slug = _slug(name)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    target = LIBRARY_DIR / f"{slug}.npz"
    shutil.copy2(best, target)
    md = _re.search(r"min_d([\d.]+)", best.name)
    sc = _re.search(r"succ([\d.]+)", best.name)
    meta = {
        "name": slug,
        "task": task,
        "source": str(best.relative_to(REPO_ROOT)),
        "best_succ": float(sc.group(1)) if sc else None,
        "min_dist": float(md.group(1)) if md else None,
        "saved_at": time.time(),
    }
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return {"ok": True, "name": slug, "saved": str(target.relative_to(REPO_ROOT))}


@app.post("/api/library/delete")
def library_delete(name: str):
    slug = _slug(name)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    p = LIBRARY_DIR / f"{slug}.npz"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no library entry '{slug}'")
    p.unlink()
    p.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/library/use")
def library_use(name: str, task: str = "walk"):
    """Seed runs/ape_<task>_mjx/ with a library entry as the starting
    checkpoint for the next training run. Refuses while training is
    running so you don't clobber active state.
    """
    import json, shutil
    if _trainer_pids():
        raise HTTPException(status_code=400, detail="stop training first")
    slug = _slug(name)
    src = LIBRARY_DIR / f"{slug}.npz"
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"no library entry '{slug}'")
    runs = REPO_ROOT / "runs" / f"ape_{task}_mjx"
    runs.mkdir(parents=True, exist_ok=True)
    # Highest step we've seen so the trainer picks this as "latest" on next start.
    next_step = 1_000_000_000
    target = runs / f"mjx_{next_step:012d}.npz"
    shutil.copy2(src, target)
    # Drop a marker JSON so we can trace where this came from.
    (runs / "library_origin.json").write_text(json.dumps({
        "library_name": slug, "loaded_at": time.time(),
    }, indent=2))
    return {"ok": True, "seeded": str(target.relative_to(REPO_ROOT)), "from": slug}


@app.get("/api/runs")
def runs_list(limit: int = 20):
    """Compact, in-dashboard run history. Reads MLflow's SQLite store
    directly so we don't need to launch the mlflow UI as a separate
    process.

    Returns the last `limit` runs across all experiments, sorted newest
    first. Each entry shows the few columns that matter for comparison:
    when, status, best success_frac / mean_rew, scene fingerprint,
    and the headline params.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT/'mlflow.db'}")
        client = mlflow.MlflowClient()
        exps = client.search_experiments()
        if not exps:
            return {"runs": [], "note": "no MLflow experiments yet"}
        exp_ids = [e.experiment_id for e in exps]
        mlruns = client.search_runs(
            experiment_ids=exp_ids,
            order_by=["attributes.start_time DESC"],
            max_results=int(limit),
        )
    except Exception as e:
        return {"runs": [], "error": str(e)}

    HEADLINE_PARAMS = ("W_PROGRESS", "W_UPRIGHT", "MOTOR_SPEED_FRAC",
                       "ACTION_SMOOTH_ALPHA", "FLIP_PENALTY", "N_ENVS")
    out = []
    for r in mlruns:
        params = r.data.params or {}
        metrics = r.data.metrics or {}
        tags = r.data.tags or {}
        out.append({
            "id": r.info.run_id[:8],
            "name": r.info.run_name,
            "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(r.info.start_time / 1000)),
            "status": r.info.status,
            "task": tags.get("task", "?"),
            "scene_fp": (tags.get("scene_fingerprint", "") or tags.get("fingerprint", ""))[:8],
            "best_succ": metrics.get("success_frac"),
            "best_rew": metrics.get("mean_rew"),
            "min_dist": metrics.get("mean_dist"),
            "steps": metrics.get("update", 0) * int(params.get("N_STEPS", 0) or 0) * int(params.get("N_ENVS", 0) or 0),
            "params": {k: params.get(k) for k in HEADLINE_PARAMS if k in params},
        })
    return {"runs": out}


@app.get("/api/params/get")
def params_get(task: str = "walk"):
    src = _params_path(task).read_text()
    out = {}
    for key in _PARAM_KEYS:
        m = re.search(rf"^{key}\s*=\s*([0-9.+\-eE]+)", src, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    # Pass (lo, hi) pairs to the UI; default lives server-side via /api/params/reset.
    bounds = {k: (v[0], v[1]) for k, v in _PARAM_KEYS.items()}
    return {"task": task, "params": out, "bounds": bounds}


@app.post("/api/params/set")
def params_set(task: str = "walk", key: str = "", value: float = 0.0):
    if key not in _PARAM_KEYS:
        raise HTTPException(status_code=400, detail=f"Key '{key}' not editable. Allowed: {list(_PARAM_KEYS)}")
    lo, hi, _default = _PARAM_KEYS[key]
    if not (lo <= value <= hi):
        raise HTTPException(status_code=400, detail=f"{key}={value} out of bounds [{lo}, {hi}]")
    path = _params_path(task)
    src = path.read_text()
    new_line = f"{key} = {value}"
    new_src, n = re.subn(
        rf"^{key}\s*=\s*[0-9.+\-eE]+.*$", new_line, src, count=1, flags=re.MULTILINE,
    )
    if n == 0:
        raise HTTPException(status_code=404, detail=f"{key} not found in {path.name}")
    path.write_text(new_src)
    return {"ok": True, "task": task, "key": key, "value": value, "note": "picked up next episode (~few seconds)"}


@app.get("/api/train/metrics")
def get_metrics():
    if os.path.exists(training_log_file):
        with open(training_log_file, "r") as f:
            lines = f.readlines()[-100:]
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].isdigit():
                        try:
                            return {
                                "step": int(parts[0]),
                                "reward": float(parts[1].replace(",", "")),
                                "mean_rew": float(parts[2].replace(",", "")),
                                "mean_dist": float(parts[3].replace(",", "")),
                            }
                        except (ValueError, IndexError):
                            pass
    return {"error": "No metrics found"}


@app.get("/api/train/logs")
def get_logs(tail: int = 50):
    if not os.path.exists(training_log_file):
        return {"logs": []}
    with open(training_log_file, "r") as f:
        lines = f.readlines()[-tail:]
    return {"logs": lines}


# ============================================================================
# BRAIN API
# ============================================================================


@app.get("/api/brain/modules")
def list_modules():
    try:
        from brain.ape_brain import ApeBrain

        brain = ApeBrain()
        return {"modules": list(brain.modules.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brain/state")
def get_brain_state():
    try:
        from brain.ape_brain import ApeBrain

        brain = ApeBrain()
        state = {}
        for name, module in brain.modules.items():
            if hasattr(module, "state"):
                state[name] = {
                    "type": type(module.state).__name__,
                }
            else:
                state[name] = "No state attribute"
        return {"state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brain/memory")
def get_memory(limit: int = 10):
    try:
        from brain.ape_brain import ApeBrain

        brain = ApeBrain()
        hippo = brain.modules.get("hippocampus")
        if hippo and hasattr(hippo, "memories"):
            memories = []
            for i, mem in enumerate(hippo.memories[-limit:]):
                memories.append(
                    {
                        "index": i,
                        "reward": float(mem.reward) if hasattr(mem, "reward") else None,
                        "punishment": float(mem.punishment)
                        if hasattr(mem, "punishment")
                        else None,
                        "timestamp": int(mem.timestamp)
                        if hasattr(mem, "timestamp")
                        else None,
                    }
                )
            return {"memories": memories}
        return {"memories": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# VIEWER API
# ============================================================================


def _create_transparent_png(width: int = 1, height: int = 1) -> bytes:
    """Create a transparent PNG placeholder."""
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Pre-generate transparent PNG
_TRANSPARENT_PNG = _create_transparent_png()


_LIVE_QPOS_CACHE: dict = {"qpos": None, "ts": 0.0, "src_mtime": 0.0}
LIVE_QPOS_MIN_INTERVAL = 0.2  # 5fps — slow enough that resets don't blink past


TRIAL_QPOS = REPO_ROOT / "runs" / ".trial_qpos.npy"
TRIAL_INFO = REPO_ROOT / "runs" / ".trial_info.json"


def _load_live_qpos(task: str):
    """Read qpos for the viewer. Trial ONLY — we don't show the trainer's
    env-0 (jerky, resets visible). If the trial isn't running, the viewer
    returns the idle placeholder (handled by _render_png_bytes)."""
    if _trial_proc is not None and _trial_proc.poll() is None and TRIAL_QPOS.exists():
        try:
            return np.load(str(TRIAL_QPOS))
        except Exception:
            pass
    return None


@app.get("/api/trial/info")
def trial_info():
    """What the viewer's playing right now: episode #, checkpoint filename,
    policy kind (trained / random), last outcome."""
    if not TRIAL_INFO.exists():
        return {"running": False}
    try:
        import json
        d = json.loads(TRIAL_INFO.read_text())
        d["running"] = _trial_proc is not None and _trial_proc.poll() is None
        return d
    except Exception as e:
        return {"running": False, "error": str(e)}


# ───── Single-trial playback (real-time CPU subprocess) ─────
_trial_proc: subprocess.Popen | None = None


def _kill_stray_trials():
    """Reap any play_trial.py / demo_walk.py processes from previous
    sessions BEFORE spawning a new one. If two of these run at once they
    both write to runs/.trial_qpos.npy and the viewer flips between
    robot states (the 'many robots, fast' bug)."""
    try:
        subprocess.run(["pkill", "-f", "scripts/play_trial.py"],
                       check=False, timeout=3)
        subprocess.run(["pkill", "-f", "scripts/demo_walk.py"],
                       check=False, timeout=3)
        time.sleep(0.3)
    except Exception:
        pass


@app.post("/api/demo/walk")
def demo_walk(freq: float = 1.2, amp: float = 0.40, seconds: float = 20.0, loop: bool = True):
    """Run the scripted sinusoidal walk demo — NO policy, just a hand-coded
    gait. Visual reference for 'what walking SHOULD look like'."""
    global _trial_proc
    _kill_stray_trials()
    max_steps = max(30, int(seconds / 0.032))
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "scripts" / "demo_walk.py"),
        "--max-steps", str(max_steps),
        "--freq", str(freq),
        "--amp", str(amp),
    ]
    if loop:
        cmd.append("--loop")
    log = open(REPO_ROOT / "runs" / "demo.log", "w")
    _trial_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    return {"ok": True, "pid": _trial_proc.pid, "freq": freq, "amp": amp, "seconds": seconds}


def _spawn_trial(task: str, max_steps: int, realtime: float, loop: bool,
                 stuck_window: float = 3.0, stuck_dist: float = 0.05) -> int:
    """Spawn the play_trial subprocess. Caller owns the global _trial_proc."""
    global _trial_proc
    _kill_stray_trials()
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "scripts" / "play_trial.py"),
        "--task", task,
        "--max-steps", str(max_steps),
        "--realtime", str(realtime),
        "--stuck-window", str(stuck_window),
        "--stuck-dist", str(stuck_dist),
    ]
    if loop:
        cmd.append("--loop")
    log = open(REPO_ROOT / "runs" / "trial.log", "w")
    _trial_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    return _trial_proc.pid


@app.post("/api/trial/start")
def trial_start(task: str = "walk", seconds: float = 20.0, realtime: float = 0.5, loop: bool = True):
    """Run playback of the latest policy in a CPU subprocess.

    realtime: 1.0 = real time, 0.5 = half speed (default, more visible),
              0.25 = quarter speed. Independent of MOTOR_SPEED_FRAC,
              this just paces the wall-clock per step.
    seconds: max episode duration in wall-clock seconds.
    loop=true (default): reload latest checkpoint between episodes."""
    global _trial_proc
    if _trial_proc is not None and _trial_proc.poll() is None:
        raise HTTPException(status_code=400, detail="trial already running — stop it first")
    max_steps = max(30, int(seconds / 0.032))
    pid = _spawn_trial(task, max_steps, realtime, loop)
    return {"ok": True, "pid": pid, "loop": loop, "max_steps": max_steps, "seconds": seconds, "realtime": realtime}


@app.post("/api/trial/stop")
def trial_stop():
    global _trial_proc
    if _trial_proc is not None and _trial_proc.poll() is None:
        _trial_proc.terminate()
        try: _trial_proc.wait(timeout=3)
        except subprocess.TimeoutExpired: _trial_proc.kill()
    return {"ok": True}


@app.get("/api/trial/status")
def trial_status():
    if _trial_proc is None or _trial_proc.poll() is not None:
        return {"running": False}
    age = (time.time() - TRIAL_QPOS.stat().st_mtime) if TRIAL_QPOS.exists() else None
    return {"running": True, "pid": _trial_proc.pid, "qpos_age_s": age}


# ───── Manual joint control (used when no training is running) ─────
_HOME_CTRL = None
_LAST_STEP_TIME = None
# pose | physics — pose drives joints kinematically (no gravity, won't blow up
# when you drag a slider). physics runs mj_step with current ctrl as target.
_SIM_MODE = "physics"

# (azimuth, elevation, distance, lookat_z_offset)
_CAMERA_PRESETS = {
    "perspective": (135, -18, 1.6, 0.0),
    "iso":         (45, -30, 2.0, 0.0),
    "top":         (90, -89, 2.5, 0.0),
    "front":       (90, -5, 1.8, 0.0),
    "side":        (180, -5, 1.8, 0.0),
}
_CAMERA_VIEW = "perspective"
# Fixed lookat at the arena mid-point between robot spawn (0,0) and
# tree (3,0,0.8). World stays still, robot moves through frame.
_CAMERA_LOOKAT_FIXED = [1.5, 0.0, 0.4]


@app.get("/api/viewer/views")
def viewer_views():
    return {"views": list(_CAMERA_PRESETS), "current": _CAMERA_VIEW}


@app.post("/api/viewer/view")
def viewer_set_view(name: str):
    global _CAMERA_VIEW
    if name not in _CAMERA_PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown view {name}")
    _CAMERA_VIEW = name
    return {"ok": True, "view": _CAMERA_VIEW}


def _ensure_home():
    """Initialize ctrl to mid-range on first access."""
    global _HOME_CTRL
    model, data = get_model()
    if _HOME_CTRL is None:
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        _HOME_CTRL = ((lo + hi) / 2.0).copy()
        data.ctrl[:] = _HOME_CTRL


def _apply_pose_from_ctrl():
    """Drag-slider feedback: start from the XML's rest pose, then layer
    the user's ctrl on top, joint by joint. Using qpos0 (the model's
    default rest pose) ensures the robot is rendered in a configuration
    where limbs clear the floor — same as look.py shows it."""
    import mujoco
    model, data = get_model()

    # Reset to the XML's default qpos (arms/legs in canonical rest, torso
    # at its design spawn z). Avoids the broken below-floor pose that
    # mid-range ctrl produces.
    data.qpos[:] = model.qpos0

    # Then apply the user's ctrl as joint targets.
    for i in range(model.nu):
        if model.actuator_trntype[i] != 0:
            continue
        jid = int(model.actuator_trnid[i, 0])
        if jid < 0:
            continue
        adr = int(model.jnt_qposadr[jid])
        if model.jnt_type[jid] in (2, 3):
            data.qpos[adr] = float(data.ctrl[i])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


@app.post("/api/sim/mode")
def sim_set_mode(mode: str):
    """Switch manual-sim mode: 'pose' (kinematic) or 'physics' (mj_step)."""
    global _SIM_MODE
    if mode not in ("pose", "physics"):
        raise HTTPException(status_code=400, detail="mode must be 'pose' or 'physics'")
    _SIM_MODE = mode
    return {"ok": True, "mode": _SIM_MODE}


@app.get("/api/sim/mode")
def sim_get_mode():
    return {"mode": _SIM_MODE}


@app.get("/api/sim/joints")
def sim_joints():
    """Actuators with name, range, target ctrl, and *current* joint qpos.

    `ctrl` = where the user (or policy) wants the joint to go.
    `qpos` = where it actually is right now. UI mirrors this back into
    the slider so you can watch the robot move during training.
    """
    import mujoco
    with _SIM_LOCK:
        model, data = get_model()
        _ensure_home()
        # If training is running, overlay the trainer's live qpos so we
        # report what the real robot is doing, not our local stale data.
        task = _read_active(ACTIVE_TASK_FILE, "walk")
        live = _load_live_qpos(task) if _trainer_pids() else None
        if live is not None and live.shape == data.qpos.shape:
            data.qpos[:] = live
            mujoco.mj_forward(model, data)
        out = []
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
            lo, hi = float(model.actuator_ctrlrange[i, 0]), float(model.actuator_ctrlrange[i, 1])
            qpos = None
            if model.actuator_trntype[i] == 0:  # joint actuator
                jid = int(model.actuator_trnid[i, 0])
                if jid >= 0 and model.jnt_type[jid] in (2, 3):
                    qpos = float(data.qpos[int(model.jnt_qposadr[jid])])
            out.append({
                "i": i, "name": name, "lo": lo, "hi": hi,
                "ctrl": float(data.ctrl[i]),
                "qpos": qpos if qpos is not None else float(data.ctrl[i]),
            })
    return {"actuators": out}


@app.post("/api/sim/ctrl")
def sim_ctrl(i: int, value: float):
    """Set one actuator's ctrl value. Takes effect on next /api/viewer/frame."""
    with _SIM_LOCK:
        model, data = get_model()
        _ensure_home()
        if i < 0 or i >= model.nu:
            raise HTTPException(status_code=400, detail=f"actuator {i} out of range")
        lo, hi = float(model.actuator_ctrlrange[i, 0]), float(model.actuator_ctrlrange[i, 1])
        data.ctrl[i] = float(np.clip(value, lo, hi))
        return {"ok": True, "i": i, "value": float(data.ctrl[i])}


@app.post("/api/sim/reset")
def sim_reset():
    """Reset robot to default qpos with mid-range ctrl."""
    import mujoco
    with _SIM_LOCK:
        model, data = get_model()
        mujoco.mj_resetData(model, data)
        _ensure_home()
        data.ctrl[:] = _HOME_CTRL
    return {"ok": True}


@app.post("/api/sim/zero")
def sim_zero():
    """Zero all ctrls (let robot fall / relax)."""
    with _SIM_LOCK:
        model, data = get_model()
        data.ctrl[:] = 0.0
    return {"ok": True}


# ───── System stats ─────
def _liana_pids():
    """All pids that belong to this LIANA stack: dashboard, trainer, viewers."""
    import psutil
    pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if any(k in cmd for k in ("dashboard.server", "train_mjx.py", "uvicorn dashboard")):
                pids.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def _gpu_stats():
    """Parse nvidia-smi for util / mem / per-process memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip().splitlines()[0]
        def _num(s):
            s = s.strip()
            if not s or s.startswith("[") or s.lower() == "n/a":
                return None
            try: return int(s)
            except ValueError:
                try: return int(float(s))
                except ValueError: return None
        util, mem_used, mem_total, temp = [_num(x) for x in out.split(",")]
        proc_out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip()
        procs = []
        if proc_out:
            for line in proc_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    pid = _num(parts[0])
                    mem = _num(parts[2])
                    if pid is None:
                        continue
                    procs.append({"pid": pid, "name": parts[1], "mem_mb": mem or 0})
        return {
            "util_pct": util, "mem_used_mb": mem_used,
            "mem_total_mb": mem_total, "temp_c": temp,
            "processes": procs,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/system/stats")
def system_stats():
    import psutil
    gpu = _gpu_stats()
    total_cpu = psutil.cpu_percent(interval=None)
    total_mem = psutil.virtual_memory()
    liana = []
    for proc in _liana_pids():
        try:
            with proc.oneshot():
                cmd = " ".join(proc.info.get("cmdline") or [])
                role = "dashboard" if "uvicorn" in cmd or "dashboard.server" in cmd else (
                    "trainer" if "train_mjx" in cmd else "other"
                )
                liana.append({
                    "pid": proc.pid,
                    "role": role,
                    "cpu_pct": proc.cpu_percent(interval=None),
                    "mem_mb": int(proc.memory_info().rss / 1_048_576),
                    "cmd": cmd[:120],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        "gpu": gpu,
        "cpu_pct": total_cpu,
        "mem_used_gb": round(total_mem.used / 1e9, 1),
        "mem_total_gb": round(total_mem.total / 1e9, 1),
        "liana_processes": liana,
        "active_task": _read_active(ACTIVE_TASK_FILE, "walk"),
        "active_envs": _read_active(ACTIVE_ENVS_FILE, "auto"),
    }


# ───── Per-task training overview ─────
@app.get("/api/tasks/overview")
def tasks_overview():
    """Scan runs/ape_*_mjx/ to summarize what we know about every task.

    Returns one entry per task with: best mean_rew + step, last update,
    checkpoint count, keeper count. Lets the UI show 'stand vs walk vs
    turn_left' progress at a glance.
    """
    out = []
    for d in sorted((REPO_ROOT / "runs").glob("ape_*_mjx")):
        if not d.is_dir():
            continue
        task = d.name[len("ape_"):-len("_mjx")]
        ckpts = sorted(d.glob("mjx_*.npz"))
        keepers = sorted((d / "keepers").glob("*.npz")) if (d / "keepers").exists() else []
        last_log = d / "train.log"
        best_rew = None
        last_rew = None
        last_step = None
        if last_log.exists():
            try:
                # Parse the structured per-update lines emitted in train_mjx.py.
                # Format: "%8d %12d %8d  rew=%+.2f  d=%.2f  vx=%+.2f  z=%.2f ..."
                tail = subprocess.check_output(
                    ["tail", "-200", str(last_log)], text=True
                ).splitlines()
                for line in tail:
                    m = re.search(r"\s+(\d{1,3}(?:,\d{3})*)\s+\d+\s+rew=([+-]?[\d.]+)", line)
                    if m:
                        step = int(m.group(1).replace(",", ""))
                        rew = float(m.group(2))
                        last_step = step
                        last_rew = rew
                        if best_rew is None or rew > best_rew:
                            best_rew = rew
            except Exception:
                pass
        latest_ts = max((p.stat().st_mtime for p in ckpts), default=None)
        out.append({
            "task": task,
            "checkpoints": len(ckpts),
            "keepers": len(keepers),
            "last_rew": last_rew,
            "best_rew": best_rew,
            "last_step": last_step,
            "latest_ts": latest_ts,
            "latest_iso": time.strftime("%Y-%m-%d %H:%M", time.localtime(latest_ts)) if latest_ts else None,
        })
    return {"tasks": out}


# ───── Scene selection ─────
SCENES_DIR = REPO_ROOT / "data"
SCENES_ARCHIVE = REPO_ROOT / "data" / "scenes_archive"
ACTIVE_SCENE_FILE = REPO_ROOT / "runs" / ".active_scene"


@app.get("/api/scenes/list")
def scenes_list():
    """List all scene*.xml files in data/ and data/scenes_archive/."""
    items = []
    for p in sorted(SCENES_DIR.glob("scene*.xml")):
        items.append({"name": p.name, "path": str(p.relative_to(REPO_ROOT)), "active": p == SCENES_DIR / "scene.xml"})
    for p in sorted(SCENES_ARCHIVE.glob("scene*.xml")):
        items.append({"name": "archive/" + p.name, "path": str(p.relative_to(REPO_ROOT)), "active": False})
    return {"scenes": items, "current": _read_active(ACTIVE_SCENE_FILE, "data/scene.xml")}


@app.post("/api/scenes/select")
def scenes_select(path: str):
    """Switch the active scene. Refuses while training is running.

    - Tries to load the candidate XML BEFORE clobbering data/scene.xml.
      If MuJoCo can't compile it (missing meshes, broken refs, …) we
      bail out with a 400 instead of corrupting the active scene.
    - Backs the current scene up to runs/.scene_backup.xml so /api/scenes/revert
      can recover.
    """
    import shutil
    import mujoco
    if _trainer_pids():
        raise HTTPException(status_code=400, detail="stop training first — scene swap would corrupt the trainer's view")
    src = REPO_ROOT / path
    if not src.exists() or src.suffix != ".xml":
        raise HTTPException(status_code=404, detail=f"scene not found: {path}")
    target = SCENES_DIR / "scene.xml"

    # Validate: try to load the candidate XML in isolation. Catches missing
    # meshes, bad meshdir, syntax errors before they clobber the active scene.
    try:
        mujoco.MjModel.from_xml_path(str(src))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"candidate scene won't load: {e}")

    if src.resolve() != target.resolve():
        # Back up before overwriting so /api/scenes/revert works.
        backup = REPO_ROOT / "runs" / ".scene_backup.xml"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        shutil.copy2(src, target)
    ACTIVE_SCENE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_SCENE_FILE.write_text(str(target.relative_to(REPO_ROOT)))
    # Invalidate the cached model so next /api/viewer/frame reloads.
    global _model_cache, _data_cache, _renderer_cache, _HOME_CTRL
    with _SIM_LOCK:
        _model_cache = None
        _data_cache = None
        _renderer_cache = None
        _HOME_CTRL = None
    return {"ok": True, "active": path}


@app.post("/api/scenes/revert")
def scenes_revert():
    """Restore data/scene.xml from the last backup written by scenes_select."""
    import shutil
    backup = REPO_ROOT / "runs" / ".scene_backup.xml"
    if not backup.exists():
        raise HTTPException(status_code=404, detail="no backup to revert to")
    target = SCENES_DIR / "scene.xml"
    shutil.copy2(backup, target)
    global _model_cache, _data_cache, _renderer_cache, _HOME_CTRL
    with _SIM_LOCK:
        _model_cache = None
        _data_cache = None
        _renderer_cache = None
        _HOME_CTRL = None
    return {"ok": True}


@app.get("/api/viewer/frame")
def get_frame(width: int = 640, height: int = 480):
    """Return the latest frame the stream thread rendered.

    Does NOT render itself — that would fight the stream for the EGL
    context. If the stream hasn't produced a frame yet, returns the
    1x1 placeholder.
    """
    global _LAST_FRAME_JPEG
    if _LAST_FRAME_JPEG:
        return Response(content=_LAST_FRAME_JPEG, media_type="image/jpeg")
    # No frame yet — fall back to legacy direct render only on cold start.
    try:
        import mujoco
        with _SIM_LOCK:
            model, data = get_model()
            renderer = get_renderer()

            task = _read_active(ACTIVE_TASK_FILE, "walk")
            live = _load_live_qpos(task) if _trainer_pids() else None
            if live is not None and live.shape == data.qpos.shape:
                # Training is active — show the policy's actual state.
                data.qpos[:] = live
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            else:
                # No training — run a free-running sim driven by /api/sim/ctrl sliders.
                _ensure_home()
                global _LAST_STEP_TIME
                now = time.time()
                dt = min(0.1, now - _LAST_STEP_TIME) if _LAST_STEP_TIME else model.opt.timestep * 20
                _LAST_STEP_TIME = now
                n_steps = max(1, int(dt / model.opt.timestep))
                for _ in range(min(n_steps, 50)):
                    mujoco.mj_step(model, data)

            cam = mujoco.MjvCamera()
            cam.distance = 3.0
            cam.azimuth = 90
            cam.elevation = -20
            cam.lookat = [0, 0, 1]

            renderer.update_scene(data, camera=cam)
            img = renderer.render()

        # Convert to PNG
        from PIL import Image

        img_pil = Image.fromarray(img)
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        # Surface the error in headers so we can diagnose without swallowing it.
        import traceback
        err = f"{type(e).__name__}: {e}"
        sys.stderr.write(f"[viewer] render failed: {err}\n{traceback.format_exc()}\n")
        sys.stderr.flush()
        return Response(
            content=_TRANSPARENT_PNG, media_type="image/png",
            headers={"X-Viewer-Error": err[:200]},
        )


_IDLE_FRAME_CACHE: bytes | None = None


def _idle_frame() -> bytes:
    """A static dark placeholder shown when no training is running.

    Don't render the robot pre-Start — user explicitly doesn't want
    to see it. Generates a black JPEG with a tiny status line, once,
    cached forever.
    """
    global _IDLE_FRAME_CACHE
    if _IDLE_FRAME_CACHE is not None:
        return _IDLE_FRAME_CACHE
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1280, 960), (8, 8, 8))
    d = ImageDraw.Draw(img)
    msg = "PRESS START TO TRAIN"
    # Approx-center the text without bothering with font metrics.
    d.text((520, 470), msg, fill=(110, 110, 110))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    _IDLE_FRAME_CACHE = buf.getvalue()
    return _IDLE_FRAME_CACHE


def _render_png_bytes() -> bytes:
    """Single rendered frame as JPEG bytes — used by /stream ONLY."""
    import mujoco
    from PIL import Image
    global _RENDER_OWNER_THREAD, _renderer_cache
    # Viewer shows the TRIAL only. Never trainer env-0 (too jerky).
    # If the trial isn't running, return the idle placeholder.
    trial_running = _trial_proc is not None and _trial_proc.poll() is None
    if not trial_running:
        return _idle_frame()
    tid = threading.get_ident()
    with _SIM_LOCK:
        if _RENDER_OWNER_THREAD is not None and _RENDER_OWNER_THREAD != tid:
            _renderer_cache = None
        _RENDER_OWNER_THREAD = tid
        model, data = get_model()
        renderer = get_renderer()
        task = _read_active(ACTIVE_TASK_FILE, "walk")
        live = _load_live_qpos(task) if _trainer_pids() else None
        if live is not None and live.shape == data.qpos.shape:
            data.qpos[:] = live
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        elif _SIM_MODE == "pose":
            # Pose mode: qpos ← ctrl directly. No gravity, won't blow up while
            # you drag sliders. Instant visual feedback.
            _ensure_home()
            _apply_pose_from_ctrl()
        else:
            # Physics mode: actually simulate with current ctrl as target.
            _ensure_home()
            global _LAST_STEP_TIME
            now = time.time()
            dt = min(0.1, now - _LAST_STEP_TIME) if _LAST_STEP_TIME else model.opt.timestep * 20
            _LAST_STEP_TIME = now
            n_steps = max(1, int(dt / model.opt.timestep))
            for _ in range(min(n_steps, 50)):
                mujoco.mj_step(model, data)
            # Recover if integration blew up.
            if not np.all(np.isfinite(data.qpos)):
                mujoco.mj_resetData(model, data)
                _apply_pose_from_ctrl()
        cam = mujoco.MjvCamera()
        az, el, dist, dz = _CAMERA_PRESETS.get(_CAMERA_VIEW, _CAMERA_PRESETS["perspective"])
        cam.lookat = [_CAMERA_LOOKAT_FIXED[0], _CAMERA_LOOKAT_FIXED[1], _CAMERA_LOOKAT_FIXED[2] + dz]
        cam.distance = dist * 1.6  # pull back so spawn + tree both fit
        cam.azimuth = az
        cam.elevation = el
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=92, optimize=False)
    return buf.getvalue()


@app.get("/api/viewer/stream")
def viewer_stream(fps: int = 20):
    """MJPEG stream — one connection, server pushes frames at `fps` Hz.

    Browser sets <img src="/api/viewer/stream"> and the connection stays
    open. Far cheaper than polling /api/viewer/frame every 100ms.
    """
    fps = max(1, min(60, int(fps)))
    interval = 1.0 / fps
    boundary = b"liana-frame"

    last_err = [None, 0]  # (msg, count)

    def gen():
        global _LAST_FRAME_JPEG, _LAST_FRAME_TS
        while True:
            t0 = time.time()
            try:
                jpg = _render_png_bytes()
                _LAST_FRAME_JPEG = jpg
                _LAST_FRAME_TS = t0
                last_err[1] = 0
            except Exception as e:
                # Rate-limit: only log when the error changes or every 50 occurrences.
                msg = f"{type(e).__name__}: {e}"[:200]
                if msg != last_err[0]:
                    last_err[0] = msg
                    last_err[1] = 0
                    sys.stderr.write(f"[stream] render failed: {msg}\n")
                last_err[1] += 1
                if last_err[1] % 50 == 0:
                    sys.stderr.write(f"[stream] same error x{last_err[1]}\n")
                jpg = _LAST_FRAME_JPEG or b""
                # Drop the cached renderer so the next attempt rebuilds it
                # on whichever thread we're on. Most EGL errors recover this way.
                global _renderer_cache
                with _SIM_LOCK:
                    _renderer_cache = None
            if jpg:
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg + b"\r\n"
                )
            sleep_for = interval - (time.time() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
    )


@app.get("/api/whats-happening")
def whats_happening():
    """Plain-English summary of viewer state + training state.

    Always returns BOTH sides so the UI can show them side by side and
    the user never has to wonder which subsystem they're looking at.
    """
    task = _read_active(ACTIVE_TASK_FILE, "walk")
    trial_running = _trial_proc is not None and _trial_proc.poll() is None
    trainer_pids = _trainer_pids()
    training_running = bool(trainer_pids)
    qpos_path = REPO_ROOT / "runs" / f"ape_{task}_mjx" / "live_qpos.npy"
    qpos_age = (time.time() - qpos_path.stat().st_mtime) if qpos_path.exists() else None

    # VIEWER: what's the user looking at?
    if trial_running:
        info = trial_info() if TRIAL_INFO.exists() else {}
        ckpt = info.get("checkpoint", "random policy")
        viewer = {
            "label": f"TRIAL · {ckpt}",
            "detail": "Real-time single robot. Reloads the best checkpoint between episodes.",
            "color": "ok",
        }
    elif training_running:
        viewer = {
            "label": "TRAINING IN BACKGROUND",
            "detail": "Trainer is running but no trial yet. Click Replay (or wait for auto-trial).",
            "color": "warn",
        }
    else:
        viewer = {
            "label": "IDLE",
            "detail": "Press Start to launch trainer + auto-playback.",
            "color": "muted",
        }

    # TRAINING: what's actually being trained?
    if training_running:
        envs_val = _read_active(ACTIVE_ENVS_FILE, "auto")
        training = {
            "label": f"RUNNING · task={task} · envs={envs_val}",
            "detail": f"PID {trainer_pids[0]}. GPU MJX. Auto-stops at 90% success or on user Stop.",
            "color": "ok",
        }
    else:
        training = {
            "label": "NOT RUNNING",
            "detail": "No policy is improving right now. Press Start.",
            "color": "muted",
        }

    return {"viewer": viewer, "training": training, "task": task}


@app.get("/api/viewer/info")
def viewer_info():
    """Tells the UI what the viewer is actually showing right now:

    - idle      : nothing running — placeholder frame
    - trial     : single-trial subprocess running, real-time
    - compiling : trainer running but no fresh qpos yet (MJX JIT)
    - live      : trainer running and qpos is being updated
    """
    task = _read_active(ACTIVE_TASK_FILE, "walk")
    p = REPO_ROOT / "runs" / f"ape_{task}_mjx" / "live_qpos.npy"
    age = None
    if p.exists():
        age = time.time() - p.stat().st_mtime
    trial_running = _trial_proc is not None and _trial_proc.poll() is None
    if trial_running:
        state = "trial"
    elif not _trainer_pids():
        state = "idle"
    elif age is None or age > 3.0:
        state = "compiling"
    else:
        state = "live"
    return {"state": state, "task": task, "live_age_s": age, "mode": _SIM_MODE}


@app.get("/api/viewer/state")
def get_viewer_state():
    """Get viewer state - joint positions, body positions."""
    try:
        import mujoco

        with _SIM_LOCK:
            model, data = get_model()
            mujoco.mj_forward(model, data)

            qpos = data.qpos.tolist() if hasattr(data, "qpos") else []

            body_pos = {}
            for i in range(min(model.nbody, 10)):
                try:
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if name:
                        body_pos[name] = {
                            "pos": data.xpos[i].tolist() if hasattr(data, "xpos") else []
                        }
                except:
                    pass

            return {
                "qpos": qpos,
                "bodies": body_pos,
                "time": float(data.time) if hasattr(data, "time") else 0.0,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "GibbonBot Dashboard API",
        "docs": "/docs",
        "static": "/static/index.html",
        "endpoints": {
            "training": [
                "/api/train/start",
                "/api/train/stop",
                "/api/train/status",
                "/api/train/metrics",
                "/api/train/logs",
            ],
            "brain": ["/api/brain/modules", "/api/brain/state", "/api/brain/memory"],
            "viewer": ["/api/viewer/frame", "/api/viewer/state"],
        },
        "note": "For MJX/Warp rendering, set MUJOCO_BACKEND=mjx before starting the server",
    }
