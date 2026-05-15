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
        "/api/tasks/overview",
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
    """Load or cache MuJoCo model."""
    global _model_cache, _data_cache, _renderer_cache
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
            # Reduce collision for faster rendering
            for i in range(_model_cache.ngeom):
                if _model_cache.geom_type[i] == 5:  # mjGEOM_MESH
                    _model_cache.geom_contype[i] = 2
                    _model_cache.geom_conaffinity[i] = 1
            # Create renderer (uses OpenGL, works on systems with GL support)
            # 1280x960 (4:3) is plenty for the dashboard's viewer card —
            # higher resolutions make EGL slow without visible benefit.
            _renderer_cache = mujoco.Renderer(_model_cache, height=960, width=1280)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return _model_cache, _data_cache


def get_renderer():
    """Get cached renderer; rebuild if it was invalidated for thread/EGL reasons."""
    global _renderer_cache
    if _renderer_cache is None:
        import mujoco
        model, _ = get_model()
        _renderer_cache = mujoco.Renderer(model, height=960, width=1280)
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
    env = {
        **os.environ,
        "ALLOW_GPU_SHARE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
    }
    res = subprocess.run(
        ["bash", str(TRAIN_BG), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60, env=env,
    )
    return {"ok": res.returncode == 0, "stdout": res.stdout, "stderr": res.stderr}


@app.post("/api/train/start")
def start_training(task: str = "walk", envs: int | None = None, total_steps: int | None = None, resume: bool = True):
    if _trainer_pids():
        raise HTTPException(status_code=400, detail="Trainer already running — use /api/train/restart")
    args = ["start", task]
    if envs:
        args += ["--envs", str(envs)]
    if total_steps:
        args += ["--total-steps", str(total_steps)]
    if resume:
        # Don't trash checkpoints on minor scene/param tweaks.
        args += ["--resume"]
    res = _bg(*args)
    if not res["ok"]:
        raise HTTPException(status_code=500, detail=res["stderr"] or res["stdout"])
    ACTIVE_TASK_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_TASK_FILE.write_text(task)
    if envs:
        ACTIVE_ENVS_FILE.write_text(str(envs))
    return {"status": "started", "task": task, "envs": envs, "detail": res["stdout"].strip()}


@app.post("/api/train/stop")
def stop_training():
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
    if total_steps:
        args += ["--total-steps", str(total_steps)]
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
_PARAM_KEYS = {
    "MOTOR_SPEED_FRAC": (0.05, 1.5),
    "MOTOR_ACCEL_RATIO": (0.5, 5.0),
    "ACTION_SMOOTH_ALPHA": (0.0, 1.0),
    "W_UPRIGHT": (0.0, 50.0),
    "W_HEIGHT": (0.0, 50.0),
    "W_PROGRESS": (0.0, 500.0),
    "W_SMOOTH": (0.0, 5.0),
    "FLIP_PENALTY": (0.0, 500.0),
}


def _params_path(task: str) -> Path:
    p = REPO_ROOT / "envs" / f"params_{task}_mjx.py"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No params file for task '{task}'")
    return p


@app.get("/api/params/get")
def params_get(task: str = "walk"):
    src = _params_path(task).read_text()
    out = {}
    for key in _PARAM_KEYS:
        m = re.search(rf"^{key}\s*=\s*([0-9.+\-eE]+)", src, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return {"task": task, "params": out, "bounds": _PARAM_KEYS}


@app.post("/api/params/set")
def params_set(task: str = "walk", key: str = "", value: float = 0.0):
    if key not in _PARAM_KEYS:
        raise HTTPException(status_code=400, detail=f"Key '{key}' not editable. Allowed: {list(_PARAM_KEYS)}")
    lo, hi = _PARAM_KEYS[key]
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


def _load_live_qpos(task: str):
    """Read the latest qpos snapshot the trainer dumps each PPO batch."""
    p = REPO_ROOT / "runs" / f"ape_{task}_mjx" / "live_qpos.npy"
    if not p.exists():
        return None
    try:
        return np.load(str(p))
    except Exception:
        return None


# ───── Manual joint control (used when no training is running) ─────
_HOME_CTRL = None
_LAST_STEP_TIME = None
# pose | physics — pose drives joints kinematically (no gravity, won't blow up
# when you drag a slider). physics runs mj_step with current ctrl as target.
_SIM_MODE = "pose"

# (azimuth, elevation, distance, lookat_z_offset)
_CAMERA_PRESETS = {
    "perspective": (135, -18, 1.6, 0.0),
    "iso":         (45, -30, 2.0, 0.0),
    "top":         (90, -89, 2.5, 0.0),
    "front":       (90, -5, 1.8, 0.0),
    "side":        (180, -5, 1.8, 0.0),
}
_CAMERA_VIEW = "perspective"


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
    """Set qpos directly from ctrl so dragging sliders shows the pose instantly.

    Also grounds the torso at a sensible standing height (0.30m) so the
    robot isn't rendered floating in mid-air at the XML's default spawn.
    Skips actuators that aren't bound to a single hinge/slide joint.
    """
    import mujoco
    model, data = get_model()
    # Ground the free-flying torso (qpos[0:7] = xyz + quat for the world_to_torso joint).
    if model.nq >= 7 and model.jnt_type[0] == 0:  # 0 = free joint
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.30
        data.qpos[3] = 1.0  # quat w
        data.qpos[4:7] = 0.0  # quat x,y,z
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


def _render_png_bytes() -> bytes:
    """Single rendered frame as JPEG bytes — used by /stream ONLY.

    Tracks which thread "owns" the EGL context; if a different thread
    sneaks in (request handlers etc.), it'll force a renderer rebuild on
    that thread to recover, but normal traffic stays on the stream thread.
    """
    import mujoco
    from PIL import Image
    global _RENDER_OWNER_THREAD, _renderer_cache
    tid = threading.get_ident()
    with _SIM_LOCK:
        # If a different thread is trying to render, drop the existing
        # renderer (EGL context is bound to the previous thread).
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
        torso_xy = data.qpos[0:2] if data.qpos.shape[0] >= 3 else np.zeros(2)
        torso_z = float(data.qpos[2]) if data.qpos.shape[0] >= 3 else 0.3
        az, el, dist, dz = _CAMERA_PRESETS.get(_CAMERA_VIEW, _CAMERA_PRESETS["perspective"])
        cam.lookat = [float(torso_xy[0]), float(torso_xy[1]), torso_z + dz]
        cam.distance = dist
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


@app.get("/api/viewer/info")
def viewer_info():
    """Tells the UI what the viewer is actually showing right now:

    - manual    : no trainer, sim driven by slider ctrl
    - compiling : trainer running but no fresh qpos yet (MJX JIT)
    - live      : trainer running and qpos is being updated
    """
    task = _read_active(ACTIVE_TASK_FILE, "walk")
    p = REPO_ROOT / "runs" / f"ape_{task}_mjx" / "live_qpos.npy"
    age = None
    if p.exists():
        age = time.time() - p.stat().st_mtime
    if not _trainer_pids():
        state = "manual"
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
