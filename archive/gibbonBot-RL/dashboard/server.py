"""
GibbonBot Dashboard Server
FastAPI backend for training control, brain state retrieval, and MuJoCo viewer frames.

Run: uvicorn dashboard.server:app --reload --port 8000
Access: http://localhost:8000/static/index.html

Note: For MJX/Warp rendering, frames are generated on-demand.
"""

import os
import re
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
            _renderer_cache = mujoco.Renderer(_model_cache, height=480, width=640)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return _model_cache, _data_cache


def get_renderer():
    """Get cached renderer."""
    global _renderer_cache
    if _renderer_cache is None:
        get_model()  # This will initialize the renderer
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
    """Invoke scripts/train_bg.sh — returns {ok, stdout, stderr}."""
    res = subprocess.run(
        ["bash", str(TRAIN_BG), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    return {"ok": res.returncode == 0, "stdout": res.stdout, "stderr": res.stderr}


@app.post("/api/train/start")
def start_training(task: str = "walk", envs: int | None = None):
    if _trainer_pids():
        raise HTTPException(status_code=400, detail="Trainer already running — use /api/train/restart")
    args = ["start", task]
    if envs:
        args += ["--envs", str(envs)]
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
def restart_training(task: str | None = None, envs: int | None = None):
    """Stop + start with new task / num_envs. Required for changing batch size mid-run
    (MJX recompiles on batch-shape change)."""
    cur_task = _read_active(ACTIVE_TASK_FILE, "walk")
    cur_envs = _read_active(ACTIVE_ENVS_FILE, "")
    new_task = task or cur_task
    new_envs = envs if envs is not None else (int(cur_envs) if cur_envs else None)
    _bg("stop")
    # Give kernel a beat to release the GPU.
    time.sleep(2)
    args = ["start", new_task]
    if new_envs:
        args += ["--envs", str(new_envs)]
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


@app.get("/api/viewer/frame")
def get_frame(width: int = 640, height: int = 480):
    """Get frame as PNG showing the current trained robot.

    Reads runs/ape_<task>_mjx/live_qpos.npy (dumped by the trainer each batch),
    applies it to the cached model, and renders. Falls back to the static
    scene if no dump is available yet.
    """
    try:
        import mujoco

        model, data = get_model()
        renderer = get_renderer()

        task = _read_active(ACTIVE_TASK_FILE, "walk")
        live = _load_live_qpos(task)
        if live is not None and live.shape == data.qpos.shape:
            data.qpos[:] = live
            data.qvel[:] = 0.0

        # Step simulation forward
        mujoco.mj_forward(model, data)

        # Set up camera for good view
        cam = mujoco.MjvCamera()
        cam.distance = 3.0
        cam.azimuth = 90
        cam.elevation = -20
        cam.lookat = [0, 0, 1]

        # Render scene
        renderer.update_scene(data, camera=cam)
        img = renderer.render()

        # Convert to PNG
        from PIL import Image

        img_pil = Image.fromarray(img)
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        # Return transparent 1x1 PNG as fallback
        return Response(content=_TRANSPARENT_PNG, media_type="image/png")


@app.get("/api/viewer/state")
def get_viewer_state():
    """Get viewer state - joint positions, body positions."""
    try:
        import mujoco

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
