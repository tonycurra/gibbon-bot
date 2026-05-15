"""
GibbonBot Dashboard Server
FastAPI backend for training control, brain state retrieval, and MuJoCo viewer.

Run: uvicorn dashboard.server:app --reload --port 8000
Access: http://localhost:8000/static/index.html
"""

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

# ============================================================================
# TRAINING API
# ============================================================================


@app.post("/api/train/start")
def start_training(task: str = "stand"):
    global training_process
    if training_process and training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training already running")
    cmd = [sys.executable, "train_mjx.py", "--task", task, "--no-watch"]
    training_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return {"status": "started", "task": task, "pid": training_process.pid}


@app.post("/api/train/stop")
def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        training_process.wait(timeout=10)
        return {"status": "stopped"}
    return {"status": "not running"}


@app.get("/api/train/status")
def get_status():
    global training_process
    if training_process:
        return {"running": training_process.poll() is None, "pid": training_process.pid}
    return {"running": False}


@app.get("/api/train/metrics")
def get_metrics():
    metrics = {}
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
                    "fields": vars(module.state)
                    if hasattr(module.state, "__dict__")
                    else str(module.state),
                }
            elif hasattr(module, "get_state"):
                state[name] = str(module.get_state())
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

_model_cache = None
_data_cache = None


def get_model():
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
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return _model_cache, _data_cache


@app.get("/api/viewer/frame")
def get_frame(width: int = 640, height: int = 480):
    """Get frame as PNG - Returns transparent 1x1 PNG placeholder.

    To enable real rendering, implement offscreen GL context or use MJX backend.
    """
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x00IDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


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
    }
