"""
GibbonBot Dashboard Server
FastAPI backend for training control, brain state retrieval, and MuJoCo viewer frames.

Run: uvicorn dashboard.server:app --reload --port 8000
Access: http://localhost:8000/static/index.html

Note: For MJX/Warp rendering, frames are generated on-demand.
"""

import os
import subprocess
import sys
import time
from io import BytesIO
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


@app.get("/api/viewer/frame")
def get_frame(width: int = 640, height: int = 480):
    """Get frame as PNG.

    Uses mujoco.Renderer which works with standard OpenGL.
    Falls back to transparent PNG if rendering fails.
    """
    try:
        import mujoco

        model, data = get_model()
        renderer = get_renderer()

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
