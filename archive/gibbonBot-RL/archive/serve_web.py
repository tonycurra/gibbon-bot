#!/usr/bin/env python3
"""Stream MuJoCo scene to browser as MJPEG with camera control.

Usage:
    source .venv/bin/activate
    python serve_web.py
    # open http://localhost:8080 in your browser

Controls (browser):
    Left-drag   orbit camera
    Scroll      zoom in/out
    Space / button  pause/play physics
"""

import io
import json
import math
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

# ── model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "data/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
sim_lock = threading.Lock()
paused = False

# camera shared with all clients (a single global view)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance  = 3.0
cam.azimuth   = 135.0
cam.elevation = -20.0

# latest rendered JPEG, written by the render thread, read by HTTP handlers
H, W = 1280, 1280      # square — works portrait + landscape without ugly letterboxing
FPS = 20
JPEG_Q = 85
frame_cond = threading.Condition()
latest_jpeg: bytes = b""
frame_id = 0
viewers = 0            # active stream/poll clients
chat_log: list[dict] = []   # [{t, text}]
CHAT_MAX = 50

# current terrain params (mirrors gen_terrain.py defaults)
terrain_params = {"slope": 15.0, "roughness": 0.08, "softness": 0.30, "grip": 0.70,
                  "sun_elev": 60.0, "sun_azim": 135.0}
reload_request: dict | None = None
reload_status = "idle"   # 'idle' | 'building' | 'error: ...'
terrain_top = 0.0        # max world-z of heightfield surface (for camera clamp)

# current grove params (mirror trees.generate defaults)
grove_params = {"variety": "mature", "size": 1.0, "branches": 5,
                "leaves": 0.6, "detachment": 0.3, "seed": 42}
grove_request: dict | None = None
grove_status = "idle"


def _terrain_top(m: mujoco.MjModel) -> float:
    """World-space max-z of the heightfield surface (used for camera init only)."""
    if m.nhfield == 0:
        return 0.0
    nrow = int(m.hfield_nrow[0]); ncol = int(m.hfield_ncol[0])
    zmax = float(m.hfield_size[0][2])
    base_z = float(m.geom_pos[m.geom("floor").id][2]) if "floor" in [m.geom(i).name for i in range(m.ngeom)] else 0.0
    hf = np.asarray(m.hfield_data).reshape(nrow, ncol)
    return base_z + float(hf.max()) * zmax


def _terrain_height_at(m: mujoco.MjModel, wx: float, wy: float) -> float:
    """Bilinear-interpolated terrain height at world position (wx, wy)."""
    if m.nhfield == 0:
        return 0.0
    nrow = int(m.hfield_nrow[0]); ncol = int(m.hfield_ncol[0])
    size = m.hfield_size[0]
    x_half, y_half, zmax = float(size[0]), float(size[1]), float(size[2])
    try:
        base_z = float(m.geom_pos[m.geom("floor").id][2])
    except Exception:
        base_z = 0.0
    u = max(0.0, min(1.0, (wx + x_half) / (2 * x_half)))
    v = max(0.0, min(1.0, (wy + y_half) / (2 * y_half)))
    col_f = u * (ncol - 1);  c0 = int(col_f);  c1 = min(c0 + 1, ncol - 1);  tc = col_f - c0
    row_f = (1.0 - v) * (nrow - 1);  r0 = int(row_f);  r1 = min(r0 + 1, nrow - 1);  tr = row_f - r0
    hf = np.asarray(m.hfield_data).reshape(nrow, ncol)
    h = (hf[r0, c0]*(1-tc)*(1-tr) + hf[r0, c1]*tc*(1-tr) +
         hf[r1, c0]*(1-tc)*tr     + hf[r1, c1]*tc*tr)
    return base_z + h * zmax


def _clamp_camera(c: mujoco.MjvCamera, m: mujoco.MjModel) -> None:
    """Reduce camera distance so the camera stays above the terrain surface."""
    clearance = 0.20
    az  = math.radians(c.azimuth)
    el  = math.radians(c.elevation)
    sin_el = math.sin(el)
    # camera world position
    cam_x = c.lookat[0] - c.distance * math.cos(el) * math.cos(az)
    cam_y = c.lookat[1] - c.distance * math.cos(el) * math.sin(az)
    cam_z = c.lookat[2] - c.distance * sin_el
    terrain_h = _terrain_height_at(m, cam_x, cam_y)
    if cam_z < terrain_h + clearance and abs(sin_el) > 0.02:
        # cam_z = lookat_z - dist*sin_el >= terrain_h + clearance
        # → dist <= (lookat_z - terrain_h - clearance) / sin_el
        max_d = (c.lookat[2] - terrain_h - clearance) / sin_el
        c.distance = max(0.5, min(c.distance, max_d))


# ── physics thread ─────────────────────────────────────────────────────────
def _physics_loop():
    dt = model.opt.timestep
    while True:
        if not paused:
            with sim_lock:
                mujoco.mj_step(model, data)
        time.sleep(dt)


# ── render thread (owns the GL context) ────────────────────────────────────
def _render_loop():
    """Renderer's GL context lives in this thread — never touch it elsewhere.
    Idles when no client is watching; also handles terrain hot-reload here so
    the GL context and model stay in sync.
    """
    global latest_jpeg, frame_id, model, data
    global reload_request, reload_status, grove_request, grove_status, terrain_top
    renderer = mujoco.Renderer(model, H, W)
    terrain_top = _terrain_top(model)
    period = 1.0 / FPS

    def _params_to_cli(params: dict) -> list[str]:
        out = []
        for k, v in params.items():
            flag = f"--{k.replace('_','-')}"
            if isinstance(v, (list, tuple)):
                out += [flag, *(str(x) for x in v)]
            else:
                out += [flag, str(v)]
        return out

    def _swap_model() -> mujoco.Renderer:
        """Reload model from disk, swap globals, recreate renderer."""
        nonlocal renderer
        global model, data, terrain_top
        with sim_lock:
            new_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
            new_data = mujoco.MjData(new_model)
            mujoco.mj_resetDataKeyframe(new_model, new_data, 0)
            mujoco.mj_forward(new_model, new_data)
            model = new_model
            data = new_data
        renderer.close()
        renderer = mujoco.Renderer(model, H, W)
        terrain_top = _terrain_top(model)
        _clamp_camera(cam, model)
        return renderer

    while True:
        # handle pending terrain reload
        if reload_request is not None:
            params = reload_request
            reload_request = None
            reload_status = "building"
            try:
                cmd = [sys.executable, "gen_terrain.py"] + _params_to_cli(params)
                subprocess.run(cmd, check=True, capture_output=True, timeout=15)
                _swap_model()
                terrain_params.update(params)
                reload_status = "idle"
            except subprocess.CalledProcessError as e:
                reload_status = f"error: {e.stderr.decode()[:120] if e.stderr else e}"
                print(f"[terrain] {reload_status}")
            except Exception as e:
                reload_status = f"error: {e}"
                print(f"[terrain] {reload_status}")

        # handle pending grove reload
        if grove_request is not None:
            params = grove_request
            grove_request = None
            grove_status = "building"
            try:
                cmd = [sys.executable, "-m", "trees.generate"] + _params_to_cli(params)
                subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                _swap_model()
                grove_params.update(params)
                grove_status = "idle"
            except subprocess.CalledProcessError as e:
                grove_status = f"error: {e.stderr.decode()[:120] if e.stderr else e}"
                print(f"[grove] {grove_status}")
            except Exception as e:
                grove_status = f"error: {e}"
                print(f"[grove] {grove_status}")

        # gate: nothing to do if nobody is watching
        with frame_cond:
            while viewers == 0:
                frame_cond.wait()

        t0 = time.perf_counter()
        with sim_lock:
            renderer.update_scene(data, cam)
        rgb = renderer.render()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_Q)
        with frame_cond:
            latest_jpeg = buf.getvalue()
            frame_id += 1
            frame_cond.notify_all()
        slack = period - (time.perf_counter() - t0)
        if slack > 0:
            time.sleep(slack)


threading.Thread(target=_physics_loop, daemon=True).start()
threading.Thread(target=_render_loop,  daemon=True).start()


# ── HTTP handler ──────────────────────────────────────────────────────────
BOUNDARY = b"mjframe"
HTML = Path("web/viewer.html").read_bytes()


def _viewer_join():
    global viewers
    with frame_cond:
        viewers += 1
        frame_cond.notify_all()       # wake render thread


def _viewer_leave():
    global viewers
    with frame_cond:
        viewers = max(0, viewers - 1)


def _wait_first_frame(timeout=5.0):
    with frame_cond:
        if frame_id == 0:
            frame_cond.wait(timeout)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._reply(200, "text/html", HTML)

        elif self.path == "/frame.jpg":
            _viewer_join()
            try:
                _wait_first_frame()
                with frame_cond:
                    jpg = latest_jpeg
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                self.wfile.write(jpg)
            finally:
                _viewer_leave()

        elif self.path == "/stream":
            _viewer_join()
            try:
                _wait_first_frame()
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
                )
                self.end_headers()
                last_id = -1
                while True:
                    with frame_cond:
                        while frame_id == last_id:
                            frame_cond.wait(timeout=1.0)
                        jpg = latest_jpeg
                        last_id = frame_id
                    self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                _viewer_leave()

        elif self.path == "/state":
            with sim_lock:
                try:
                    torso_id = model.body("torso").id
                    torso_pos = list(data.xpos[torso_id])
                except Exception:
                    torso_pos = [0.0, 0.0, 0.5]
            body = json.dumps(
                {
                    "azimuth":   cam.azimuth,
                    "elevation": cam.elevation,
                    "distance":  cam.distance,
                    "lookat":    list(cam.lookat),
                    "paused":    paused,
                    "fps":       FPS,
                    "chat":      chat_log[-20:],
                    "terrain":   terrain_params,
                    "reload":    reload_status,
                    "torso":     torso_pos,
                    "grove":     grove_params,
                    "grove_status": grove_status,
                }
            ).encode()
            self._reply(200, "application/json", body)

        else:
            self.send_error(404)

    def do_POST(self):
        global paused
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}

        if self.path == "/camera":
            if "azimuth"   in body: cam.azimuth   = body["azimuth"]
            if "elevation" in body: cam.elevation = max(-89, min(89, body["elevation"]))
            if "distance"  in body: cam.distance  = max(0.5, min(20, body["distance"]))
            if "lookat"    in body and len(body["lookat"]) == 3:
                cam.lookat[:] = body["lookat"]
            _clamp_camera(cam, model)
            self._reply(200, "application/json", b"{}")

        elif self.path == "/pause":
            paused = not paused
            self._reply(200, "application/json", json.dumps({"paused": paused}).encode())

        elif self.path == "/reset":
            with sim_lock:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            self._reply(200, "application/json", b"{}")

        elif self.path == "/terrain":
            global reload_request
            allowed = {"slope", "roughness", "softness", "grip", "sun_elev", "sun_azim"}
            params = {k: float(v) for k, v in body.items() if k in allowed}
            # merge with current so partial updates work
            merged = {**terrain_params, **params}
            reload_request = merged
            with frame_cond:
                frame_cond.notify_all()    # wake render thread if idle
            self._reply(200, "application/json", json.dumps({"queued": merged}).encode())

        elif self.path == "/grove":
            global grove_request
            VARIETIES_OK = {"young", "mature", "ancient"}
            FLOAT_KEYS = {"size", "leaves", "detachment"}
            INT_KEYS   = {"branches", "seed"}
            params: dict = {}
            for k, v in body.items():
                if k == "variety" and v in VARIETIES_OK:
                    params[k] = v
                elif k in FLOAT_KEYS:
                    params[k] = float(v)
                elif k in INT_KEYS:
                    params[k] = int(v)
                elif k == "pos" and isinstance(v, list) and len(v) == 2:
                    params[k] = [float(v[0]), float(v[1])]
            merged = {**grove_params, **params}
            grove_request = merged
            with frame_cond:
                frame_cond.notify_all()
            self._reply(200, "application/json", json.dumps({"queued": merged}).encode())

        elif self.path == "/chat":
            text = (body.get("text") or "").strip()[:500]
            if text:
                msg = {"t": time.time(), "text": text}
                chat_log.append(msg)
                if len(chat_log) > CHAT_MAX:
                    del chat_log[: len(chat_log) - CHAT_MAX]
                print(f"[chat] {text}")
            self._reply(200, "application/json", b"{}")

        else:
            self.send_error(404)

    def _reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    port = 8080
    print(f"MuJoCo scene viewer → http://localhost:{port}")
    print(f"  model : {MODEL_PATH}")
    print(f"  render: {W}×{H} @ {FPS} fps")
    print(f"  drag to orbit · scroll to zoom · Space to pause")
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer(("", port), Handler).serve_forever()
