"""
Terrain generation for the ape simulation.

Generates a heightfield that looks like agricultural land:
  - rolling hills (low frequency)
  - furrow ridges in one direction (medium frequency)
  - fine surface bumps (noise)

Used by view_robot.py at runtime to fill model.hfield_data.
"""

import numpy as np
import mujoco

NROW = 64
NCOL = 64


def generate(nrow: int = NROW, ncol: int = NCOL, seed: int = 42) -> np.ndarray:
    """Return a (nrow, ncol) float32 heightmap, values in [0, 1]."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, ncol)
    y = np.linspace(0, 4 * np.pi, nrow)
    X, Y = np.meshgrid(x, y)

    hills   = (np.sin(X * 0.7 + 0.5) * np.cos(Y * 0.5)
             + np.sin(X * 0.4)       * np.cos(Y * 0.8 + 1.2))
    furrows = 0.5 * np.sin(X * 2.5)                  # ridges along Y axis
    noise   = rng.standard_normal((nrow, ncol)) * 0.1

    h = hills + furrows + noise
    h = (h - h.min()) / (h.max() - h.min())          # normalize → [0, 1]
    return h.astype(np.float32)


def apply(model: mujoco.MjModel, nrow: int = NROW, ncol: int = NCOL) -> None:
    """Write terrain heights into the hfield named 'terrain' in the model."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    if hid < 0:
        print("Warning: no hfield named 'terrain' in model — run build_scene.py")
        return
    model.hfield_data[:] = generate(nrow, ncol).flatten()
