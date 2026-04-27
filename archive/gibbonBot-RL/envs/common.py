"""
Shared constants and utilities for stand_env.py and stand_env_mjx.py.
"""

import importlib
import pathlib
import numpy as np

import envs.params as params_mod

ROOT  = pathlib.Path(__file__).parent.parent
SCENE = ROOT / "data" / "scene.xml"
PARAMS_FILE = pathlib.Path(__file__).parent / "params.py"

MAX_STEPS = 2000
ACT_DIM   = 24
ARM_CONTACT_DIST = 0.02  # info-only collision flag threshold (metres)

# body IDs: FL=2..7, FR=8..13, BL=14..19, BR=20..25
ARM_GROUPS = [
    [2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25],
]

# Cross-arm pairs for proximity penalty: indices into ARM_GROUPS
PAIR_A = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
PAIR_B = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)

# arm group array for JAX (importers can convert with jnp.array)
ARM_GROUPS_NP = np.array(ARM_GROUPS, dtype=np.int32)


_params_mtime = 0.0

def load_params(current_mtime: float) -> tuple:
    """Reload params.py if it changed. Returns (module, new_mtime)."""
    global _params_mtime
    try:
        mtime = PARAMS_FILE.stat().st_mtime
        if mtime != current_mtime:
            importlib.reload(params_mod)
            _params_mtime = mtime
            return params_mod, mtime
    except Exception as e:
        print(f"[params] reload error: {e}")
    return params_mod, current_mtime


def read_reward_params():
    """Read current reward params as plain floats. Used by train_mjx.py."""
    global _params_mtime
    try:
        mtime = PARAMS_FILE.stat().st_mtime
        if mtime != _params_mtime:
            importlib.reload(params_mod)
            _params_mtime = mtime
    except Exception as e:
        print(f"[params] reload error: {e}")
    p = params_mod
    return (
        float(p.TARGET_Z),
        float(p.TARGET_TOL),
        float(p.W_HEIGHT),
        float(p.W_UPRIGHT),
        float(p.ARM_SAFETY_MARGIN),
        float(p.W_PROXIMITY),
        float(p.FLIP_PENALTY),
    )
