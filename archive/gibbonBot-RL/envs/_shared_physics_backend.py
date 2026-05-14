"""Physics backend abstraction for MJX (JAX) and MJWarp (Warp) compatibility.

This module provides a unified interface that transparently uses either
MuJoCo MJX (JAX backend) or MuJoCo Warp (NVIDIA Warp backend).

The API mirrors mujoco.mjx exactly, so existing code needs only an import change.

Usage:
    # Instead of:
    # import mujoco.mjx as mjx

    # Use:
    from envs._shared_physics_backend import mjx

    # All existing code (mjx.put_model, mjx.step, mjx.forward, mjx.make_data) works unchanged

Environment variable:
    MUJOCO_BACKEND: "mjx" (default) or "mjwarp"
    Set to "mjwarp" to force MJWarp, or "mjx" to force MJX.
    Auto-detection used if not set.
"""

import os
import sys

# =============================================================================
# Backend detection
# =============================================================================


def _get_backend_name():
    """Determine which backend to use."""
    env_backend = os.environ.get("MUJOCO_BACKEND", "").lower()

    if env_backend == "mjx":
        return "mjx"
    elif env_backend == "mjwarp":
        return "mjwarp"

    # Auto-detect: try MJWarp first
    try:
        import mujoco_warp

        return "mjwarp"
    except ImportError:
        pass

    return "mjx"


# Lazy backend initialization - only created on first use
_backend = None


def _get_backend():
    """Get or create the backend implementation."""
    global _backend
    if _backend is None:
        backend_name = _get_backend_name()

        if backend_name == "mjwarp":
            try:
                _backend = _MJWarpBackend()
            except ImportError as e:
                print(f"[WARNING] MJWarp requested but not available: {e}")
                print("[WARNING] Falling back to MJX backend")
                _backend = _MJXBackend()
        else:
            _backend = _MJXBackend()
    return _backend


# =============================================================================
# Backend implementations
# =============================================================================


class _MJXBackend:
    """MJX (JAX) backend implementation."""

    def __init__(self):
        import mujoco.mjx as mjx

        self._mjx = mjx

    def put_model(self, mj_model):
        return self._mjx.put_model(mj_model)

    def make_data(self, mx):
        return self._mjx.make_data(mx)

    def step(self, mx, dx):
        return self._mjx.step(mx, dx)

    def forward(self, mx, dx):
        return self._mjx.forward(mx, dx)

    # Pass through to mujoco module for model loading
    def MjModel(self):
        import mujoco

        return mujoco.MjModel

    def mj_id2name(self, model, obj_type, id):
        import mujoco

        return mujoco.mj_id2name(model, obj_type, id)

    @property
    def backend_name(self):
        return "mjx"


class _MJWarpBackend:
    """MJWarp (Warp) backend implementation.

    MJWarp uses Warp arrays internally, but we need to return JAX arrays
    to maintain compatibility with the rest of the codebase.
    """

    def __init__(self):
        import jax.numpy as jnp
        import mujoco
        import mujoco_warp as mjwarp
        import warp as wp

        self._mujoco = mujoco
        self._mjwarp = mjwarp
        self._wp = wp
        self._jnp = jnp

        # Cache for model conversions
        self._model_cache = {}

    def put_model(self, mj_model):
        """Convert MjModel to MJWarp model.

        Stash the raw CPU MjModel — mujoco_warp.make_data() requires
        the original `mujoco.MjModel`, not the converted Warp Model
        (it does its own put_model-like work internally and reads
        host-side numpy fields like geom_type during sizing). Without
        this we hit `AttributeError: 'bool' object has no attribute 'any'`
        inside mujoco_warp's _default_nconmax.
        """
        self._mj_model = mj_model
        return self._mjwarp.put_model(mj_model)

    def make_data(self, mx):
        """Create data for a model, with JAX arrays."""
        d = self._mjwarp.make_data(self._mj_model)
        return self._to_jax_data(d, mx)

    def step(self, mx, dx):
        """Physics step with JAX array conversion."""
        d = self._to_mjwarp_data(dx, mx)
        d = self._mjwarp.step(mx, d)
        return self._to_jax_data(d, mx)

    def forward(self, mx, dx):
        """Forward kinematics with JAX array conversion."""
        d = self._to_mjwarp_data(dx, mx)
        d = self._mjwarp.forward(mx, d)
        return self._to_jax_data(d, mx)

    def _to_jax_data(self, d, mx):
        """Convert MJWarp data (with warp arrays) to JAX arrays."""
        import jax.numpy as jnp

        # Get all the standard MJX data fields
        fields = {
            "qpos": d.qpos,
            "qvel": d.qvel,
            "qacc": d.qacc,
            "qfrc_actuator": d.qfrc_actuator,
            "qfrc_passive": d.qfrc_passive if hasattr(d, "qfrc_passive") else None,
            "ctrl": d.ctrl,
            "act": d.act if hasattr(d, "act") else None,
            "time": d.time if hasattr(d, "time") else 0.0,
            "energy": d.energy if hasattr(d, "energy") else None,
        }

        # Convert warp arrays to numpy, then to JAX
        jax_fields = {}
        for key, value in fields.items():
            if value is None:
                continue
            # Check if it's a warp array
            if hasattr(value, "numpy"):
                # Warp array - convert to numpy then JAX
                jax_fields[key] = jnp.array(value.numpy())
            elif hasattr(value, "__array__"):
                # Already array-like
                jax_fields[key] = jnp.asarray(value)
            else:
                # Scalar or other
                jax_fields[key] = value

        # Create a MJX-style named tuple
        from mujoco.mjx import Data as MJXData

        return MJXData(**jax_fields)

    def _to_mjwarp_data(self, dx, mx):
        """Convert JAX data to MJWarp data (with warp arrays)."""
        d = self._mjwarp.make_data(self._mj_model)

        # Copy values from dx to d
        for field in ["qpos", "qvel", "qacc", "qfrc_actuator", "ctrl", "act"]:
            if hasattr(dx, field) and hasattr(d, field):
                arr = getattr(dx, field)
                # Convert JAX array to warp array
                if hasattr(arr, "shape"):  # Is an array
                    setattr(d, field, self._wp.array(arr))

        return d

    # Pass through to mujoco module for model loading
    def MjModel(self):
        return self._mujoco.MjModel

    def mj_id2name(self, model, obj_type, id):
        return self._mujoco.mj_id2name(model, obj_type, id)

    @property
    def backend_name(self):
        return "mjwarp"


# =============================================================================
# Public API - mirrors mujoco.mjx
# =============================================================================


class _MJXProxy:
    """Proxy class that lazily forwards calls to the backend.

    This allows the module to be imported even when mujoco is not installed,
    as long as the backend functions are not actually called.
    """

    def __getattr__(self, name):
        # Lazy backend initialization on first attribute access
        backend = _get_backend()
        return getattr(backend, name)


# Create proxy object
mjx = _MJXProxy()


def MjModel():
    """Lazy wrapper for MuJoCo MjModel."""
    backend = _get_backend()
    return backend.MjModel()


def mj_id2name(model, obj_type, id):
    """Lazy wrapper for mujoco.mj_id2name."""
    backend = _get_backend()
    return backend.mj_id2name(model, obj_type, id)


def get_backend():
    """Get the current physics backend instance."""
    return _get_backend()


def get_backend_info():
    """Get information about the current backend."""
    backend = _get_backend()
    return {
        "backend": backend.backend_name,
        "auto_detected": os.environ.get("MUJOCO_BACKEND", "") == "",
    }


def set_backend(backend_name):
    """Manually set the backend. Useful for testing.

    Note: This clears the cached backend, forcing re-initialization.
    """
    global _backend
    _backend = None
    os.environ["MUJOCO_BACKEND"] = backend_name.lower()


# =============================================================================
# Module-level info
# =============================================================================


def print_backend_info():
    """Print backend information."""
    info = get_backend_info()
    print(f"[Physics Backend] Using: {info['backend'].upper()}")
    if info["auto_detected"]:
        print(f"[Physics Backend] Auto-detected")
    else:
        print(f"[Physics Backend] Set via MUJOCO_BACKEND env var")


# Print info on import for debugging
if False:  # Set to True for debugging
    print_backend_info()
