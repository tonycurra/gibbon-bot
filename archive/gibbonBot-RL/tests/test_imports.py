#!/usr/bin/env python3
"""Quick import test for MJWarp integration.

This test verifies that the import structure is correct.
It doesn't require mujoco to be installed.
"""

import os
import sys

print("Testing import structure...")

# Test 1: Can we import the backend module?
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from envs._shared_physics_backend import (
        MjModel,
        get_backend,
        get_backend_info,
        mj_id2name,
        mjx,
        set_backend,
    )

    print("✓ Backend module imports successfully")
except Exception as e:
    print(f"✗ Backend module import failed: {e}")
    sys.exit(1)

# Test 2: Can we import the shared modules?
try:
    from envs import _shared_load_model
    from envs import _shared_dynamics
    from envs import _shared_obs

    print("✓ Shared modules import successfully")
except Exception as e:
    print(f"✗ Shared modules import failed: {e}")
    sys.exit(1)

# Test 3: Can we import the environment modules?
try:
    from envs import walk_env_mjx
    from envs import stand_env_mjx
    from envs import turn_left_env_mjx
    from envs import turn_right_env_mjx
    from envs import walk_forward_env_mjx

    print("✓ Environment modules import successfully")
except Exception as e:
    print(f"✗ Environment modules import failed: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
print("\nNote: This test only verifies import structure.")
print("To test actual functionality, install mujoco and mujoco-mjx:")
print("  pip install mujoco==3.8.0 mujoco-mjx==3.8.0")
