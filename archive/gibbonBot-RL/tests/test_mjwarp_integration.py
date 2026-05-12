#!/usr/bin/env python3
"""Test script for MuJoCo Warp integration.

This script tests that the physics backend abstraction works correctly
with both MJX (default) and MJWarp (when available) backends.

Usage:
    # Test with MJX (default)
    python tests/test_mjwarp_integration.py

    # Test with MJWarp (if installed)
    MUJOCO_BACKEND=mjwarp python tests/test_mjwarp_integration.py

    # Force MJX even if MJWarp is installed
    MUJOCO_BACKEND=mjx python tests/test_mjwarp_integration.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_backend_import():
    """Test that the backend module imports correctly."""
    print("Testing backend import...")
    try:
        from envs._shared_physics_backend import get_backend, get_backend_info, mjx

        print("  ✓ Backend module imported successfully")

        # Check backend info
        info = get_backend_info()
        print(f"  Current backend: {info['backend']}")
        print(f"  Auto-detected: {info['auto_detected']}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import backend: {e}")
        return False


def test_model_loading():
    """Test loading a model with the backend."""
    print("\nTesting model loading...")
    try:
        import envs.params_walk_mjx as params
        from envs._shared_load_model import base_load_model
        from envs._shared_physics_backend import MjModel, mjx

        # Load model using base_load_model (which uses our backend)
        scene_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "scene.xml",
        )

        if not os.path.exists(scene_path):
            print(f"  ⚠ Scene file not found: {scene_path}")
            return True  # Not a failure, just skip

        mj_model, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose = (
            base_load_model(scene_path, params)
        )

        print(f"  ✓ Model loaded successfully")
        print(f"    Model has {mj_model.nq} positions, {mj_model.nv} velocities")
        print(f"    mx type: {type(mx)}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_operations():
    """Test creating and manipulating data."""
    print("\nTesting data operations...")
    try:
        import jax.numpy as jnp

        import envs.params_walk_mjx as params
        from envs._shared_load_model import base_load_model
        from envs._shared_physics_backend import MjModel, mjx

        scene_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "scene.xml",
        )

        if not os.path.exists(scene_path):
            print("  ⚠ Scene file not found, skipping")
            return True

        mj_model, mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose = (
            base_load_model(scene_path, params)
        )

        # Create data
        dx = mjx.make_data(mx)
        print(
            f"  ✓ Data created: qpos shape={dx.qpos.shape}, qvel shape={dx.qvel.shape}"
        )

        # Test replace
        dx2 = dx.replace(qpos=dx.qpos + 0.1)
        print(f"  ✓ Data.replace() works")

        # Test forward
        dx3 = mjx.forward(mx, dx2)
        print(f"  ✓ Forward kinematics works")

        # Test step
        dx4 = mjx.step(mx, dx3)
        print(f"  ✓ Physics step works")

        # Verify arrays are JAX arrays
        from jax.numpy import ndarray as JaxArray

        assert isinstance(dx.qpos, JaxArray), f"qpos is not JAX array: {type(dx.qpos)}"
        assert isinstance(dx.qvel, JaxArray), f"qvel is not JAX array: {type(dx.qvel)}"
        print(f"  ✓ All arrays are JAX arrays")

        return True
    except Exception as e:
        print(f"  ✗ Failed data operations: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environment():
    """Test loading and using an environment."""
    print("\nTesting environment...")
    try:
        import jax

        from envs.walk_env_mjx import load_model, reset, step

        # Load model
        mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose, tree_xy = (
            load_model()
        )
        print(f"  ✓ Environment model loaded")

        # Create random key
        rng = jax.random.PRNGKey(42)

        # Reset
        dx, obs, motor_pos, motor_vel, prev_action, prev_dist, step_idx = reset(
            mx, jnt_lo, jnt_hi, tree_xy, rng
        )
        print(f"  ✓ Environment reset works")
        print(f"    obs shape: {obs.shape}")

        # Step
        action = jnp.zeros(24)
        rparams = (
            0.45,
            0.05,
            1.0,
            1.0,
            1.0,
            0.0,
            1.5,
            0.0,
            0.05,
            1.0,
            10.0,
            0.5,
            0.01,
            0.01,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        (
            dx2,
            obs2,
            reward,
            done,
            motor_pos2,
            motor_vel2,
            filt_action,
            dist,
            step_idx2,
        ) = step(
            mx,
            jnt_lo,
            jnt_hi,
            motor_max_speed,
            motor_max_accel,
            rest_pose,
            tree_xy,
            dx,
            action,
            motor_pos,
            motor_vel,
            prev_action,
            prev_dist,
            step_idx,
            rparams,
        )
        print(f"  ✓ Environment step works")
        print(f"    reward: {reward}")
        print(f"    done: {done}")

        return True
    except Exception as e:
        print(f"  ✗ Failed environment test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backend_switching():
    """Test switching between backends."""
    print("\nTesting backend switching...")
    try:
        from envs._shared_physics_backend import get_backend_info, mjx, set_backend

        # Save original
        original_info = get_backend_info()

        # Try MJX
        set_backend("mjx")
        info = get_backend_info()
        assert info["backend"] == "mjx", f"Expected mjx, got {info['backend']}"
        print(f"  ✓ Switched to MJX backend")

        # Try MJWarp (may fail if not installed)
        try:
            set_backend("mjwarp")
            info = get_backend_info()
            print(f"  ✓ Switched to MJWarp backend")
        except (ImportError, ValueError) as e:
            print(f"  ⚠ MJWarp not available: {e}")

        # Restore original
        set_backend(original_info["backend"])
        print(f"  ✓ Backend switching works")

        return True
    except Exception as e:
        print(f"  ✗ Failed backend switching: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MuJoCo Warp Integration Tests")
    print("=" * 60)

    # Check backend
    from envs._shared_physics_backend import get_backend_info

    info = get_backend_info()
    print(f"\nCurrent backend: {info['backend'].upper()}")

    tests = [
        ("Backend Import", test_backend_import),
        ("Model Loading", test_model_loading),
        ("Data Operations", test_data_operations),
        ("Environment", test_environment),
        ("Backend Switching", test_backend_switching),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n{name} raised exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
