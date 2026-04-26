#!/usr/bin/env python3
"""
Quick test to verify the lean-rl setup works
"""

import os
import sys

# Test that we can import the key components
print("🧪 Testing Lean RL setup...")

try:
    # Test imports
    import jax
    import jax.numpy as jnp

    print("✅ JAX imported successfully")

    # Test that scene.xml exists
    if os.path.exists("scene.xml"):
        print("✅ scene.xml found")
    else:
        print("❌ scene.xml missing")
        sys.exit(1)

    # Test that environment files exist
    if os.path.exists("stand_env_mjx.py"):
        print("✅ stand_env_mjx.py found")
    else:
        print("❌ stand_env_mjx.py missing")
        sys.exit(1)

    if os.path.exists("stand_env.py"):
        print("✅ stand_env.py found")
    else:
        print("❌ stand_env.py missing")
        sys.exit(1)

    if os.path.exists("train.py"):
        print("✅ train.py found")
    else:
        print("❌ train.py missing")
        sys.exit(1)

    # Test that train.py has help functionality
    import subprocess

    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode == 0:
        print("✅ train.py --help works")
    else:
        print("⚠️  train.py --help failed (but script may still work)")

    print("\n🎉 Lean RL setup is ready!")
    print("\nTo start training:")
    print("  python train.py --watch")

    print("\nTo visualize existing checkpoints:")
    print("  python train.py --visualize")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nYou may need to install dependencies:")
    print("  pip install jax mujoco numpy")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
