#!/usr/bin/env python3
"""
UNIFIED TRAINING SYSTEM

Merges all past training data and saves to single location.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def merge_training_data(target_dir="runs/unified"):
    """Merge all past training data into unified location"""
    target = Path(target_dir)
    target.mkdir(exist_ok=True)

    # Find all past training runs
    past_runs = []
    for run_dir in Path("runs").glob("*"):
        if run_dir.is_dir() and "stand" in str(run_dir):
            past_runs.append(run_dir)

    print(f"Found {len(past_runs)} past training runs:")
    for i, run in enumerate(past_runs, 1):
        print(f"  {i}. {run.name}")

    # Copy best models to unified location
    best_models = []
    for run in past_runs:
        best_model = run / "best" / "best_model.zip"
        if best_model.exists():
            new_name = target / f"best_model_{run.name}.zip"
            shutil.copy2(best_model, new_name)
            best_models.append(new_name)

    print(f"\n✅ Merged {len(best_models)} best models to: {target}")
    return best_models


def main():
    parser = argparse.ArgumentParser(description="Unified Training System")
    parser.add_argument("--task", default="stand", help="Task name")
    parser.add_argument("--steps", type=int, default=1000000, help="Training steps")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge past data")
    args = parser.parse_args()

    # Set up unified directory
    unified_dir = Path(f"runs/unified_{args.task}")
    unified_dir.mkdir(exist_ok=True)

    print(f"🔄 Unified Training System")
    print(f"   Task: {args.task}")
    print(f"   Target: {unified_dir}")
    print()

    if not args.no_merge:
        # Merge all past training data
        best_models = merge_training_data(unified_dir)

        if best_models:
            print(f"\n📊 Past Training Data Available:")
            for model in best_models:
                print(f"   - {model.name}")

            # Set up inheritance
            latest_model = max(best_models, key=lambda x: x.stat().st_mtime)
            print(f"\n🎯 Will inherit from: {latest_model.name}")
        else:
            print("\n⚠️  No past training data found")
    else:
        print("📝 Starting fresh (no past data merging)")

    # Run training with unified approach
    import subprocess

    cmd = [sys.executable, "scripts/train_simple.py", f"--steps", str(args.steps)]

    if not args.no_merge:
        cmd.append("--inherit")

    print(f"🚀 Starting unified training: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    finally:
        print(f"\n✅ Training completed!")
        print(f"   All data saved to: {unified_dir}")
        print(f"   Use '--no-merge' to start fresh next time")


if __name__ == "__main__":
    main()
