#!/usr/bin/env python3
"""
Analyze all training data and generate summary
"""

import json
import os
from datetime import datetime
from pathlib import Path


def analyze_training_data():
    """Generate comprehensive training summary"""

    print("🔍 Analyzing Training Data...")
    print("=" * 50)

    # 1. Count training runs
    ppo_dirs = list(Path("runs/ape_stand").glob("PPO_*"))
    mjx_runs = list(Path("runs/ape_stand_mjx").glob("checkpoint_*"))

    print(f"\n📊 Training Runs Found:")
    print(f"   CPU (SB3): {len(ppo_dirs)} runs")
    print(f"   GPU (MJX): {len(mjx_runs)} runs")

    # 2. Check best models
    cpu_best = Path("runs/ape_stand/best/best_model.zip")
    gpu_best = Path("runs/ape_stand_mjx/best/best_model.zip")

    print(f"\n🏆 Best Models:")
    print(f"   CPU: {cpu_best.exists() * '✅' or '❌'}")
    print(f"   GPU: {gpu_best.exists() * '✅' or '❌'}")

    # 3. Analyze lineage
    try:
        with open("runs/ape_stand/lineage.json") as f:
            cpu_lineage = json.load(f)
        print(f"\n📜 CPU Training Lineage:")
        print(f"   Runs: {len(cpu_lineage.get('runs', []))}")
        if "best" in cpu_lineage:
            print(f"   Best reward: {cpu_lineage['best']['reward']:.1f}")
            print(f"   Best steps: {cpu_lineage['best']['steps']:,}")
    except:
        print("\n📜 CPU Training Lineage: None found")

    try:
        with open("runs/ape_stand_mjx/lineage.json") as f:
            gpu_lineage = json.load(f)
        print(f"\n📜 GPU Training Lineage:")
        print(f"   Runs: {len(gpu_lineage.get('runs', []))}")
        if "best" in gpu_lineage:
            print(f"   Best reward: {gpu_lineage['best']['reward']:.1f}")
            print(f"   Best steps: {gpu_lineage['best']['steps']:,}")
    except:
        print("\n📜 GPU Training Lineage: None found")

    # 4. Estimate total training steps
    total_steps = 0
    for ppo_dir in ppo_dirs:
        try:
            with open(ppo_dir / "lineage.json") as f:
                data = json.load(f)
                if "steps" in data:
                    total_steps += data["steps"]
        except:
            pass

    print(f"\n📈 Estimated Total Training:")
    print(f"   CPU Steps: {total_steps:,}")

    # 5. List all checkpoints
    print(f"\n🗃️ Available Checkpoints:")
    for ppo_dir in ppo_dirs[:5]:  # Show first 5
        print(f"   - {ppo_dir.name}/")

    if mjx_runs[:5]:
        print("   GPU Checkpoints:")
        for chk in mjx_runs[:5]:
            print(f"   - {chk.name}")

    print(f"\n✅ Analysis Complete!")
    print("=" * 50)


if __name__ == "__main__":
    analyze_training_data()
