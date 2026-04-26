#!/usr/bin/env python3
"""
Automatic Training Evaluation Script

Monitors training progress and detects stagnation.
Run this alongside your training to get progress updates.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


class TrainingEvaluator:
    def __init__(self, task_name="stand"):
        self.task_name = task_name
        self.runs_dir = Path(f"runs/ape_{task_name}")
        self.progress_file = self.runs_dir / "progress.json"

        # Load previous progress if exists
        self.progress_history = []
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    self.progress_history = json.load(f)
            except:
                pass

        # Target values
        self.START_HEIGHT = 0.07  # Starting height
        self.TARGET_HEIGHT = 0.30  # Target standing height
        self.STAGNATION_THRESHOLD = 0.01  # 1cm improvement
        self.STAGNATION_WINDOW = 5000  # steps

    def evaluate_current_training(self, step_count, current_height):
        """Evaluate training progress and detect stagnation"""

        # Calculate progress percentage
        progress_pct = (
            (current_height - self.START_HEIGHT)
            / (self.TARGET_HEIGHT - self.START_HEIGHT)
        ) * 100
        progress_pct = max(0, min(100, progress_pct))

        # Store current progress
        self.progress_history.append(
            {
                "step": step_count,
                "height": current_height,
                "progress_pct": progress_pct,
                "timestamp": time.time(),
            }
        )

        # Save progress
        with open(self.progress_file, "w") as f:
            json.dump(self.progress_history, f, indent=2)

        # Detect stagnation
        stagnant = False
        if len(self.progress_history) > 1:
            recent_progress = self.progress_history[-5:]  # Last 5 evaluations
            if len(recent_progress) >= 2:
                steps_diff = recent_progress[-1]["step"] - recent_progress[0]["step"]
                height_diff = (
                    recent_progress[-1]["height"] - recent_progress[0]["height"]
                )

                if steps_diff >= self.STAGNATION_WINDOW:
                    improvement_rate = height_diff / steps_diff
                    if height_diff < self.STAGNATION_THRESHOLD:
                        stagnant = True
                        status = f"STAGNANT ({progress_pct:.1f}%) - {improvement_rate:.4f}m/step"
                    else:
                        status = f"PROGRESSING ({progress_pct:.1f}%) - {improvement_rate:.4f}m/step"
                else:
                    status = f"PROGRESSING ({progress_pct:.1f}%)"
            else:
                status = f"PROGRESSING ({progress_pct:.1f}%)"
        else:
            status = f"PROGRESSING ({progress_pct:.1f}%)"

        # Find max height achieved
        max_height = max([p["height"] for p in self.progress_history])

        return {
            "current_height": current_height,
            "progress_pct": progress_pct,
            "max_height": max_height,
            "status": status,
            "stagnant": stagnant,
        }

    def print_evaluation(self, result, step_count):
        """Print evaluation results"""
        print(f"\n{'=' * 60}")
        print(f"📊 TRAINING EVALUATION @ Step {step_count}")
        print(f"{'=' * 60}")
        print(f"📏 Current Height: {result['current_height']:.3f}m")
        print(f"🎯 Target Height: {self.TARGET_HEIGHT:.2f}m")
        print(f"📈 Progress: {result['progress_pct']:.1f}%")
        print(f"🏆 Max Height: {result['max_height']:.3f}m")
        print(f"🔄 Status: {result['status']}")

        if result["stagnant"]:
            print(f"⚠️  STAGNATION DETECTED!")
            print(f"   Consider:")
            print(f"   - Adjusting learning rate")
            print(f"   - Changing reward shaping")
            print(f"   - Modifying physics parameters")
            print(f"   - Adding curriculum learning")
        else:
            print(f"✅ Training is progressing well!")

        print(f"{'=' * 60}\n")

    def monitor_training(self, max_steps=100000, eval_interval=1000):
        """Monitor training progress periodically"""
        print(f"Starting training monitor for task: {self.task_name}")
        print(f"Will evaluate every {eval_interval} steps")

        # In a real implementation, this would connect to the training process
        # For now, we'll simulate monitoring by checking files

        last_step = 0
        while last_step < max_steps:
            # Check for progress files
            if self.progress_file.exists():
                with open(self.progress_file, "r") as f:
                    try:
                        progress = json.load(f)
                        if progress and progress[-1]["step"] > last_step:
                            last_step = progress[-1]["step"]
                            result = self.evaluate_current_training(
                                last_step, progress[-1]["height"]
                            )
                            self.print_evaluation(result, last_step)
                    except:
                        pass

            time.sleep(10)  # Check every 10 seconds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Progress Monitor")
    parser.add_argument("--task", default="stand", help="Task name (stand/reach)")
    parser.add_argument(
        "--interval", type=int, default=1000, help="Evaluation interval in steps"
    )
    args = parser.parse_args()

    evaluator = TrainingEvaluator(args.task)
    evaluator.monitor_training(eval_interval=args.interval)
