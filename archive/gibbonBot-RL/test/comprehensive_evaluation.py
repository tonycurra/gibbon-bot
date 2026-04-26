#!/usr/bin/env python3
"""
Comprehensive Training Script Evaluation

Runs all training scripts with standardized tests and collects performance metrics.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Configuration
TEST_STEPS = 50000  # Short test run
SCRIPTS_TO_TEST = [
    {
        "name": "train_simple",
        "command": [
            "python",
            "scripts/train_simple.py",
            "--steps",
            str(TEST_STEPS),
            "--no-inherit",
        ],
        "description": "Simplified training with direct rewards",
        "timeout": 120,  # 2 minutes for simple training
    },
    {
        "name": "train_fixed",
        "command": [
            "python",
            "scripts/train_fixed.py",
            "--task",
            "stand",
            "--envs",
            "1",  # Use minimal envs for short test
        ],
        "description": "Fixed training with inheritance options",
        "timeout": 30,  # Short timeout since it will run full training
    },
    {
        "name": "train_unified",
        "command": [
            "python",
            "scripts/train.py",
            "stand",
            "--cpu",
            "--no-watch",  # No steps parameter available
        ],
        "description": "Unified training script (CPU mode)",
        "timeout": 30,  # Short timeout since it will run full training
    },
]


class ScriptEvaluator:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def run_script(self, script_info):
        """Run a single training script and collect metrics"""
        print(f"\n🔍 Evaluating: {script_info['name']}")
        print(f"   Description: {script_info['description']}")
        print(f"   Command: {' '.join(script_info['command'])}")

        start_time = time.time()

        try:
            # Run the script with timeout
            timeout = script_info.get(
                "timeout", 300
            )  # Use script-specific timeout or default 5 minutes
            result = subprocess.run(
                script_info["command"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            end_time = time.time()
            training_time = end_time - start_time

            # Parse results
            metrics = self._parse_output(result.stdout, result.stderr)
            metrics["training_time"] = training_time
            metrics["exit_code"] = result.returncode

            # Store results
            self.results[script_info["name"]] = {
                "script_info": script_info,
                "metrics": metrics,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            print(f"✅ Completed in {training_time:.1f}s")
            self._print_metrics(metrics)

        except subprocess.TimeoutExpired:
            print(f"⏹️  Timeout after 300 seconds")
            self.results[script_info["name"]] = {
                "script_info": script_info,
                "metrics": {"timeout": True},
                "error": "Timeout",
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results[script_info["name"]] = {
                "script_info": script_info,
                "metrics": {"error": str(e)},
                "error": str(e),
            }

    def _parse_output(self, stdout, stderr):
        """Parse training output to extract metrics"""
        metrics = {
            "steps_second": None,
            "final_performance": None,
            "learning_stability": "unknown",
            "resource_usage": "unknown",
            "errors": [],
        }

        # Combine output for parsing
        output = stdout + stderr

        # Look for performance indicators
        if "Training completed" in output or "training completed" in output:
            metrics["completed_successfully"] = True

        # Look for step information
        step_match = re.search(r"(\d+) steps", output, re.IGNORECASE)
        if step_match:
            steps = int(step_match.group(1))
            if metrics["training_time"]:
                metrics["steps_second"] = steps / metrics["training_time"]

        # Look for performance metrics (height, reward, etc.)
        height_match = re.search(r"height[\s:]+([0-9.]+)", output, re.IGNORECASE)
        if height_match:
            metrics["final_performance"] = float(height_match.group(1))

        # Look for reward information
        reward_match = re.search(r"reward[\s:]+([0-9.-]+)", output, re.IGNORECASE)
        if reward_match:
            metrics["final_reward"] = float(reward_match.group(1))

        # Check for errors
        error_lines = [
            line
            for line in output.split("\n")
            if "error" in line.lower() or "warning" in line.lower()
        ]
        if error_lines:
            metrics["errors"] = error_lines[:5]  # Top 5 errors

        return metrics

    def _print_metrics(self, metrics):
        """Print formatted metrics"""
        print("   Metrics:")
        for key, value in metrics.items():
            if key != "errors":
                print(f"     - {key}: {value}")
        if metrics.get("errors"):
            print("     - errors:")
            for error in metrics["errors"][:3]:  # Show first 3 errors
                print(f"       • {error[:80]}...")

    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        print(f"\n📊 Evaluation Report")
        print(f"===================")
        print(f"Generated: {time.ctime()}")
        print(f"Test duration: {TEST_STEPS:,} steps per script")
        print()

        # Print results for each script
        for script_name, result in self.results.items():
            print(f"### {script_name}")
            print(f"**Description**: {result['script_info']['description']}")
            print(f"**Command**: {' '.join(result['script_info']['command'])}")

            if "error" in result:
                print(f"**Status**: ❌ FAILED - {result['error']}")
            elif result["metrics"].get("timeout"):
                print(f"**Status**: ⏹️  TIMEOUT")
            else:
                print(f"**Status**: ✅ COMPLETED")
                print(f"**Training Time**: {result['metrics']['training_time']:.1f}s")

                if result["metrics"]["steps_second"]:
                    print(f"**Steps/Second**: {result['metrics']['steps_second']:.1f}")

                if result["metrics"]["final_performance"]:
                    print(
                        f"**Final Performance**: {result['metrics']['final_performance']:.3f}m"
                    )

                if result["metrics"].get("final_reward"):
                    print(f"**Final Reward**: {result['metrics']['final_reward']:.3f}")

                if result["metrics"]["errors"]:
                    print(f"**Errors Detected**: {len(result['metrics']['errors'])}")

            print()

        # Generate comparative analysis
        self._generate_comparative_analysis()

    def _generate_comparative_analysis(self):
        """Generate comparative analysis"""
        print("## Comparative Analysis")
        print()

        # Find successful runs
        successful = [
            r
            for r in self.results.values()
            if "error" not in r and not r["metrics"].get("timeout")
        ]

        if not successful:
            print("No successful runs to compare.")
            return

        # Compare performance
        print("### Performance Comparison")

        # Sort by training time (fastest first)
        by_speed = sorted(successful, key=lambda x: x["metrics"]["training_time"])
        print(
            f"**Fastest**: {by_speed[0]['script_info']['name']} ({by_speed[0]['metrics']['training_time']:.1f}s)"
        )
        print(
            f"**Slowest**: {by_speed[-1]['script_info']['name']} ({by_speed[-1]['metrics']['training_time']:.1f}s)"
        )

        # Compare steps/second
        with_steps = [r for r in successful if r["metrics"]["steps_second"]]
        if with_steps:
            by_efficiency = sorted(
                with_steps, key=lambda x: x["metrics"]["steps_second"], reverse=True
            )
            print(
                f"**Most Efficient**: {by_efficiency[0]['script_info']['name']} ({by_efficiency[0]['metrics']['steps_second']:.1f} steps/s)"
            )

        # Compare final performance
        with_performance = [r for r in successful if r["metrics"]["final_performance"]]
        if with_performance:
            by_performance = sorted(
                with_performance,
                key=lambda x: x["metrics"]["final_performance"],
                reverse=True,
            )
            print(
                f"**Best Performance**: {by_performance[0]['script_info']['name']} ({by_performance[0]['metrics']['final_performance']:.3f}m)"
            )

        print()
        print("### Recommendations")
        print("- For fastest training: Use", by_speed[0]["script_info"]["name"])
        if with_steps:
            print(
                "- For best sample efficiency: Use",
                by_efficiency[0]["script_info"]["name"],
            )
        if with_performance:
            print(
                "- For best final performance: Use",
                by_performance[0]["script_info"]["name"],
            )

    def save_results(self):
        """Save evaluation results to file"""
        results_dir = Path("test/results")
        results_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"evaluation_{timestamp}.json"

        # Save full results
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save summary report
        report_filename = results_dir / f"evaluation_{timestamp}_report.md"
        with open(report_filename, "w") as f:
            # Redirect print output to file
            original_stdout = sys.stdout
            sys.stdout = f
            self.generate_report()
            sys.stdout = original_stdout

        print(f"\n💾 Results saved to:")
        print(f"   - Full data: {filename}")
        print(f"   - Report: {report_filename}")


def main():
    print("🧪 Comprehensive Training Script Evaluation")
    print("=" * 50)

    evaluator = ScriptEvaluator()

    # Run each script
    for script_info in SCRIPTS_TO_TEST:
        evaluator.run_script(script_info)

    # Generate and save report
    evaluator.generate_report()
    evaluator.save_results()

    print(f"\n✅ Evaluation completed in {time.time() - evaluator.start_time:.1f}s")


if __name__ == "__main__":
    main()
