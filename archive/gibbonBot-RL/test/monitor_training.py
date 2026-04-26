#!/usr/bin/env python3
"""
Training Monitor - Tracks performance metrics during training runs
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil


class TrainingMonitor:
    def __init__(self, script_name, command):
        self.script_name = script_name
        self.command = command
        self.start_time = time.time()
        self.process = None
        self.metrics = {
            "start_time": self.start_time,
            "end_time": None,
            "training_time": None,
            "steps_second": None,
            "initial_height": None,
            "final_height": None,
            "improvement": None,
            "max_gpu_usage": 0,
            "max_cpu_usage": 0,
            "max_memory_usage": 0,
            "learning_curve": [],
            "exit_code": None,
        }

    def start(self):
        """Start the training process"""
        print(f"🚀 Starting {self.script_name}: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self.process

    def monitor(self):
        """Monitor the training process and collect metrics"""
        if not self.process:
            return False

        # Monitor resource usage
        try:
            proc = psutil.Process(self.process.pid)

            # CPU usage
            cpu_percent = proc.cpu_percent(interval=0.1)
            self.metrics["max_cpu_usage"] = max(
                self.metrics["max_cpu_usage"], cpu_percent
            )

            # Memory usage
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.metrics["max_memory_usage"] = max(
                self.metrics["max_memory_usage"], memory_mb
            )

            # GPU usage (if available)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.metrics["max_gpu_usage"] = max(
                    self.metrics["max_gpu_usage"], gpu_util.gpu
                )
                pynvml.nvmlShutdown()
            except:
                pass

        except psutil.NoSuchProcess:
            pass

        # Check if process is still running
        return self.process.poll() is None

    def parse_output(self, line):
        """Parse training output for performance metrics"""
        # Look for height information
        height_match = re.search(r"height[\s:]+([0-9.]+)", line, re.IGNORECASE)
        if height_match:
            height = float(height_match.group(1))

            # Record initial height
            if self.metrics["initial_height"] is None:
                self.metrics["initial_height"] = height

            # Update final height
            self.metrics["final_height"] = height

            # Record learning curve point
            current_time = time.time()
            self.metrics["learning_curve"].append(
                {"time": current_time - self.start_time, "height": height}
            )

        # Look for step information
        step_match = re.search(r"step[\s:]+([0-9]+)", line, re.IGNORECASE)
        if step_match and len(self.metrics["learning_curve"]) > 0:
            steps = int(step_match.group(1))
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                steps_per_second = steps / elapsed
                self.metrics["steps_second"] = steps_per_second

    def wait_for_completion(self):
        """Wait for training to complete and collect final metrics"""
        if not self.process:
            return

        # Read output in real-time
        while self.monitor():
            # Read stdout
            if self.process.stdout:
                line = self.process.stdout.readline()
                if line:
                    print(line.strip())
                    self.parse_output(line)

            # Read stderr
            if self.process.stderr:
                line = self.process.stderr.readline()
                if line:
                    print(line.strip(), file=sys.stderr)
                    self.parse_output(line)

            time.sleep(0.1)

        # Process completed, get exit code
        self.metrics["exit_code"] = self.process.wait()
        self.metrics["end_time"] = time.time()
        self.metrics["training_time"] = self.metrics["end_time"] - self.start_time

        # Calculate improvement
        if (
            self.metrics["initial_height"] is not None
            and self.metrics["final_height"] is not None
        ):
            self.metrics["improvement"] = (
                self.metrics["final_height"] - self.metrics["initial_height"]
            )

    def save_results(self, output_dir="test/results"):
        """Save monitoring results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        results_file = Path(output_dir) / f"{self.script_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save learning curve
        curve_file = Path(output_dir) / f"{self.script_name}_learning_curve.json"
        with open(curve_file, "w") as f:
            json.dump(self.metrics["learning_curve"], f, indent=2)

        print(f"✅ Results saved to {results_file}")

    def get_summary(self):
        """Get a summary of the monitoring results"""
        steps_per_second = (
            self.metrics["steps_second"]
            if self.metrics["steps_second"] is not None
            else "N/A"
        )
        initial_height = (
            self.metrics["initial_height"]
            if self.metrics["initial_height"] is not None
            else "N/A"
        )
        final_height = (
            self.metrics["final_height"]
            if self.metrics["final_height"] is not None
            else "N/A"
        )
        improvement = (
            self.metrics["improvement"]
            if self.metrics["improvement"] is not None
            else "N/A"
        )

        if isinstance(steps_per_second, (int, float)):
            steps_per_second = f"{steps_per_second:.1f}"
        if isinstance(initial_height, (int, float)):
            initial_height = f"{initial_height:.3f}"
        if isinstance(final_height, (int, float)):
            final_height = f"{final_height:.3f}"
        if isinstance(improvement, (int, float)):
            improvement = f"{improvement:.3f}"

        summary = f"""
### {self.script_name}
- **Command Used**: {" ".join(self.command)}
- **Training Time**: {self.metrics["training_time"]:.1f}s
- **Steps/Second**: {steps_per_second}
- **Initial Height**: {initial_height}m
- **Final Height**: {final_height}m
- **Improvement**: {improvement}m
- **GPU Usage**: {self.metrics["max_gpu_usage"]}%
- **CPU Usage**: {self.metrics["max_cpu_usage"]:.1f}%
- **Memory Usage**: {self.metrics["max_memory_usage"]:.1f}MB
- **Exit Code**: {self.metrics["exit_code"]}
- **Learning Curve Points**: {len(self.metrics["learning_curve"])}
"""
        return summary


def monitor_training_script(script_name, command, test_steps=100000):
    """Monitor a single training script"""
    # Add test steps parameter if the script supports it
    if "--steps" in " ".join(command):
        # Already has steps parameter
        pass
    elif any("--steps" in arg or arg == "--steps" for arg in command):
        # Already has steps parameter
        pass
    else:
        # Add steps parameter
        command = command + [f"--steps", str(test_steps)]

    # Create monitor
    monitor = TrainingMonitor(script_name, command)

    # Start and monitor training
    monitor.start()
    monitor.wait_for_completion()

    # Save results
    monitor.save_results()

    # Return summary
    return monitor.get_summary()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python monitor_training.py <script_name> <command...>")
        sys.exit(1)

    script_name = sys.argv[1]
    command = sys.argv[2:]

    summary = monitor_training_script(script_name, command)
    print(summary)
