#!/usr/bin/env python3
"""
UNIFIED TRAINING SYSTEM for gibbonBot-RL

Single command for all training needs:
  python train_unified.py stand          # Auto-detect GPU/CPU, launch viewer
  python train_unified.py reach          # Train reach task
  python train_unified.py --help         # Show all options

Features:
  ✅ Auto GPU/CPU detection
  ✅ Automatic viewer launch
  ✅ Unified interface
  ✅ Clean output
  ✅ Easy to extend
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class TrainingSession:
    def __init__(self):
        self.process = None
        self.viewer_process = None
        self.args = None
        self.start_time = None

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Unified gibbonBot-RL Training System",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Task selection
        parser.add_argument(
            "task",
            nargs="?",
            default="stand",
            choices=["stand", "reach"],
            help="Training task (stand or reach)",
        )

        # Mode selection
        parser.add_argument(
            "--cpu", action="store_true", help="Force CPU training (SB3)"
        )
        parser.add_argument(
            "--gpu", action="store_true", help="Force GPU training (MJX)"
        )

        # Training options
        parser.add_argument(
            "--envs", type=int, default=None, help="Number of parallel environments"
        )
        parser.add_argument(
            "--no-watch", action="store_true", help="Disable automatic viewer"
        )
        parser.add_argument(
            "--inherit", action="store_true", help="Inherit from best checkpoint"
        )
        parser.add_argument(
            "--resume", action="store_true", help="Resume previous training"
        )

        # Info
        parser.add_argument(
            "--info", action="store_true", help="Show system info and exit"
        )

        self.args = parser.parse_args()
        return self.args

    def detect_hardware(self):
        """Detect GPU availability"""
        try:
            import jax

            devices = jax.devices()
            has_gpu = any("cuda" in str(d).lower() for d in devices)
            return has_gpu, len(devices)
        except:
            return False, 0

    def show_info(self):
        """Display system information"""
        has_gpu, device_count = self.detect_hardware()

        print("🔧 gibbonBot-RL Training System")
        print(f"   GPU Detected: {'✅ Yes' if has_gpu else '❌ No'}")
        print(f"   Devices: {device_count}")
        print(f"   Current Task: {self.args.task}")
        print(f"   Mode: {'GPU' if (has_gpu and not self.args.cpu) else 'CPU'}")
        print(f"   Viewer: {'Enabled' if not self.args.no_watch else 'Disabled'}")
        if self.args.envs:
            print(f"   Environments: {self.args.envs}")
        if self.args.inherit:
            print(f"   Inherit: ✅")
        if self.args.resume:
            print(f"   Resume: ✅")

    def build_command(self):
        """Construct the training command"""
        has_gpu, _ = self.detect_hardware()
        use_gpu = has_gpu and not self.args.cpu or self.args.gpu

        if use_gpu:
            # GPU training
            cmd = [sys.executable, "config/train_mjx.py"]
            if self.args.envs:
                cmd.extend(["--envs", str(self.args.envs)])
            if self.args.inherit:
                cmd.append("--inherit")
            if self.args.resume:
                cmd.append("--resume")
        else:
            # CPU training
            cmd = [sys.executable, "train_fixed.py", "--task", self.args.task]
            if self.args.envs:
                cmd.extend(["--envs", str(self.args.envs)])
            if self.args.inherit:
                cmd.append("--inherit")
            if self.args.resume:
                cmd.append("--resume")
            if not self.args.no_watch:
                cmd.append("--watch")

        return cmd

    def launch_viewer(self):
        """Launch viewer in background"""
        if self.args.no_watch:
            return

        viewer_cmd = [sys.executable, "tools/watch.py", "--task", self.args.task]
        print(f"\n🎬 Launching viewer: {' '.join(viewer_cmd)}")

        # Use setsid to make it independent
        self.viewer_process = subprocess.Popen(viewer_cmd, preexec_fn=os.setsid)

    def start_training(self):
        """Start the training process"""
        cmd = self.build_command()
        print(f"🚀 Starting training: {' '.join(cmd)}")
        print(f"   Task: {self.args.task}")
        print(f"   Mode: {'GPU' if 'train_mjx.py' in cmd[1] else 'CPU'}")

        self.process = subprocess.Popen(cmd)
        self.start_time = time.time()

        # Launch viewer after a short delay
        time.sleep(2)
        self.launch_viewer()

    def stop(self):
        """Cleanly stop training and viewer"""
        if self.process:
            print("\n⏹️  Stopping training...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self.viewer_process:
            print("🎬 Stopping viewer...")
            try:
                os.killpg(os.getpgid(self.viewer_process.pid), signal.SIGTERM)
            except:
                pass

    def run(self):
        """Main execution"""
        try:
            args = self.parse_args()

            if args.info:
                self.show_info()
                return

            self.start_training()
            self.process.wait()

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.stop()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.stop()
        finally:
            if self.process:
                print(
                    f"\n✅ Training completed in {time.time() - self.start_time:.1f}s"
                )


if __name__ == "__main__":
    session = TrainingSession()
    session.run()
