#!/usr/bin/env python3
"""
Find the best past model for the current scene.

Usage:
    python -m utils.best_for_scene                    # evaluate all stand checkpoints
    python -m utils.best_for_scene --task stand      # specific task
    python -m utils.best_for_scene --all-tasks       # all tasks
    python -m utils.best_for_scene --view            # view best in GUI
    python -m utils.best_for_scene --steps 500       # longer evaluation
"""

import argparse
import json
import pathlib
import subprocess
import sys


def find_best_for_scene(task=None, all_tasks=False, scene="data/scene.xml", top=5, steps=200):
    """Run evaluate_checkpoints and return top results."""
    cmd = [
        sys.executable, "-m", "utils.evaluate_checkpoints",
        "--scene", scene,
        "--top", str(top),
        "--steps", str(steps),
        "--output", "/tmp/ckpt_results.json",
    ]
    if all_tasks:
        cmd.append("--all-tasks")
    elif task:
        cmd.extend(["--task", task])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Load results
    results_path = pathlib.Path("/tmp/ckpt_results.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="stand", help="Task to search")
    parser.add_argument("--all-tasks", action="store_true", help="Search all tasks")
    parser.add_argument("--scene", default="data/scene.xml", help="Scene XML")
    parser.add_argument("--top", type=int, default=5, help="Top N results")
    parser.add_argument("--steps", type=int, default=200, help="Evaluation steps")
    parser.add_argument("--view", action="store_true", help="View best in GUI")
    parser.add_argument("--checkpoint", "-c", help="View specific checkpoint")
    args = parser.parse_args()
    
    if args.checkpoint:
        # View a specific checkpoint
        import os
        os.system(f"python view.py --task {args.task} --checkpoint {args.checkpoint}")
        return
    
    results = find_best_for_scene(
        task=args.task if not args.all_tasks else None,
        all_tasks=args.all_tasks,
        scene=args.scene,
        top=args.top,
        steps=args.steps,
    )
    
    if not results:
        print("No results found")
        sys.exit(1)
    
    # If --view, open the best one
    if args.view and results:
        best = results[0]
        ckpt_path = best.get("path", "")
        if ckpt_path:
            print(f"\nViewing best checkpoint: {ckpt_path}")
            import os
            os.system(f"python view.py --task {args.task} --checkpoint {ckpt_path}")
    
    return results


if __name__ == "__main__":
    main()
