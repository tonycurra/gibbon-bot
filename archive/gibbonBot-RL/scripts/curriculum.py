#!/usr/bin/env python3
"""Curriculum runner — train atomic skills sequentially, transferring policy
weights from each task to the next.

Order: stand → walk_forward → turn_left → turn_right.

Each task self-stops when its sustained success criterion is met (early stop
inside trainer) or hits a step budget. After each task, the best keeper is
located and passed to the next task as --resume-from.

Usage:
    python scripts/curriculum.py                  # full curriculum
    python scripts/curriculum.py --from walk_forward    # restart partway
    python scripts/curriculum.py --task-budget 5_000_000 --dry-run    # plumbing test
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (task, prior_task_for_transfer, default_step_budget,
#  early_stop_metric, op, value, window)
TASKS = [
    ("stand",        None,           50_000_000, "mean_upright",  "gt", 0.90, 20),
    ("walk_forward", "stand",        50_000_000, "mean_speed_x",  "gt", 0.20, 20),
    ("turn_left",    "walk_forward", 30_000_000, "mean_angvel_z", "gt", 0.40, 20),
    ("turn_right",   "turn_left",    30_000_000, "mean_angvel_z", "lt", -0.40, 20),
]


def best_keeper(task: str):
    """Return path to the best keeper .npz for a task, or None if none exist.
    Picks by sidecar JSON score (task-specific metric)."""
    runs_dir = ROOT / f"runs/ape_{task}_mjx" / "keepers"
    if not runs_dir.exists():
        return None
    sidecars = list(runs_dir.glob("auto_*.json"))
    if not sidecars:
        return None
    # Score function per task
    score_keys = {
        "stand":        ("mean_upright", "max"),
        "walk_forward": ("mean_speed_x", "max"),
        "turn_left":    ("mean_angvel_z", "max"),
        "turn_right":   ("mean_angvel_z", "min"),  # most negative
        "walk":         ("min_dist", "min"),
    }
    key, direction = score_keys.get(task, ("mean_rew", "max"))
    candidates = []
    for sc_path in sidecars:
        try:
            sc = json.loads(sc_path.read_text())
            score = sc.get("metrics", {}).get(key)
            if score is None: continue
            npz_path = sc_path.with_suffix(".npz")
            if npz_path.exists():
                candidates.append((score, npz_path))
        except Exception:
            continue
    if not candidates:
        return None
    if direction == "max":
        candidates.sort(key=lambda x: x[0], reverse=True)
    else:
        candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def run_task(task, prior_task, budget, early_metric, early_op, early_value, early_window,
             dry_run=False, auto_push=True, no_watch=True):
    print(f"\n{'='*70}")
    print(f"  CURRICULUM STEP: {task}")
    print(f"{'='*70}")

    cmd = [sys.executable, "train_mjx.py", "--task", task, "--no-watch" if no_watch else ""]
    cmd = [c for c in cmd if c]

    # Resume-from prior task's best keeper
    resume_path = None
    if prior_task:
        resume_path = best_keeper(prior_task)
        if resume_path:
            cmd += ["--resume-from", str(resume_path)]
            print(f"  resume-from: {resume_path}")
        else:
            print(f"  (no keeper found in {prior_task}; starting from scratch)")

    if auto_push:
        cmd.append("--auto-push")

    cmd += ["--label", f"curriculum:{task}{' from ' + prior_task if prior_task else ''}"]
    cmd += ["--early-stop-metric", early_metric,
            "--early-stop-op", early_op,
            "--early-stop-value", str(early_value),
            "--early-stop-window", str(early_window)]

    if dry_run:
        # Tiny budget for plumbing test
        cmd += ["--envs", "64"]

    print(f"  early-stop: {early_metric} {early_op} {early_value} for {early_window} consecutive blocks")
    print(f"  budget: {budget:,} steps")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] would launch above command — skipping")
        return True

    t_start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    wall = time.time() - t_start
    rc = proc.returncode
    print(f"\n  task '{task}' exit={rc}  wall={wall:.1f}s")

    keeper = best_keeper(task)
    if keeper:
        print(f"  best keeper: {keeper}")
    else:
        print(f"  no keeper saved")
    return rc == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start_from", default=None,
                   help="restart curriculum from this task name")
    p.add_argument("--task-budget", type=int, default=None,
                   help="override per-task step budget (default per-task in TASKS)")
    p.add_argument("--dry-run", action="store_true",
                   help="show commands but don't actually train")
    p.add_argument("--no-auto-push", action="store_true")
    p.add_argument("--watch", action="store_true",
                   help="have trainer auto-spawn viewer (default: no — viewer dies a lot)")
    args = p.parse_args()

    skip = args.start_from is not None
    for task, prior, default_budget, em, eo, ev, ew in TASKS:
        if skip:
            if task == args.start_from:
                skip = False
            else:
                print(f"[curriculum] skipping {task} (--from {args.start_from})")
                continue
        budget = args.task_budget or default_budget
        ok = run_task(task, prior, budget, em, eo, ev, ew,
                      dry_run=args.dry_run,
                      auto_push=not args.no_auto_push,
                      no_watch=not args.watch)
        if not ok:
            print(f"[curriculum] task '{task}' failed — continuing to next anyway")

    print("\n[curriculum] complete.")


if __name__ == "__main__":
    main()
