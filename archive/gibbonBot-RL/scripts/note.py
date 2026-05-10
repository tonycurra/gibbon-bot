#!/usr/bin/env python3
"""Attach a free-form note to an MLflow run.

Usage:
    scripts/note.py "this run hit rew=15 at 8M steps, heading reward sweet spot"
    scripts/note.py --run-id <RUN_ID> "old run worth re-checking"

By default targets the most recent run in the ape_walk_mjx experiment.
Notes accumulate as the run's `mlflow.note.content` (the description shown
in the MLflow UI). Existing note content is preserved; new text is appended
with a timestamp.
"""
import argparse
import datetime
import sys

import mlflow

EXPERIMENT = "ape_walk_mjx"
TRACKING_URI = "sqlite:///mlflow.db"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("note", help="free-form note to append")
    p.add_argument("--run-id", default=None, help="target a specific run (default: latest)")
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if args.run_id:
        run_id = args.run_id
    else:
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            print(f"experiment '{EXPERIMENT}' not found", file=sys.stderr)
            sys.exit(1)
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs:
            print("no runs found", file=sys.stderr)
            sys.exit(1)
        run_id = runs[0].info.run_id

    run = client.get_run(run_id)
    existing = run.data.tags.get("mlflow.note.content", "")
    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    new_entry = f"[{stamp}] {args.note}"
    combined = (existing + "\n\n" + new_entry).strip() if existing else new_entry
    client.set_tag(run_id, "mlflow.note.content", combined)
    print(f"[note] attached to run {run_id[:8]}: {args.note}")


if __name__ == "__main__":
    main()
