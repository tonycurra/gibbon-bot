#!/usr/bin/env python3
"""Find the best checkpoint from local mlflow logs.

`best`-ness is whichever step had the highest `mean_rew`. Picks the
checkpoint .npz whose step count is closest to that peak.

Usage:
    python -m utils.best                    # print path of best checkpoint
    python -m utils.best --task walk        # default
    python -m utils.best --top 5            # top-5 by mean_rew
"""
import argparse
import pathlib
import sys


def find_best(task: str = "walk", top: int = 1):
    runs_dir = pathlib.Path(f"runs/ape_{task}_mjx")
    ckpts = sorted(runs_dir.glob("mjx_*.npz"))
    if not ckpts:
        print(f"[best] no checkpoints in {runs_dir}", file=sys.stderr)
        return []

    try:
        import mlflow
    except ImportError:
        print("[best] mlflow not installed, falling back to latest checkpoint", file=sys.stderr)
        return [(ckpts[-1], None, None)]

    mlflow.set_tracking_uri("file:./mlruns")
    exp = mlflow.get_experiment_by_name(f"ape_{task}_mjx")
    if exp is None:
        print(f"[best] no mlflow experiment 'ape_{task}_mjx' yet — using latest checkpoint", file=sys.stderr)
        return [(ckpts[-1], None, None)]

    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=20)
    if runs.empty:
        return [(ckpts[-1], None, None)]

    client = mlflow.tracking.MlflowClient()
    # Collect (step, mean_rew, run_id) across recent runs
    points = []
    for run_id in runs["run_id"]:
        for h in client.get_metric_history(run_id, "mean_rew"):
            points.append((h.step, h.value, run_id))
    if not points:
        return [(ckpts[-1], None, None)]

    points.sort(key=lambda x: -x[1])
    seen_steps = set()
    out = []
    for step, mean_rew, run_id in points:
        # find the checkpoint nearest to this step
        nearest = min(ckpts, key=lambda c: abs(int(c.stem.split("_")[1]) - step))
        if nearest in seen_steps:
            continue
        seen_steps.add(nearest)
        out.append((nearest, mean_rew, step))
        if len(out) >= top:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="walk")
    ap.add_argument("--top",  type=int, default=1)
    args = ap.parse_args()

    results = find_best(args.task, args.top)
    if not results:
        sys.exit(1)
    for path, rew, step in results:
        if rew is None:
            print(path)
        else:
            print(f"{path}\tmean_rew={rew:.3f}\tat step {step:,}")
