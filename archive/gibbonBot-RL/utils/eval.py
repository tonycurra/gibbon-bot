#!/usr/bin/env python3
"""Cross-run evaluation summary.

mlflow stores each `./remote.sh start` invocation as a separate run, but
the policy weights are continuous (auto-resume). This tool stitches them
back together into a single training trajectory.

Usage:
    python -m utils.eval                      # full summary
    python -m utils.eval --task walk
    python -m utils.eval --runs 5             # just show last 5 runs
    python -m utils.eval --plot               # also write reward.png
"""
import argparse
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_SCHEMA_DETAILS_LOGGING", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="walk")
    ap.add_argument("--runs", type=int, default=0, help="show last N runs (0 = all)")
    ap.add_argument("--plot", action="store_true", help="also save reward curve to reward.png")
    args = ap.parse_args()

    try:
        import mlflow
    except ImportError:
        sys.exit("mlflow not installed locally. pip install mlflow")

    mlflow.set_tracking_uri("file:./mlruns")
    exp = mlflow.get_experiment_by_name(f"ape_{args.task}_mjx")
    if exp is None:
        sys.exit(f"no mlflow experiment 'ape_{args.task}_mjx' yet")

    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time ASC"])
    if runs.empty:
        sys.exit("no runs in this experiment yet")

    client = mlflow.tracking.MlflowClient()

    # Stitch metric history across all runs in chronological order
    points = []  # list of (step, metric_name, value, run_name)
    metric_names = ["mean_rew", "mean_dist", "min_dist", "success_frac", "sps"]
    for _, run in runs.iterrows():
        rid = run["run_id"]
        rname = run.get("tags.mlflow.runName", rid[:8])
        for mname in metric_names:
            try:
                hist = client.get_metric_history(rid, mname)
            except Exception:
                continue
            for h in hist:
                points.append((h.step, mname, h.value, rname))

    if not points:
        sys.exit("no metric points across runs (training never logged any?)")

    # Per-metric: sorted by step, take latest 10 + best/worst
    print(f"\n=== {args.task.upper()} TASK SUMMARY ===")
    print(f"runs in experiment: {len(runs)}")
    print(f"total metric points: {len(points)}")

    by_metric = {m: sorted([p for p in points if p[1] == m]) for m in metric_names}

    for m in metric_names:
        pts = by_metric[m]
        if not pts:
            print(f"  {m:14s} : no data")
            continue
        steps = [p[0] for p in pts]
        vals  = [p[2] for p in pts]
        best  = max(vals) if m in ("mean_rew", "best_rew", "success_frac", "sps") else min(vals)
        latest = vals[-1]
        first  = vals[0]
        print(f"  {m:14s} : first={first:7.3f}  latest={latest:7.3f}  best={best:7.3f}  (over {min(steps):,}→{max(steps):,} steps)")

    print(f"\ntotal training steps reached: {max(p[0] for p in points):,}")

    if args.runs:
        recent = runs.tail(args.runs)
    else:
        recent = runs
    print(f"\n=== last {len(recent)} runs ===")
    cols = []
    for c in ["tags.mlflow.runName", "tags.resume", "metrics.mean_rew",
              "metrics.mean_dist", "metrics.min_dist", "metrics.success_frac",
              "metrics.update", "tags.stop_reason"]:
        if c in runs.columns:
            cols.append(c)
    print(recent[cols].to_string(index=False))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, mname in zip(axes.flat, ["mean_rew", "mean_dist", "min_dist", "success_frac"]):
                pts = by_metric[mname]
                if pts:
                    ax.plot([p[0] for p in pts], [p[2] for p in pts], ".-", markersize=2)
                ax.set_title(mname)
                ax.set_xlabel("steps")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("reward.png", dpi=100)
            print("\nwrote reward.png")
        except Exception as e:
            print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
