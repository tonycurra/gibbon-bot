#!/usr/bin/env python3
"""Compare MLflow runs side-by-side — reward curves, distance, velocity, success rate.

Usage:
    scripts/compare_runs.py                      # last 5 runs
    scripts/compare_runs.py --top 10             # last 10
    scripts/compare_runs.py --label CPG          # only runs with "CPG" in label
    scripts/compare_runs.py --metric mean_rew    # specific metric (default: all)
    scripts/compare_runs.py --plot reward.png    # save instead of show
"""
import argparse
import pathlib
from collections import defaultdict

import mlflow
import numpy as np

EXPERIMENT = "ape_walk_mjx"
TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_METRICS = ["mean_rew", "mean_dist", "mean_speed", "mean_upright", "success_frac", "gait_coherence"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=5, help="how many recent runs")
    p.add_argument("--label", type=str, default=None, help="filter by label tag substring")
    p.add_argument("--metric", action="append", default=None, help="metric to plot (repeatable). default = all")
    p.add_argument("--plot", type=str, default=None, help="save PNG instead of showing (requires matplotlib)")
    p.add_argument("--summary-only", action="store_true", help="text-only summary, no plot")
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print(f"experiment '{EXPERIMENT}' not found")
        return

    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=args.top * 5)
    if args.label:
        runs = [r for r in runs if args.label.lower() in r.data.tags.get("label", "").lower()]
    runs = runs[: args.top]
    if not runs:
        print("no matching runs")
        return

    metrics = args.metric or DEFAULT_METRICS

    # Header
    print(f"\nComparing {len(runs)} runs from experiment '{EXPERIMENT}':")
    print(f"{'run_id':<10} {'started':<20} {'label':<60}")
    print("-" * 92)
    for r in runs:
        ts = r.info.start_time / 1000
        import datetime
        ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        label = r.data.tags.get("label", "")[:58]
        print(f"{r.info.run_id[:8]:<10} {ts_str:<20} {label}")
    print()

    # Metric summaries (final value per run)
    print(f"{'run_id':<10} | " + " | ".join(f"{m:>10}" for m in metrics))
    print("-" * (12 + sum(13 for _ in metrics)))
    for r in runs:
        rid = r.info.run_id
        row = [f"{rid[:8]}"]
        for m in metrics:
            try:
                hist = client.get_metric_history(rid, m)
                if hist:
                    row.append(f"{hist[-1].value:>10.3f}")
                else:
                    row.append(f"{'—':>10}")
            except Exception:
                row.append(f"{'?':>10}")
        print(" | ".join(row))

    # Optional plotting
    if not args.summary_only:
        try:
            import matplotlib
            if args.plot:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(metrics)
            fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
            if n == 1:
                axes = [axes]
            for i, m in enumerate(metrics):
                for r in runs:
                    rid = r.info.run_id
                    label = r.data.tags.get("label", rid[:8])[:50]
                    try:
                        hist = client.get_metric_history(rid, m)
                        if hist:
                            xs = [h.step for h in hist]
                            ys = [h.value for h in hist]
                            axes[i].plot(xs, ys, label=label, alpha=0.8)
                    except Exception:
                        continue
                axes[i].set_ylabel(m)
                axes[i].legend(fontsize=7, loc="best")
                axes[i].grid(alpha=0.3)
            axes[-1].set_xlabel("env steps")
            plt.tight_layout()
            if args.plot:
                plt.savefig(args.plot, dpi=120)
                print(f"\n[compare] plot saved to {args.plot}")
            else:
                plt.show()
        except ImportError:
            print("\n[compare] matplotlib not available — text summary only. pip install matplotlib")


if __name__ == "__main__":
    main()
