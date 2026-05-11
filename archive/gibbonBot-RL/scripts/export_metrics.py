#!/usr/bin/env python3
"""Export MLflow metric rows to a compact parquet snapshot.

Reads from mlflow.db (or the MLFLOW_TRACKING_URI) and writes one parquet file
per experiment under runs/<exp_name>/metrics.parquet. Parquet is columnar +
zstd-compressed: typically 10-50x smaller than the SQLite blob and diffs as
content (parquet files with stable row order are roughly stable on disk).

Usage:
    python scripts/export_metrics.py [--db mlflow.db] [--out runs]
"""
import argparse
import pathlib
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def export(db_uri: str, out_root: pathlib.Path) -> None:
    mlflow.set_tracking_uri(db_uri)
    client = MlflowClient()
    out_root.mkdir(parents=True, exist_ok=True)
    n_runs_total = 0
    for exp in client.search_experiments():
        rows: list[dict] = []
        for run in client.search_runs([exp.experiment_id], max_results=10_000):
            rid = run.info.run_id
            for key, val in run.data.metrics.items():
                # `val` is the latest value; pull full history for time series.
                for m in client.get_metric_history(rid, key):
                    rows.append({
                        "experiment": exp.name,
                        "run_id": rid,
                        "metric": key,
                        "step": m.step,
                        "value": m.value,
                        "timestamp": m.timestamp,
                    })
            n_runs_total += 1
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["run_id", "metric", "step"]).reset_index(drop=True)
        out_path = out_root / exp.name / "metrics.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, compression="zstd", index=False)
        print(f"[export] {exp.name}: {len(df)} rows → {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    print(f"[export] done — {n_runs_total} runs across {len(list(out_root.iterdir()))} experiments")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    ap.add_argument("--out", default="runs", help="output root (one dir per experiment)")
    args = ap.parse_args()
    db = args.db if args.db.startswith(("sqlite:", "http", "file:")) else f"sqlite:///{args.db}"
    export(db, pathlib.Path(args.out))
