#!/usr/bin/env python3
"""End-to-end pipeline for OpenApePose clustering and visualization.

Runs cluster_openapepose.py, then visualize_clusters.py in sequence.
Designed to be extended with plot_clusters.py once functional.

Usage:
    python scripts/pipeline_openapepose.py --annotations /path/to/oap_all.json \
        --images /path/to/OpenApePose/images \
        --output data/openapepose_clustered \
        --vis-output scripts/data/openapepose_visual_check
"""

import argparse
import pathlib
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Run OpenApePose clustering and visualization pipeline"
    )
    # cluster_openapepose args
    p.add_argument(
        "--annotations",
        type=pathlib.Path,
        required=True,
        help="Path to OpenApePose annotations JSON file",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/openapepose_clustered"),
        help="Output directory for clustered data (default: data/openapepose_clustered)",
    )
    p.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=["gibbon"],
        help="Species to filter (default: gibbon)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=2,
        help="Number of clusters (default: 2)",
    )
    # visualize_clusters args
    p.add_argument(
        "--images",
        type=pathlib.Path,
        required=True,
        help="Path to OpenApePose images directory",
    )
    p.add_argument(
        "--vis-output",
        type=pathlib.Path,
        default=pathlib.Path("scripts/data/openapepose_visual_check"),
        help="Output directory for visualization images (default: scripts/data/openapepose_visual_check)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per cluster for visualization (default: 10)",
    )
    # pipeline control
    p.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip clustering step (use existing clustered data)",
    )
    p.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip visualization step",
    )
    return p.parse_args()


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDERR: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    args = parse_args()

    scripts_dir = pathlib.Path(__file__).parent.absolute()

    # Step 1: Clustering
    if not args.skip_cluster:
        cluster_cmd = [
            sys.executable,
            str(scripts_dir / "cluster_openapepose.py"),
            "--annotations",
            str(args.annotations),
            "--output",
            str(args.output),
            "--species",
            *args.species,
            "--seed",
            str(args.seed),
            "--n-clusters",
            str(args.n_clusters),
        ]
        if not run_command(cluster_cmd, "Clustering OpenApePose data"):
            sys.exit(1)

    # Step 2: Visualization
    if not args.skip_visualize:
        vis_cmd = [
            sys.executable,
            str(scripts_dir / "visualize_clusters.py"),
            "--clustered",
            str(args.output),
            "--images",
            str(args.images),
            "--output",
            str(args.vis_output),
            "--samples",
            str(args.samples),
            "--seed",
            str(args.seed),
        ]
        if not run_command(vis_cmd, "Visualizing clusters"):
            sys.exit(1)

    # TODO: Add plot_clusters.py once functional
    # Step 3: Plot (placeholder)
    # if not args.skip_plot:
    #     plot_cmd = [
    #         sys.executable,
    #         str(scripts_dir / "plot_clusters.py"),
    #         "--clustered", str(args.output),
    #         "--output", str(args.plot_output),
    #     ]
    #     run_command(plot_cmd, "Plotting clusters")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"  Clustered data: {args.output.absolute()}")
    print(f"  Visualization: {args.vis_output.absolute()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
