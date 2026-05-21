#!/usr/bin/env python3
"""Copy random images from each cluster for visual verification.

Usage:
    python scripts/visualize_clusters.py --clustered data/openapepose_clustered \
        --images /path/to/OpenApePose/images \
        --output scripts/data/openapepose_visual_check \
        --samples 10
"""

import argparse
import json
import pathlib
import random
import shutil


def parse_args():
    p = argparse.ArgumentParser(
        description="Copy random images from clusters for visual verification"
    )
    p.add_argument(
        "--clustered",
        type=pathlib.Path,
        required=True,
        help="Directory with cluster_*.json files from cluster_openapepose",
    )
    p.add_argument(
        "--images",
        type=pathlib.Path,
        required=True,
        help="OpenApePose images directory",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("scripts/data/openapepose_visual_check"),
        help="Output directory for copied images",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random samples per cluster (default: 10)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Validate inputs
    if not args.clustered.exists():
        print(f"ERROR: Clustered directory not found: {args.clustered}")
        return
    if not args.images.exists():
        print(f"ERROR: Images directory not found: {args.images}")
        return

    # Find all cluster files
    cluster_files = sorted(args.clustered.glob("cluster_*.json"))
    if not cluster_files:
        print(f"ERROR: No cluster_*.json files found in {args.clustered}")
        return

    print(f"Found {len(cluster_files)} clusters")

    # Clear output directory if it exists, then recreate
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for cluster_file in cluster_files:
        cluster_name = cluster_file.stem  # e.g., "cluster_0"
        cluster_dir = args.output / cluster_name
        cluster_dir.mkdir(exist_ok=True)

        # Load cluster data
        with open(cluster_file, "r") as f:
            data = json.load(f)
        annotations = data.get("data", [])

        if not annotations:
            print(f"WARNING: No annotations in {cluster_file}")
            continue

        # Get unique files (in case of duplicates) and sort for deterministic sampling
        image_files = [ann.get("file", "") for ann in annotations if ann.get("file")]
        unique_files = sorted(set(image_files))

        if len(unique_files) < args.samples:
            print(
                f"WARNING: {cluster_name} has only {len(unique_files)} images, sampling all"
            )
            samples = unique_files
        else:
            samples = random.sample(unique_files, args.samples)

        # Copy images
        for img_file in samples:
            src = args.images / img_file
            dst = cluster_dir / img_file
            if src.exists():
                shutil.copy2(src, dst)
                total_copied += 1
            else:
                print(f"WARNING: Image not found: {src}")

        print(f"  {cluster_name}: copied {len(samples)} images")

    print(f"\nDone! Copied {total_copied} images to {args.output}")
    print(f"\nTo view: open {args.output.absolute()}")


if __name__ == "__main__":
    main()
