#!/usr/bin/env python3
"""Extract joint limits from a clustered OpenApePose dataset.

Computes statistical ranges (min, max, mean, std) for approximate joint angles
derived from 2D keypoints in a single behavior cluster.

Usage:
    python scripts/extract_cluster_limits.py \
        --cluster scripts/data/openapepose_clustered/cluster_0.json \
        --output data/cluster_0_joint_limits.json
"""

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

# OpenApePose keypoint indices (16 landmarks)
Nose, LEye, REye, Head = 0, 1, 2, 3
Neck, LShoulder, LElbow, LWrist = 4, 5, 6, 7
RShoulder, RElbow, RWrist = 8, 9, 10
Hip, LKnee, LFoot = 11, 12, 13
RKnee, RFoot = 14, 15

# Joint names for output (mapping OAP keypoints to approximate joint angles)
JOINT_NAMES = [
    "fl_shoulder",  # Left shoulder angle (from shoulder to elbow vector)
    "fl_elbow",  # Left elbow angle (from elbow to wrist vector)
    "fr_shoulder",  # Right shoulder angle
    "fr_elbow",  # Right elbow angle
    "bl_hip",  # Back-left hip angle (from hip to knee vector)
    "bl_knee",  # Back-left knee angle (from knee to foot vector)
    "br_hip",  # Back-right hip angle
    "br_knee",  # Back-right knee angle
    "torso",  # Torso angle (from neck to hip vector)
    "neck",  # Neck angle (from head to neck vector)
]


def vector_to_angle(v):
    """Convert a 2D vector to angle in radians."""
    return np.arctan2(v[1], v[0])


def extract_joint_angles(landmarks, visibility, bbox):
    """Extract approximate joint angles from 2D landmarks.

    Args:
        landmarks: (16, 2) array of x,y coordinates
        visibility: (16,) array of 0/1 flags
        bbox: [x, y, w, h] bounding box

    Returns:
        Dict mapping joint names to angles in radians, or None if not enough
        visible keypoints.
    """
    kp = landmarks.copy()

    # Normalize coordinates to [0, 1] using bbox
    if bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = bbox
        kp[:, 0] = (kp[:, 0] - x) / w
        kp[:, 1] = (kp[:, 1] - y) / h
    else:
        # Fallback: normalize by landmark extent
        if len(kp) > 0:
            x_min, y_min = kp.min(axis=0)
            x_max, y_max = kp.max(axis=0)
            x_range = max(x_max - x_min, 1e-6)
            y_range = max(y_max - y_min, 1e-6)
            kp[:, 0] = (kp[:, 0] - x_min) / x_range
            kp[:, 1] = (kp[:, 1] - y_min) / y_range

    angles = {}

    # Front left arm
    if visibility[LShoulder] and visibility[LElbow]:
        v = kp[LElbow] - kp[LShoulder]
        angles["fl_shoulder"] = vector_to_angle(v)
    if visibility[LElbow] and visibility[LWrist]:
        v = kp[LWrist] - kp[LElbow]
        angles["fl_elbow"] = vector_to_angle(v)

    # Front right arm
    if visibility[RShoulder] and visibility[RElbow]:
        v = kp[RElbow] - kp[RShoulder]
        angles["fr_shoulder"] = vector_to_angle(v)
    if visibility[RElbow] and visibility[RWrist]:
        v = kp[RWrist] - kp[RElbow]
        angles["fr_elbow"] = vector_to_angle(v)

    # Back left leg
    if visibility[Hip] and visibility[LKnee]:
        v = kp[LKnee] - kp[Hip]
        angles["bl_hip"] = vector_to_angle(v)
    if visibility[LKnee] and visibility[LFoot]:
        v = kp[LFoot] - kp[LKnee]
        angles["bl_knee"] = vector_to_angle(v)

    # Back right leg
    if visibility[Hip] and visibility[RKnee]:
        v = kp[RKnee] - kp[Hip]
        angles["br_hip"] = vector_to_angle(v)
    if visibility[RKnee] and visibility[RFoot]:
        v = kp[RFoot] - kp[RKnee]
        angles["br_knee"] = vector_to_angle(v)

    # Torso (neck to hip)
    if visibility[Neck] and visibility[Hip]:
        v = kp[Hip] - kp[Neck]
        angles["torso"] = vector_to_angle(v)

    # Neck (head to neck)
    if visibility[Head] and visibility[Neck]:
        v = kp[Neck] - kp[Head]
        angles["neck"] = vector_to_angle(v)

    return angles


def compute_statistics(values):
    """Compute statistics for a list of values."""
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p1": float(np.percentile(arr, 1)),  # 1st percentile (conservative lower bound)
        "p99": float(
            np.percentile(arr, 99)
        ),  # 99th percentile (conservative upper bound)
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "n": len(arr),
    }


def extract_limits_from_cluster(cluster_path):
    """Extract joint angle statistics from a cluster file.

    Args:
        cluster_path: Path to cluster JSON file

    Returns:
        Dict mapping joint names to their statistics
    """
    with open(cluster_path, "r") as f:
        data = json.load(f)

    # Collect angles per joint
    all_angles = defaultdict(list)

    for ann in data.get("data", []):
        landmarks = np.array(ann.get("landmarks", []), dtype=float)
        visibility = np.array(ann.get("visibility", []), dtype=float)
        bbox = np.array(ann.get("bbox", []), dtype=float)

        # Skip invalid entries
        if len(landmarks) != 32 or len(visibility) != 16:
            continue

        landmarks = landmarks.reshape(-1, 2)
        angles = extract_joint_angles(landmarks, visibility, bbox)

        for joint, angle in angles.items():
            all_angles[joint].append(angle)

    # Compute statistics for each joint
    limits = {}
    for joint in JOINT_NAMES:
        if joint in all_angles and len(all_angles[joint]) > 10:
            limits[joint] = compute_statistics(all_angles[joint])

    return limits


def save_limits(limits, output_path):
    """Save joint limits to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(limits, f, indent=2)
    print(f"Saved joint limits to {output_path}")


def print_summary(limits):
    """Print a summary of the extracted limits."""
    print("\n" + "=" * 60)
    print("Joint Limits Summary")
    print("=" * 60)
    for joint, stats in sorted(limits.items()):
        print(
            f"  {joint:15s}: [{stats['p1']:7.3f}, {stats['p99']:7.3f}] "
            f"mean={stats['mean']:7.3f} std={stats['std']:6.3f} n={stats['n']}"
        )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract joint limits from a clustered OpenApePose dataset"
    )
    parser.add_argument(
        "--cluster",
        type=pathlib.Path,
        default=pathlib.Path("scripts/data/openapepose_clustered/cluster_0.json"),
        help="Path to cluster JSON file",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/cluster_0_joint_limits.json"),
        help="Output path for joint limits JSON",
    )
    args = parser.parse_args()

    print(f"Loading cluster from {args.cluster}")
    limits = extract_limits_from_cluster(args.cluster)

    if not limits:
        print("ERROR: No valid joint angles extracted. Check input file.")
        return

    print_summary(limits)
    save_limits(limits, args.output)


if __name__ == "__main__":
    main()
