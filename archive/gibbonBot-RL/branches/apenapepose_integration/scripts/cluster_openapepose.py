#!/usr/bin/env python3
"""Cluster OpenApePose images into behaviors for gibbonBot-RL.

Usage:
    python scripts/cluster_openapepose.py --annotations /path/to/oap_all.json \
        --output data/openapepose_clustered
"""

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def parse_args():
    p = argparse.ArgumentParser(description="Cluster OpenApePose images into behaviors")
    p.add_argument("--annotations", type=pathlib.Path, required=True)
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/openapepose_clustered"),
    )
    p.add_argument("--species", type=str, nargs="+", default=["gibbon"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-clusters", type=int, default=2, help="Number of clusters (default: 2)"
    )
    return p.parse_args()


# OpenApePose keypoint indices
Nose, LEye, REye, Head = 0, 1, 2, 3
Neck, LShoulder, LElbow, LWrist = 4, 5, 6, 7
RShoulder, RElbow, RWrist = 8, 9, 10
Hip, LKnee, LFoot = 11, 12, 13
RKnee, RFoot = 14, 15


def vector_angle(a, b, c):
    """Compute angle at point b between points a and c."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def load_annotations(path):
    """Load OpenApePose annotations from JSON file."""
    print(f"Loading annotations from {path}")
    with open(path, "r") as f:
        data = json.load(f)
    annotations = data.get("data", [])
    print(f"Loaded {len(annotations)} annotations")
    return annotations


def filter_by_species(annotations, species_list):
    """Filter annotations to specific species."""
    species_list = [s.lower() for s in species_list]
    filtered = [a for a in annotations if a.get("species", "").lower() in species_list]
    counts = defaultdict(int)
    for a in annotations:
        counts[a.get("species", "unknown")] += 1
    print(f"Species distribution: {dict(counts)}")
    print(f"Filtered to {len(filtered)} annotations")
    return filtered


def extract_keypoints(ann):
    """Extract landmarks, visibility, bbox from annotation."""
    landmarks = np.array(ann.get("landmarks", []), dtype=np.float32).reshape(-1, 2)
    visibility = np.array(ann.get("visibility", []), dtype=np.float32)
    bbox = np.array(ann.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
    return landmarks, visibility, bbox


def compute_behavior_features(landmarks, visibility, bbox):
    """Compute features for behavior classification."""
    # Normalize coordinates to [0,1] using bbox
    if bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = bbox
        kp = landmarks.copy()
        kp[:, 0] = (kp[:, 0] - x) / w
        kp[:, 1] = (kp[:, 1] - y) / h
    else:
        kp = landmarks.copy()
        if len(kp) > 0:
            x_min, y_min = kp.min(axis=0)
            x_max, y_max = kp.max(axis=0)
            x_range = max(x_max - x_min, 1.0)
            y_range = max(y_max - y_min, 1.0)
            kp = (kp - [x_min, y_min]) / [x_range, y_range]

    # Torso height: head y - hip y (standing = tall)
    torso_height = (
        (kp[Head][1] - kp[Hip][1]) if visibility[Head] and visibility[Hip] else 0.0
    )

    # Torso angle: angle of line from hip to head (sitting = more horizontal)
    if visibility[Head] and visibility[Hip]:
        dx = kp[Head][0] - kp[Hip][0]
        dy = kp[Head][1] - kp[Hip][1]
        torso_angle = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
    else:
        torso_angle = 0.0

    # Foot spread: |left foot x - right foot x| (walking = wider)
    foot_spread = (
        np.abs(kp[LFoot][0] - kp[RFoot][0])
        if visibility[LFoot] and visibility[RFoot]
        else 0.0
    )

    # Knee height: avg(knee y - foot y) (standing = knees high)
    knee_heights = []
    if visibility[LKnee] and visibility[LFoot]:
        knee_heights.append(kp[LKnee][1] - kp[LFoot][1])
    if visibility[RKnee] and visibility[RFoot]:
        knee_heights.append(kp[RKnee][1] - kp[RFoot][1])
    knee_height = np.mean(knee_heights) if knee_heights else 0.0

    # Leg asymmetry: |left knee y - right knee y| (walking = asymmetric)
    leg_asymmetry = (
        np.abs(kp[LKnee][1] - kp[RKnee][1])
        if visibility[LKnee] and visibility[RKnee]
        else 0.0
    )

    # Wrist height: avg wrist y - hip y (climbing = arms raised)
    wrist_heights = []
    if visibility[LWrist] and visibility[Hip]:
        wrist_heights.append(kp[LWrist][1] - kp[Hip][1])
    if visibility[RWrist] and visibility[Hip]:
        wrist_heights.append(kp[RWrist][1] - kp[Hip][1])
    wrist_height = np.mean(wrist_heights) if wrist_heights else 0.0

    # Elbow height: avg elbow y - hip y
    elbow_heights = []
    if visibility[LElbow] and visibility[Hip]:
        elbow_heights.append(kp[LElbow][1] - kp[Hip][1])
    if visibility[RElbow] and visibility[Hip]:
        elbow_heights.append(kp[RElbow][1] - kp[Hip][1])
    elbow_height = np.mean(elbow_heights) if elbow_heights else 0.0

    # Arm asymmetry: |left elbow y - right elbow y|
    arm_asymmetry = (
        np.abs(kp[LElbow][1] - kp[RElbow][1])
        if visibility[LElbow] and visibility[RElbow]
        else 0.0
    )

    # Hand spread: |left wrist x - right wrist x|
    hand_spread = (
        np.abs(kp[LWrist][0] - kp[RWrist][0])
        if visibility[LWrist] and visibility[RWrist]
        else 0.0
    )

    # Body symmetry: avg symmetry across limb pairs
    symmetries = []
    # Leg symmetry
    if visibility[LKnee] and visibility[RKnee]:
        symmetries.append(1.0 - np.abs(kp[LKnee][1] - kp[RKnee][1]))
    if visibility[LFoot] and visibility[RFoot]:
        symmetries.append(1.0 - np.abs(kp[LFoot][0] - kp[RFoot][0]))
    # Arm symmetry
    if visibility[LElbow] and visibility[RElbow]:
        symmetries.append(1.0 - np.abs(kp[LElbow][1] - kp[RElbow][1]))
    if visibility[LWrist] and visibility[RWrist]:
        symmetries.append(1.0 - np.abs(kp[LWrist][1] - kp[RWrist][1]))
    body_symmetry = np.mean(symmetries) if symmetries else 0.0

    # Arm angle: angle at elbow (0 = bent, pi = straight)
    arm_angles = []
    if visibility[LShoulder] and visibility[LElbow] and visibility[LWrist]:
        arm_angles.append(vector_angle(kp[LShoulder], kp[LElbow], kp[LWrist]))
    if visibility[RShoulder] and visibility[RElbow] and visibility[RWrist]:
        arm_angles.append(vector_angle(kp[RShoulder], kp[RElbow], kp[RWrist]))
    arm_angle = np.mean(arm_angles) if arm_angles else 0.0

    # Leg angle: angle at knee (0 = bent, pi = straight)
    leg_angles = []
    if visibility[Hip] and visibility[LKnee] and visibility[LFoot]:
        leg_angles.append(vector_angle(kp[Hip], kp[LKnee], kp[LFoot]))
    if visibility[Hip] and visibility[RKnee] and visibility[RFoot]:
        leg_angles.append(vector_angle(kp[Hip], kp[RKnee], kp[RFoot]))
    leg_angle = np.mean(leg_angles) if leg_angles else 0.0

    # Arm length: avg distance shoulder to wrist
    arm_lengths = []
    if visibility[LShoulder] and visibility[LWrist]:
        arm_lengths.append(np.linalg.norm(kp[LShoulder] - kp[LWrist]))
    if visibility[RShoulder] and visibility[RWrist]:
        arm_lengths.append(np.linalg.norm(kp[RShoulder] - kp[RWrist]))
    arm_length = np.mean(arm_lengths) if arm_lengths else 0.0
    max_arm_length = np.max(arm_lengths) if arm_lengths else 0.0

    # Leg length: avg distance hip to foot
    leg_lengths = []
    if visibility[Hip] and visibility[LFoot]:
        leg_lengths.append(np.linalg.norm(kp[Hip] - kp[LFoot]))
    if visibility[Hip] and visibility[RFoot]:
        leg_lengths.append(np.linalg.norm(kp[Hip] - kp[RFoot]))
    leg_length = np.mean(leg_lengths) if leg_lengths else 0.0
    max_leg_length = np.max(leg_lengths) if leg_lengths else 0.0

    # Wrist above head: avg wrist y - head y (positive when wrists above head)
    wrist_above_heads = []
    if visibility[LWrist] and visibility[Head]:
        wrist_above_heads.append(kp[LWrist][1] - kp[Head][1])
    if visibility[RWrist] and visibility[Head]:
        wrist_above_heads.append(kp[RWrist][1] - kp[Head][1])
    wrist_above_head = np.mean(wrist_above_heads) if wrist_above_heads else 0.0

    return {
        "torso_height": torso_height,
        "torso_angle": torso_angle,
        "foot_spread": foot_spread,
        "knee_height": knee_height,
        "leg_asymmetry": leg_asymmetry,
        "wrist_height": wrist_height,
        "elbow_height": elbow_height,
        "arm_asymmetry": arm_asymmetry,
        "hand_spread": hand_spread,
        "body_symmetry": body_symmetry,
        "arm_angle": arm_angle,
        "leg_angle": leg_angle,
        "arm_length": arm_length,
        "leg_length": leg_length,
        "max_arm_length": max_arm_length,
        "max_leg_length": max_leg_length,
        "wrist_above_head": wrist_above_head,
    }


def cluster_behaviors(annotations, n_clusters=2, seed=42):
    """Cluster annotations by behavior using k-means."""
    X = []
    valid_indices = []
    feature_keys = [
        # "torso_height",
        "torso_angle",
        "foot_spread",
        # "knee_height",
        "leg_asymmetry",
        # "wrist_height",
        # "elbow_height",
        "arm_asymmetry",
        "hand_spread",
        # "body_symmetry",
        "arm_angle",
        "leg_angle",
        # "arm_length",
        "leg_length",
        "max_arm_length",
        "max_leg_length",
        "wrist_above_head",
    ]

    for i, ann in enumerate(annotations):
        landmarks, visibility, bbox = extract_keypoints(ann)
        if len(landmarks) != 16:
            continue
        if np.sum(visibility) < 8:
            continue

        features = compute_behavior_features(landmarks, visibility, bbox)
        X.append([features[k] for k in feature_keys])
        valid_indices.append(i)

    if len(X) < n_clusters:
        print(f"ERROR: Only {len(X)} valid samples, need at least {n_clusters}")
        return None, None, None

    X = np.array(X)
    print(f"Clustering {len(X)} samples with {len(feature_keys)} features")

    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    # Map numeric labels to behavior names (use numeric for >2 clusters)
    cluster_labels = [f"cluster_{l}" for l in labels]

    # Print cluster stats
    centers = kmeans.cluster_centers_
    for c in range(n_clusters):
        mask = labels == c
        count = np.sum(mask)
        print(f"  Cluster {c}: {count} samples")
        center_str = ", ".join(
            f"{k}={centers[c, i]:.3f}" for i, k in enumerate(feature_keys)
        )
        print(f"    Center: {center_str}")

    return cluster_labels, valid_indices, feature_keys


def save_clustered_data(output_dir, annotations, labels, valid_indices, feature_keys):
    """Save clustered data by behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing files
    for f in output_dir.glob("*.json"):
        f.unlink()

    # Separate by behavior
    behavior_groups = defaultdict(list)
    for idx, label in zip(valid_indices, labels):
        behavior_groups[label].append(annotations[idx])

    # Save each cluster
    for behavior, data in behavior_groups.items():
        with open(output_dir / f"{behavior}.json", "w") as f:
            json.dump({"data": data}, f, indent=2)
        print(f"Saved {len(data)} {behavior} samples")

    # Save metadata
    metadata = {
        "total": len(labels),
        "behaviors": {b: len(d) for b, d in behavior_groups.items()},
        "features": feature_keys,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    annotations = load_annotations(args.annotations)
    gibbons = filter_by_species(annotations, args.species)

    if len(gibbons) == 0:
        print("ERROR: No gibbon annotations found")
        return

    labels, valid_indices, feature_keys = cluster_behaviors(
        gibbons, n_clusters=args.n_clusters, seed=args.seed
    )

    if labels is not None:
        save_clustered_data(args.output, gibbons, labels, valid_indices, feature_keys)

    print("Done!")


if __name__ == "__main__":
    main()
