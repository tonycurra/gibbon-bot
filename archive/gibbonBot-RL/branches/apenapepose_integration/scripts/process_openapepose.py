#!/usr/bin/env python3
"""Process OpenApePose dataset for gibbonBot-RL integration.

This script:
1. Loads OpenApePose JSON annotations
2. Extracts gibbon-specific subset
3. Converts 2D keypoints to joint angles (approximate)
4. Trains a Gaussian Mixture Model (GMM) for pose prior
5. Saves processed data for use in training

Usage:
    # First download: dataset should be in data/OpenApePose/
    #   with annotations/oap_all.json and images/ directory

    python scripts/process_openapepose.py

    # Or specify custom paths:
    python scripts/process_openapepose.py --annotations data/OpenApePose/annotations/oap_all.json \
        --output data/openapepose_processed
"""

import argparse
import json
import pathlib
from collections import defaultdict

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture


def parse_args():
    p = argparse.ArgumentParser(
        description="Process OpenApePose dataset for robot training"
    )
    p.add_argument(
        "--annotations",
        type=pathlib.Path,
        default=pathlib.Path("data/OpenApePose/annotations/oap_all.json"),
        help="Path to OpenApePose annotation JSON file",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/openapepose_processed"),
        help="Output directory for processed data",
    )
    p.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=["gibbon"],
        help="Species to extract (default: gibbon only)",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=16,
        help="Number of GMM components for pose prior",
    )
    p.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for testing",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return p.parse_args()


# OpenApePose keypoint order (16 landmarks)
OAP_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "head",
    "neck",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "hip",  # Hip/Sacrum
    "left_knee",
    "left_foot",
    "right_knee",
    "right_foot",
]

# Feature order - must match OAP_FEATURE_ORDER in pose_prior.py
OAP_FEATURE_ORDER = [
    "fl_shoulder_angle",
    "fl_elbow_angle",
    "fl_elbow_bend",
    "fr_shoulder_angle",
    "fr_elbow_angle",
    "fr_elbow_bend",
    "bl_hip_angle",
    "bl_knee_angle",
    "bl_knee_bend",
    "br_hip_angle",
    "br_knee_angle",
    "br_knee_bend",
    "torso_angle",
    "arm_symmetry",
    "leg_symmetry",
    "visible_count",
]

# Mapping from OpenApePose keypoints to robot limb chains
# Robot has: FL (front left), FR (front right), BL (back left), BR (back right)
# Each limb has: shoulder/hip, elbow/knee, wrist/foot (simplified from 6-joint robot)
ROBOT_LIMB_MAP = {
    # Front left limb
    "left_shoulder": "FL_shoulder",
    "left_elbow": "FL_elbow",
    "left_wrist": "FL_hand",
    # Front right limb
    "right_shoulder": "FR_shoulder",
    "right_elbow": "FR_elbow",
    "right_wrist": "FR_hand",
    # Back left limb
    "left_knee": "BL_knee",
    "left_foot": "BL_foot",
    # Back right limb
    "right_knee": "BR_knee",
    "right_foot": "BR_foot",
    # Torso/head
    "nose": "head",
    "head": "head",
    "neck": "neck",
    "hip": "torso",
}


def load_annotations(path):
    """Load OpenApePose annotations from JSON file."""
    print(f"[OAP] Loading annotations from {path}")
    with open(path, "r") as f:
        data = json.load(f)

    annotations = data.get("data", [])
    print(f"[OAP] Loaded {len(annotations)} annotations")
    return annotations


def filter_by_species(annotations, species_list):
    """Filter annotations to specific species."""
    species_list = [s.lower() for s in species_list]
    filtered = [a for a in annotations if a.get("species", "").lower() in species_list]
    print(f"[OAP] Filtered to {len(filtered)} annotations for species: {species_list}")

    # Count per species
    counts = defaultdict(int)
    for a in annotations:
        counts[a.get("species", "unknown")] += 1
    print(f"[OAP] Species distribution: {dict(counts)}")

    return filtered


def extract_keypoints(annotations):
    """Extract keypoints from annotations.

    Returns:
        List of dicts with: file, species, bbox, landmarks (as numpy array), visibility
    """
    results = []
    for ann in annotations:
        result = {
            "file": ann.get("file", ""),
            "species": ann.get("species", "unknown"),
            "bbox": np.array(ann.get("bbox", [0, 0, 0, 0])),
            "landmarks": np.array(ann.get("landmarks", []), dtype=np.float32).reshape(
                -1, 2
            ),
            "visibility": np.array(ann.get("visibility", []), dtype=np.float32),
        }
        results.append(result)
    return results


def keypoints_to_joint_angles(keypoints, visibility, bbox=None):
    """Convert 2D keypoints to approximate joint angles.

    This is a simplified conversion that computes:
    - Limb segment vectors from keypoints
    - Angles between segments (approximate joint angles)

    For a proper 3D conversion, you'd need:
    - Camera calibration
    - Multi-view geometry
    - Skeleton fitting (e.g., SMPL-X for animals)

    But for a pose prior, 2D-derived angles can still capture
    biologically plausible configurations.

    Args:
        keypoints: (16, 2) array of x,y coordinates
        visibility: (16,) array of 0/1 visibility flags
        bbox: [x, y, w, h] bounding box (optional, for normalization)

    Returns:
        Dict of joint angles and features for pose prior
    """
    # Index constants for keypoint array
    N = 0  # nose
    LE = 1  # left eye
    RE = 2  # right eye
    H = 3  # head
    Nk = 4  # neck
    LS = 5  # left shoulder
    LEb = 6  # left elbow
    LW = 7  # left wrist
    RS = 8  # right shoulder
    REb = 9  # right elbow
    RW = 10  # right wrist
    Hip = 11  # hip/sacrum
    LK = 12  # left knee
    LF = 13  # left foot
    RK = 14  # right knee
    RF = 15  # right foot

    features = {}

    # Normalize coordinates: use hip as origin, scale by neck-hip distance
    # This matches pose_prior.py's _compute_features_from_3d() normalization
    torso_pos = keypoints[Hip]
    kp_norm = keypoints - torso_pos

    # Compute scale factor (use distance from hip to neck as reference)
    neck_pos = kp_norm[Nk]
    scale = np.linalg.norm(neck_pos)
    if scale < 1e-6:
        scale = 1.0

    # Normalize by scale
    kp_norm = kp_norm / scale

    # Compute limb segment vectors (normalized)
    # Front left arm
    if visibility[LS] > 0 and visibility[LEb] > 0:
        fl_upper = kp_norm[LEb] - kp_norm[LS]  # shoulder to elbow
        features["fl_shoulder_angle"] = np.arctan2(fl_upper[1], fl_upper[0])
    else:
        features["fl_shoulder_angle"] = 0.0

    if visibility[LEb] > 0 and visibility[LW] > 0:
        fl_lower = kp_norm[LW] - kp_norm[LEb]  # elbow to wrist
        features["fl_elbow_angle"] = np.arctan2(fl_lower[1], fl_lower[0])
        # Angle between upper and lower arm
        if visibility[LS] > 0:
            features["fl_elbow_bend"] = angle_between(fl_upper, fl_lower)
        else:
            features["fl_elbow_bend"] = 0.0
    else:
        features["fl_elbow_angle"] = 0.0
        features["fl_elbow_bend"] = 0.0

    # Front right arm
    if visibility[RS] > 0 and visibility[REb] > 0:
        fr_upper = kp_norm[REb] - kp_norm[RS]
        features["fr_shoulder_angle"] = np.arctan2(fr_upper[1], fr_upper[0])
    else:
        features["fr_shoulder_angle"] = 0.0

    if visibility[REb] > 0 and visibility[RW] > 0:
        fr_lower = kp_norm[RW] - kp_norm[REb]
        features["fr_elbow_angle"] = np.arctan2(fr_lower[1], fr_lower[0])
        if visibility[RS] > 0:
            features["fr_elbow_bend"] = angle_between(fr_upper, fr_lower)
        else:
            features["fr_elbow_bend"] = 0.0
    else:
        features["fr_elbow_angle"] = 0.0
        features["fr_elbow_bend"] = 0.0

    # Back left leg
    if visibility[Hip] > 0 and visibility[LK] > 0:
        bl_upper = kp_norm[LK] - kp_norm[Hip]
        features["bl_hip_angle"] = np.arctan2(bl_upper[1], bl_upper[0])
    else:
        features["bl_hip_angle"] = 0.0

    if visibility[LK] > 0 and visibility[LF] > 0:
        bl_lower = kp_norm[LF] - kp_norm[LK]
        features["bl_knee_angle"] = np.arctan2(bl_lower[1], bl_lower[0])
        if visibility[Hip] > 0:
            features["bl_knee_bend"] = angle_between(bl_upper, bl_lower)
        else:
            features["bl_knee_bend"] = 0.0
    else:
        features["bl_knee_angle"] = 0.0
        features["bl_knee_bend"] = 0.0

    # Back right leg
    if visibility[Hip] > 0 and visibility[RK] > 0:
        br_upper = kp_norm[RK] - kp_norm[Hip]
        features["br_hip_angle"] = np.arctan2(br_upper[1], br_upper[0])
    else:
        features["br_hip_angle"] = 0.0

    if visibility[RK] > 0 and visibility[RF] > 0:
        br_lower = kp_norm[RF] - kp_norm[RK]
        features["br_knee_angle"] = np.arctan2(br_lower[1], br_lower[0])
        if visibility[Hip] > 0:
            features["br_knee_bend"] = angle_between(br_upper, br_lower)
        else:
            features["br_knee_bend"] = 0.0
    else:
        features["br_knee_angle"] = 0.0
        features["br_knee_bend"] = 0.0

    # Torso/head orientation
    if visibility[Nk] > 0 and visibility[Hip] > 0:
        torso_vec = kp_norm[Hip] - kp_norm[Nk]
        features["torso_angle"] = np.arctan2(torso_vec[1], torso_vec[0])
    else:
        features["torso_angle"] = 0.0

    # Symmetry features (how similar are left/right sides)
    features["arm_symmetry"] = np.abs(
        features.get("fl_elbow_bend", 0) - features.get("fr_elbow_bend", 0)
    )
    features["leg_symmetry"] = np.abs(
        features.get("bl_knee_bend", 0) - features.get("br_knee_bend", 0)
    )

    # Visibility count
    features["visible_count"] = np.sum(visibility)

    return features


def angle_between(v1, v2):
    """Compute angle between two vectors in radians."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def train_pose_prior(features_list, n_components=16, seed=42):
    """Train a Gaussian Mixture Model on pose features.

    Args:
        features_list: List of feature dicts from keypoints_to_joint_angles
        n_components: Number of GMM components
        seed: Random seed

    Returns:
        Tuple of (trained GMM model, feature keys)
    """
    # Convert list of dicts to numpy array using fixed feature order
    X = np.array([[f[k] for k in OAP_FEATURE_ORDER] for f in features_list])

    print(
        f"[OAP] Training GMM on {len(X)} samples with {len(OAP_FEATURE_ORDER)} features"
    )
    print(f"[OAP] Using {n_components} components")

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="diag", random_state=seed
    )
    gmm.fit(X)

    # Compute log-likelihood on training data
    train_ll = gmm.score(X)
    print(f"[OAP] Training log-likelihood: {train_ll:.4f}")

    return gmm, OAP_FEATURE_ORDER


def save_processed_data(
    output_dir,
    annotations,
    features_list,
    gmm,
    feature_keys,
    species,
    test_split=0.1,
    seed=42,
):
    """Save processed data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    n = len(features_list)
    indices = rng.permutation(n)
    split_idx = int(n * (1 - test_split))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # Save metadata
    metadata = {
        "total_samples": len(annotations),
        "species": species,
        "features": feature_keys,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "n_gmm_components": gmm.n_components,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save train/test indices
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "test_indices.npy", test_idx)

    # Save features
    all_features = np.array([[f[k] for k in feature_keys] for f in features_list])
    np.save(output_dir / "features.npy", all_features)

    # Save GMM
    joblib.dump(gmm, output_dir / "pose_prior_gmm.joblib")

    # Save feature names
    with open(output_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_keys))

    print(f"[OAP] Saved processed data to {output_dir}")
    print(f"[OAP]   metadata.json")
    print(f"[OAP]   features.npy ({all_features.shape})")
    print(f"[OAP]   pose_prior_gmm.joblib")
    print(f"[OAP]   train_indices.npy ({len(train_idx)} samples)")
    print(f"[OAP]   test_indices.npy ({len(test_idx)} samples)")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load annotations
    annotations = load_annotations(args.annotations)

    # Filter by species
    annotations = filter_by_species(annotations, args.species)

    # Extract keypoints
    keypoints_data = extract_keypoints(annotations)

    # Convert to features
    print("[OAP] Converting keypoints to joint angle features...")
    features_list = []
    valid_count = 0
    invalid_count = 0

    for i, kp_data in enumerate(keypoints_data):
        if kp_data["landmarks"].shape[0] == 16:  # 16 keypoints
            try:
                features = keypoints_to_joint_angles(
                    kp_data["landmarks"], kp_data["visibility"], kp_data["bbox"]
                )
                features_list.append(features)
                valid_count += 1
            except Exception as e:
                invalid_count += 1
        else:
            invalid_count += 1

    print(
        f"[OAP] Converted {valid_count} valid annotations, skipped {invalid_count} invalid"
    )

    if len(features_list) == 0:
        print("[OAP] ERROR: No valid features extracted!")
        return

    # Train pose prior
    gmm, feature_keys = train_pose_prior(
        features_list, n_components=args.n_components, seed=args.seed
    )

    # Save processed data
    save_processed_data(
        args.output,
        annotations,
        features_list,
        gmm,
        feature_keys,
        species=args.species,
        test_split=args.test_split,
        seed=args.seed,
    )

    print("[OAP] Done!")


if __name__ == "__main__":
    main()
