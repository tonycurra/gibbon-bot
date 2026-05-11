"""Biological pose prior for gibbonBot-RL.

This module provides a pose prior based on OpenApePose dataset statistics.
It scores robot joint configurations based on their similarity to real gibbon poses.

The pose prior is implemented as a Gaussian Mixture Model (GMM) trained on
2D keypoint-derived features from OpenApePose gibbon images.

Usage:
    # Load the pose prior (will load from default path or create stub if not found)
    from envs.pose_prior import PosePrior
    pose_prior = PosePrior()

    # Score a robot pose (qpos is the full joint position vector)
    # Returns log-likelihood (higher = more gibbon-like)
    score = pose_prior.score(qpos)

    # Check if prior is available
    if pose_prior.available:
        # Use in reward
        R_bio = pose_prior.score(qpos)
    else:
        R_bio = 0.0
"""

import pathlib
from typing import Optional

import numpy as np


class PosePrior:
    """Gaussian Mixture Model-based pose prior for gibbon-like configurations."""

    def __init__(self, model_path: Optional[pathlib.Path] = None):
        """Initialize pose prior.

        Args:
            model_path: Path to GMM model file. If None, uses default path.
                       If file doesn't exist, creates a stub prior that always returns 0.
        """
        if model_path is None:
            # Default path where process_openapepose.py saves the model
            model_path = (
                pathlib.Path(__file__).parent.parent
                / "data"
                / "openapepose_processed"
                / "pose_prior_gmm.joblib"
            )

        self.model_path = model_path
        self.gmm = None
        self.feature_keys = []
        self.feature_means = None
        self.feature_stds = None

        try:
            self._load_model()
        except Exception as e:
            print(f"[PosePrior] Could not load model from {model_path}: {e}")
            print(f"[PosePrior] Using stub prior (always returns 0)")
            self._create_stub()

    @property
    def available(self) -> bool:
        """Returns True if a real pose prior model is loaded."""
        return self.gmm is not None

    def _load_model(self):
        """Load GMM model and feature information from disk."""
        import joblib

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load GMM
        self.gmm = joblib.load(self.model_path)

        # Load feature names
        feature_file = self.model_path.parent / "feature_names.txt"
        if feature_file.exists():
            with open(feature_file, "r") as f:
                self.feature_keys = [line.strip() for line in f if line.strip()]

        # Load metadata
        metadata_file = self.model_path.parent / "metadata.json"
        if metadata_file.exists():
            import json

            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            print(
                f"[PosePrior] Loaded model with {metadata.get('n_gmm_components', '?')} components"
            )
            print(f"[PosePrior] Trained on {metadata.get('train_size', '?')} samples")

        # Compute feature statistics for normalization
        features_file = self.model_path.parent / "features.npy"
        if features_file.exists():
            all_features = np.load(features_file)
            self.feature_means = np.nanmean(all_features, axis=0)
            self.feature_stds = np.nanstd(all_features, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds < 1e-6] = 1.0

    def _create_stub(self):
        """Create a stub prior that always returns 0."""
        self.gmm = None
        self.feature_keys = []
        self.feature_means = None
        self.feature_stds = None

    def _extract_robot_features(self, qpos: np.ndarray) -> np.ndarray:
        """Extract features from robot joint positions.

        This is a simplified feature extraction that maps the robot's
        24 joint positions to a feature vector comparable to the
        OpenApePose-derived features.

        Robot joint order (24 joints):
        - qpos[0:2]: world_to_torso (x, z) position
        - qpos[2:7]: torso orientation quaternion
        - qpos[7:13]: FL limb (yaw, pitch, roll, elbow, wrist_yaw, wrist_pitch)
        - qpos[13:19]: FR limb
        - qpos[19:25]: BL limb
        - qpos[25:31]: BR limb (but qpos is only 24 long, so indices differ)

        Actually, let's check the actual scene.xml structure...
        For now, we'll create a simplified mapping.

        Args:
            qpos: (n,) array of joint positions (full robot state)

        Returns:
            Feature vector as numpy array
        """
        # For now, return a simple feature vector based on limb symmetry
        # This will be refined once we have the actual dataset processed

        # Assume qpos structure:
        # 0-2: torso position (x, y, z)
        # 3-6: torso quaternion
        # 7-12: FL joint angles (6 joints)
        # 13-18: FR joint angles
        # 19-24: BL joint angles
        # 25-30: BR joint angles (but qpos is 24, so adjust)

        # Actually, from the code, ACT_DIM = 24, so actuators are qpos[7:31]
        # But qpos itself includes world pose. Let's be flexible.

        if len(qpos) < 7:
            return np.zeros(16)  # Return zeros if qpos is too short

        # Extract actuator positions (assuming they start at index 7)
        # From snapshot.py: data.qpos[7:7 + ACT_DIM] for joint positions
        n = len(qpos)
        if n >= 31:
            # Full state with all 24 actuators
            actuators = qpos[7:31]
        elif n >= 7:
            # Partial state
            actuators = qpos[7 : 7 + min(24, n - 7)]
            # Pad if needed
            if len(actuators) < 24:
                actuators = np.pad(actuators, (0, 24 - len(actuators)))
        else:
            actuators = np.zeros(24)

        # Split into limbs (6 joints each)
        fl = actuators[0:6]  # front left
        fr = actuators[6:12]  # front right
        bl = actuators[12:18]  # back left
        br = actuators[18:24]  # back right

        # Compute symmetry features
        fl_fr_sym = np.mean(np.abs(fl - fr))
        bl_br_sym = np.mean(np.abs(bl - br))
        fl_bl_sym = np.mean(np.abs(fl - bl))
        fr_br_sym = np.mean(np.abs(fr - br))

        # Joint range features (how extended are the limbs)
        fl_ext = np.mean(np.abs(fl))
        fr_ext = np.mean(np.abs(fr))
        bl_ext = np.mean(np.abs(bl))
        br_ext = np.mean(np.abs(br))

        # Combined feature vector
        # This should match the features from process_openapepose.py
        # For now, use a simplified set that can be matched later
        features = np.array(
            [
                # Limb extension
                fl_ext,
                fr_ext,
                bl_ext,
                br_ext,
                # Symmetry
                fl_fr_sym,
                bl_br_sym,
                fl_bl_sym,
                fr_br_sym,
                # Individual joint angles (simplified)
                np.mean(fl),
                np.mean(fr),
                np.mean(bl),
                np.mean(br),
                np.std(fl),
                np.std(fr),
                np.std(bl),
                np.std(br),
            ]
        )

        return features

    def score(self, qpos: np.ndarray) -> float:
        """Score a robot pose based on biological plausibility.

        Args:
            qpos: Joint position vector from MuJoCo

        Returns:
            Log-likelihood score (higher = more gibbon-like)
            Returns 0.0 if pose prior is not available
        """
        if not self.available:
            return 0.0

        # Extract features
        features = self._extract_robot_features(qpos)

        # Normalize features if we have statistics
        if self.feature_means is not None and self.feature_stds is not None:
            features = (features - self.feature_means) / self.feature_stds

        # Reshape for GMM (expects 2D array)
        features_2d = features.reshape(1, -1)

        # Compute log-likelihood
        try:
            return float(self.gmm.score(features_2d))
        except Exception as e:
            print(f"[PosePrior] Scoring error: {e}")
            return 0.0

    def score_batch(self, qpos_batch: np.ndarray) -> np.ndarray:
        """Score multiple poses at once.

        Args:
            qpos_batch: (batch_size, n_joints) array of joint positions

        Returns:
            (batch_size,) array of log-likelihood scores
        """
        if not self.available:
            return np.zeros(len(qpos_batch))

        # Extract features for each pose
        features_list = []
        for qpos in qpos_batch:
            features = self._extract_robot_features(qpos)
            if self.feature_means is not None and self.feature_stds is not None:
                features = (features - self.feature_means) / self.feature_stds
            features_list.append(features)

        features_array = np.array(features_list)

        try:
            return self.gmm.score(features_array)
        except Exception as e:
            print(f"[PosePrior] Batch scoring error: {e}")
            return np.zeros(len(qpos_batch))


# Global instance (can be replaced if needed)
_pose_prior: Optional[PosePrior] = None


def get_pose_prior() -> PosePrior:
    """Get the global pose prior instance (lazy initialization)."""
    global _pose_prior
    if _pose_prior is None:
        _pose_prior = PosePrior()
    return _pose_prior


def reset_pose_prior():
    """Reset the global pose prior (useful for testing)."""
    global _pose_prior
    _pose_prior = None
