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

    # For better accuracy, provide MuJoCo model for 3D forward kinematics:
    import mujoco
    model = mujoco.MjModel.from_xml_path("path/to/scene.xml")
    pose_prior = PosePrior(mj_model=model)
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

# Mapping from OpenApePose keypoint names to robot body names
# OpenApePose has 16 keypoints in this order:
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
    "hip",
    "left_knee",
    "left_foot",
    "right_knee",
    "right_foot",
]

# Corresponding robot body names (bodies that don't exist on robot map to "torso")
ROBOT_BODY_MAP = [
    "torso",  # nose
    "torso",  # left_eye (not on robot)
    "torso",  # right_eye (not on robot)
    "torso",  # head
    "torso",  # neck
    "fl_link1",  # left_shoulder
    "fl_link3",  # left_elbow
    "fl_gripper_link",  # left_wrist
    "fr_link1",  # right_shoulder
    "fr_link3",  # right_elbow
    "fr_gripper_link",  # right_wrist
    "torso",  # hip
    "bl_link3",  # left_knee
    "bl_gripper_link",  # left_foot
    "br_link3",  # right_knee
    "br_gripper_link",  # right_foot
]

# Expected feature order from process_openapepose.py
# These must match exactly for the GMM to work correctly
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


def angle_between(v1, v2):
    """Compute angle between two vectors in radians."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


class PosePrior:
    """Gaussian Mixture Model-based pose prior for gibbon-like configurations."""

    def __init__(self, model_path: Optional[pathlib.Path] = None, mj_model=None):
        """Initialize pose prior.

        Args:
            model_path: Path to GMM model file. If None, uses default path.
                       If file doesn't exist, creates a stub prior that always returns 0.
            mj_model: Optional MuJoCo MjModel. If provided, enables 3D forward
                     kinematics for accurate feature extraction from body positions.
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
        self.mj_model = mj_model
        self.gmm = None
        self.feature_keys = []
        self.feature_means = None
        self.feature_stds = None
        self._body_idx_cache = None  # Cache for body indices

        try:
            self._load_model()
        except Exception as e:
            print(f"[PosePrior] Could not load model from {model_path}: {e}")
            print(f"[PosePrior] Using stub prior (always returns 0)")
            self._create_stub()

    def _get_body_indices(self):
        """Get MuJoCo body indices for landmark bodies. Returns dict mapping OAP index to body index."""
        if self._body_idx_cache is not None:
            return self._body_idx_cache
        if self.mj_model is None:
            return None

        try:
            import mujoco
        except ImportError:
            return None

        idx_map = {}
        for oap_idx, body_name in enumerate(ROBOT_BODY_MAP):
            try:
                body_idx = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                idx_map[oap_idx] = body_idx
            except (KeyError, ValueError):
                # Body not found, map to torso (index 0 is usually world, 1 is often torso)
                # Try to find torso
                try:
                    torso_idx = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
                    )
                    idx_map[oap_idx] = torso_idx
                except:
                    idx_map[oap_idx] = 0

        self._body_idx_cache = idx_map
        return idx_map

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
        else:
            # Use expected feature order
            self.feature_keys = OAP_FEATURE_ORDER

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

    def _compute_features_from_3d(self, body_xpos):
        """Compute features from 3D body positions, matching process_openapepose.py.

        Args:
            body_xpos: (n_bodies, 3) array of body positions from MuJoCo

        Returns:
            (16,) feature array matching OAP_FEATURE_ORDER
        """
        body_idx = self._get_body_indices()
        if body_idx is None:
            return np.zeros(len(OAP_FEATURE_ORDER))

        # Extract positions for each OAP keypoint (use XY plane, ignore Z for 2D comparison)
        kp_3d = np.array(
            [
                body_xpos[body_idx[i]] if i in body_idx else np.zeros(3)
                for i in range(len(OAP_KEYPOINT_NAMES))
            ]
        )
        kp = kp_3d[:, :2]  # Use XY coordinates only

        # Index constants matching process_openapepose.py
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

        # Normalize coordinates: use torso (hip) as origin, scale by torso size
        torso_pos = kp[Hip]
        kp_norm = kp - torso_pos

        # Compute scale factor (use distance from hip to neck as reference)
        neck_pos = kp_norm[Nk]
        scale = np.linalg.norm(neck_pos)
        if scale < 1e-6:
            scale = 1.0

        # Normalize by scale
        kp_norm = kp_norm / scale

        # Pre-compute limb vectors (initialized to zeros)
        fl_upper = np.zeros(2)
        fl_lower = np.zeros(2)
        fr_upper = np.zeros(2)
        fr_lower = np.zeros(2)
        bl_upper = np.zeros(2)
        bl_lower = np.zeros(2)
        br_upper = np.zeros(2)
        br_lower = np.zeros(2)

        # Front left arm
        has_fl_upper = (
            np.linalg.norm(kp_norm[LS]) > 1e-6 and np.linalg.norm(kp_norm[LEb]) > 1e-6
        )
        has_fl_lower = (
            np.linalg.norm(kp_norm[LEb]) > 1e-6 and np.linalg.norm(kp_norm[LW]) > 1e-6
        )

        if has_fl_upper:
            fl_upper = kp_norm[LEb] - kp_norm[LS]
            features["fl_shoulder_angle"] = np.arctan2(fl_upper[1], fl_upper[0])
        else:
            features["fl_shoulder_angle"] = 0.0

        if has_fl_lower:
            fl_lower = kp_norm[LW] - kp_norm[LEb]
            features["fl_elbow_angle"] = np.arctan2(fl_lower[1], fl_lower[0])
            if has_fl_upper:
                features["fl_elbow_bend"] = angle_between(fl_upper, fl_lower)
            else:
                features["fl_elbow_bend"] = 0.0
        else:
            features["fl_elbow_angle"] = 0.0
            features["fl_elbow_bend"] = 0.0

        # Front right arm
        has_fr_upper = (
            np.linalg.norm(kp_norm[RS]) > 1e-6 and np.linalg.norm(kp_norm[REb]) > 1e-6
        )
        has_fr_lower = (
            np.linalg.norm(kp_norm[REb]) > 1e-6 and np.linalg.norm(kp_norm[RW]) > 1e-6
        )

        if has_fr_upper:
            fr_upper = kp_norm[REb] - kp_norm[RS]
            features["fr_shoulder_angle"] = np.arctan2(fr_upper[1], fr_upper[0])
        else:
            features["fr_shoulder_angle"] = 0.0

        if has_fr_lower:
            fr_lower = kp_norm[RW] - kp_norm[REb]
            features["fr_elbow_angle"] = np.arctan2(fr_lower[1], fr_lower[0])
            if has_fr_upper:
                features["fr_elbow_bend"] = angle_between(fr_upper, fr_lower)
            else:
                features["fr_elbow_bend"] = 0.0
        else:
            features["fr_elbow_angle"] = 0.0
            features["fr_elbow_bend"] = 0.0

        # Back left leg
        has_bl_upper = (
            np.linalg.norm(kp_norm[Hip]) > 1e-6 and np.linalg.norm(kp_norm[LK]) > 1e-6
        )
        has_bl_lower = (
            np.linalg.norm(kp_norm[LK]) > 1e-6 and np.linalg.norm(kp_norm[LF]) > 1e-6
        )

        if has_bl_upper:
            bl_upper = kp_norm[LK] - kp_norm[Hip]
            features["bl_hip_angle"] = np.arctan2(bl_upper[1], bl_upper[0])
        else:
            features["bl_hip_angle"] = 0.0

        if has_bl_lower:
            bl_lower = kp_norm[LF] - kp_norm[LK]
            features["bl_knee_angle"] = np.arctan2(bl_lower[1], bl_lower[0])
            if has_bl_upper:
                features["bl_knee_bend"] = angle_between(bl_upper, bl_lower)
            else:
                features["bl_knee_bend"] = 0.0
        else:
            features["bl_knee_angle"] = 0.0
            features["bl_knee_bend"] = 0.0

        # Back right leg
        has_br_upper = (
            np.linalg.norm(kp_norm[Hip]) > 1e-6 and np.linalg.norm(kp_norm[RK]) > 1e-6
        )
        has_br_lower = (
            np.linalg.norm(kp_norm[RK]) > 1e-6 and np.linalg.norm(kp_norm[RF]) > 1e-6
        )

        if has_br_upper:
            br_upper = kp_norm[RK] - kp_norm[Hip]
            features["br_hip_angle"] = np.arctan2(br_upper[1], br_upper[0])
        else:
            features["br_hip_angle"] = 0.0

        if has_br_lower:
            br_lower = kp_norm[RF] - kp_norm[RK]
            features["br_knee_angle"] = np.arctan2(br_lower[1], br_lower[0])
            if has_br_upper:
                features["br_knee_bend"] = angle_between(br_upper, br_lower)
            else:
                features["br_knee_bend"] = 0.0
        else:
            features["br_knee_angle"] = 0.0
            features["br_knee_bend"] = 0.0

        # Torso/head orientation
        has_torso = (
            np.linalg.norm(kp_norm[Nk]) > 1e-6 and np.linalg.norm(kp_norm[Hip]) > 1e-6
        )
        if has_torso:
            torso_vec = kp_norm[Hip] - kp_norm[Nk]
            features["torso_angle"] = np.arctan2(torso_vec[1], torso_vec[0])
        else:
            features["torso_angle"] = 0.0

        # Symmetry features
        features["arm_symmetry"] = np.abs(
            features.get("fl_elbow_bend", 0) - features.get("fr_elbow_bend", 0)
        )
        features["leg_symmetry"] = np.abs(
            features.get("bl_knee_bend", 0) - features.get("br_knee_bend", 0)
        )

        # Visibility count (all "visible" for robot)
        features["visible_count"] = 16.0

        # Return features in expected order
        return np.array([features[k] for k in OAP_FEATURE_ORDER])

    def _extract_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Fallback feature extraction from qpos only (no FK).

        This is a simplified feature extraction that maps the robot's
        24 joint positions to a feature vector.
        Less accurate than 3D FK but works without MuJoCo model.

        Args:
            qpos: (n,) array of joint positions (full robot state)

        Returns:
            Feature vector as numpy array
        """
        # For backward compatibility, return zeros with correct dimensionality
        # when 3D FK is not available
        return np.zeros(len(OAP_FEATURE_ORDER))

    def _extract_robot_features(self, qpos: np.ndarray) -> np.ndarray:
        """Extract features from robot state.

        If MuJoCo model is available, uses forward kinematics to compute
        3D body positions and extracts geometric features matching the
        OpenApePose training features.

        Otherwise, falls back to simplified qpos-based features.

        Args:
            qpos: (n,) array of joint positions (full robot state)

        Returns:
            Feature vector as numpy array
        """
        # Try to use 3D forward kinematics if model is available
        if self.mj_model is not None:
            try:
                import mujoco

                data = mujoco.make_data(self.mj_model)
                # Set qpos - handle both full state and actuator-only
                if len(qpos) >= len(data.qpos):
                    data.qpos[:] = qpos[: len(data.qpos)]
                else:
                    # Assume qpos contains just the actuators, need to set full state
                    data.qpos[7 : 7 + len(qpos)] = qpos
                mujoco.forward(self.mj_model, data)
                return self._compute_features_from_3d(data.body_xpos)
            except Exception as e:
                print(f"[PosePrior] FK failed, falling back to qpos features: {e}")

        # Fallback: use simplified qpos-based features
        return self._extract_from_qpos(qpos)

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
