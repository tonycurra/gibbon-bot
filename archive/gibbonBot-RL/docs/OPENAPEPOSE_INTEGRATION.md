# OpenApePose Integration - Robot Movement Improvement

*Generated from analysis of OpenApePose dataset and gibbonBot-RL codebase.*

---

## Dataset Overview

| Aspect | Details |
|--------|---------|
| **Total Images** | 71,868 |
| **Species** | Bonobos (11,685), Chimpanzees (18,010), **Gibbons (9,274)**, Gorillas (12,905), Orangutans (12,722), Siamangs (7,272) |
| **Keypoints** | 16 landmarks: Nose, L/R eye, Head, Neck, L/R shoulder, L/R elbow, L/R wrist, Hip/Sacrum, L/R knee, L/R foot |
| **Annotation Format** | JSON with `file`, `species`, `bbox`, `landmarks` (x,y ×16), `visibility` (0/1 ×16) |
| **Pretrained Model** | HRNet-W48 (MMPose v0.26.0) |
| **License** | MIT |
| **Download** | [Google Drive](https://drive.google.com/drive/folders/1UF2Wys2JwA4W1kGIiJIvvAZXZWrnzcBk?usp=share_link) | [Dryad](https://datadryad.org/stash/share/yIccvg7QBQ_UeRMZEE3lyHasDNBiBg61y8oblJDOnEI) |

---

## Current Robot State

The gibbonBot-RL uses:
- **CPG (Central Pattern Generator)** - provides rhythmic baseline gait (trot/pace)
- **PPO** - learns corrections to CPG for balance, direction, amplitude
- **Reward shaping** - hand-tuned weights for height, upright, progress, smoothness
- **24 actuators** - 4 limbs × 6 joints each

Current limitation: CPG-only motion doesn't translate body forward without body lift. PPO converges to functional but potentially unnatural gaits.

---

## Integration Opportunities

### 1. Imitation Learning (AMP/DeepMimic-style)

**Goal:** Use real gibbon motion as reference for more natural-looking gait.

**Pipeline:**
```
OpenApePose 2D keypoints (gibbon subset: 9,274 images)
    ↓
Sequence into motion clips (sort by temporal coherence)
    ↓
Lift to 3D (multi-view geometry or monocular depth + temporal consistency)
    ↓
Retarget to URDF skeleton (map ape joints → robot joints)
    ↓
Create reference motion library
    ↓
Add imitation reward term: R_imit = -||q_robot - q_reference||²
```

**Keypoint to Robot Body Mapping:**

| OpenApePose Landmark | Robot URDF Body | Notes |
|---------------------|-----------------|-------|
| Nose, Head, Neck | `torso` | Orientation reference |
| Left Shoulder | `fl_link1` | Front left shoulder |
| Left Elbow | `fl_link3` | Front left elbow |
| Left Wrist | `fl_gripper_link` | Front left hand |
| Right Shoulder | `fr_link1` | Front right shoulder |
| Right Elbow | `fr_link3` | Front right elbow |
| Right Wrist | `fr_gripper_link` | Front right hand |
| Hip/Sacrum | `torso` | Root reference |
| Left Knee | `bl_link3` | Back left knee |
| Left Foot | `bl_gripper_link` | Back left foot |
| Right Knee | `br_link3` | Back right knee |
| Right Foot | `br_gripper_link` | Back right foot |

**Note:** The `snapshot.py` script already extracts these exact robot landmarks for analysis.

**Cost:** 1-2 weeks (pose extraction → skeleton retarget → physics validation → reward integration → tuning)

**Priority:** Do if reward-shaping plateaus and gait naturalism matters.

---

### 2. Biological Pose Prior Reward

**Goal:** Penalize unnatural robot configurations using real gibbon pose statistics.

**Implementation:**
```python
# In walk_env_mjx.py reward function
W_BIO_POSE = 0.5  # New weight

# Train GMM on OpenApePose gibbon keypoints converted to joint angles
gibbon_pose_model = load_gmm("data/openapepose/gibbon_pose_gmm.pkl")

# Compute log-likelihood of current robot pose
pose_logprob = gibbon_pose_model.score(robot_joint_angles)
R_bio = pose_logprob  # Maximize likelihood of being gibbon-like
```

**Steps:**
1. Download OpenApePose dataset
2. Extract gibbon subset (9,274 images)
3. Convert 2D keypoints to approximate joint angles
4. Train Gaussian Mixture Model (GMM) on joint angle data
5. Add `W_BIO_POSE` term to reward function

**Cost:** 2-3 days

**Priority:** High - quick win for more natural poses

---

### 3. Gait Phase Validation Metric

**Goal:** Automated evaluation of learned gait quality against real gibbon locomotion.

**Implementation:**
```python
# Extract limb phase relationships from OpenApePose sequences
# Compare to robot limb phases from snapshot.py landmarks

def gait_similarity(robot_landmarks, gibbon_reference):
    # Compute phase correlation between limb movements
    robot_phases = compute_limb_phases(robot_landmarks)
    gibbon_phases = gibbon_reference['phases']
    return np.corrcoef(robot_phases, gibbon_phases)
```

**Use cases:**
- Automated evaluation metric beyond distance-to-tree
- Debugging: identify which limb coordination patterns are unnatural
- Curriculum: gate progression on gait quality thresholds

**Cost:** 1 day (after dataset download)

---

### 4. Curriculum Learning with Pose Difficulty

**Goal:** Progressive training on easier → harder motions.

**Implementation:**
```python
# Classify OpenApePose images by pose difficulty
POSE_DIFFICULTY = {
    'resting': 1,      # Low velocity, symmetric
    'standing': 1,
    'walking': 2,      # Asymmetric, moderate velocity
    'reaching': 3,
    'leaping': 4,      # High velocity, extreme poses
}

# Use difficulty as curriculum stage
STAGE_1: Train on resting/standing poses only
STAGE_2: Add walking poses
STAGE_3: Add reaching poses
STAGE_4: Add leaping poses
```

**Cost:** 2-3 days

---

## Recommended Roadmap

| Phase | Task | Cost | Priority | Dependencies |
|-------|------|------|----------|--------------|
| 1 | Download OpenApePose dataset | 1 hr | P0 | None |
| 2 | Extract gibbon subset, analyze statistics | 1 day | P0 | Phase 1 |
| 3 | Add `W_BIO_POSE` reward (2D GMM) | 2 days | **P1** | Phase 2 |
| 4 | Implement gait similarity metric | 1 day | P2 | Phase 2 |
| 5 | Build 3D reference motion library | 1 week | P2 | Phase 2 |
| 6 | Full imitation learning with AMP | 1-2 weeks | P2 | Phase 5 |
| 7 | Pose-based curriculum | 2 days | P2 | Phase 2 |

---

## Immediate Next Steps

1. **Download the data** from Google Drive or Dryad
2. **Verify existing infrastructure** - `snapshot.py` already has landmark extraction
3. **Start with Biological Pose Prior** - lowest cost, immediate improvement

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `data/OpenApePose/` | Download | Dataset storage |
| `scripts/process_openapepose.py` | Create | Extract gibbon subset, convert to useful format |
| `envs/pose_prior.py` | Create | GMM training and scoring |
| `envs/walk_env_mjx.py` | Modify | Add `W_BIO_POSE` reward term |
| `envs/params_walk_mjx.py` | Modify | Add `W_BIO_POSE` parameter |
| `utils/gait_analysis.py` | Create | Gait similarity metrics |

---

## References

- [OpenApePose Paper](https://elifesciences.org/articles/86873)
- [MMPose Documentation](https://mmpose.readthedocs.io/en/latest/)
- [DeepMimic Paper](https://arxiv.org/abs/1804.02717) - imitation learning reference
- [AMP Paper](https://arxiv.org/abs/2010.08184) - Adversarial Motion Priors

---

*See also: BACKLOG.md for related items (AMP, OpenApePose/DeepLabCut combine)*
