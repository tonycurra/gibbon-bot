# OpenApePose Clustering Pipeline

Scripts for clustering and visualizing OpenApePose annotations into gibbon behaviors.

## Overview

The pipeline consists of three modular scripts:

1. **`cluster_openapepose.py`** — Extracts behavioral features from keypoints and clusters annotations using K-means
2. **`visualize_clusters.py`** — Copies sample images from each cluster for visual verification
3. **`plot_clusters.py`** — Creates 2D projections (PCA, t-SNE, UMAP) of clusters for analytical visualization
4. **`pipeline_openapepose.py`** — Wrapper that runs the first two scripts in sequence

## Quick Start

### Full Pipeline (Recommended)

Run clustering and visualization in one command:

```bash
python scripts/pipeline_openapepose.py \
    --annotations /path/to/oap_all.json \
    --images /path/to/OpenApePose/images \
    --n-clusters 3 \
    --samples 5
```

This creates:
- `data/openapepose_clustered/` — Cluster JSON files (`cluster_0.json`, `cluster_1.json`, ...)
- `scripts/data/openapepose_visual_check/` — Sample images organized by cluster

### Step-by-Step

#### 1. Cluster Annotations

```bash
python scripts/cluster_openapepose.py \
    --annotations /path/to/oap_all.json \
    --output data/openapepose_clustered \
    --n-clusters 3 \
    --species gibbon
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--annotations` | *required* | Path to OpenApePose annotations JSON file |
| `--output` | `data/openapepose_clustered` | Output directory for cluster files |
| `--species` | `gibbon` | Species to filter (can specify multiple) |
| `--n-clusters` | `2` | Number of behavior clusters |
| `--seed` | `42` | Random seed for reproducibility |

**Output:** Creates `cluster_0.json`, `cluster_1.json`, etc. in the output directory, plus `metadata.json` with cluster statistics.

#### 2. Visualize Clusters

```bash
python scripts/visualize_clusters.py \
    --clustered data/openapepose_clustered \
    --images /path/to/OpenApePose/images \
    --output scripts/data/openapepose_visual_check \
    --samples 5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--clustered` | *required* | Directory with `cluster_*.json` files from step 1 |
| `--images` | *required* | OpenApePose images directory |
| `--output` | `scripts/data/openapepose_visual_check` | Output directory for copied images |
| `--samples` | `10` | Number of random images to copy per cluster |
| `--seed` | `42` | Random seed for reproducibility |

**Output:** Creates subdirectories per cluster (`cluster_0/`, `cluster_1/`, ...) with sample images for manual inspection.

#### 3. Plot Clusters (Optional)

```bash
python scripts/plot_clusters.py \
    --clustered data/openapepose_clustered \
    --method umap \
    --output plots/openapepose_clusters.png
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--clustered` | *required* | Directory with `cluster_*.json` files |
| `--method` | `umap` | Dimensionality reduction: `pca`, `tsne`, or `umap` |
| `--output` | *none* (shows plot) | Output image path |
| `--perplexity` | `30.0` | t-SNE perplexity |
| `--n-neighbors` | `15` | UMAP n_neighbors |
| `--min-dist` | `0.1` | UMAP min_dist |
| `--seed` | `42` | Random seed |

## Pipeline Wrapper

The `pipeline_openapepose.py` wrapper combines steps 1 and 2:

```bash
python scripts/pipeline_openapepose.py \
    --annotations /path/to/oap_all.json \
    --images /path/to/OpenApePose/images \
    --n-clusters 3 \
    --samples 5 \
    --output data/openapepose_clustered \
    --vis-output scripts/data/openapepose_visual_check
```

**Additional arguments:**

| Argument | Description |
|----------|-------------|
| `--skip-cluster` | Skip clustering, use existing clustered data |
| `--skip-visualize` | Skip visualization step |

## Behavior Features

The clustering uses 10 behavioral features extracted from OpenApePose keypoints:

| Feature | Description |
|---------|-------------|
| `torso_height` | Head y - hip y (standing = tall) |
| `torso_angle` | Angle of line from hip to head (sitting = more horizontal) |
| `foot_spread` | \|left foot x - right foot x\| (walking = wider) |
| `knee_height` | Average knee y - foot y (standing = knees high) |
| `leg_asymmetry` | \|left knee y - right knee y\| (walking = asymmetric) |
| `wrist_height` | Average wrist y - hip y (climbing = arms raised) |
| `elbow_height` | Average elbow y - hip y |
| `arm_asymmetry` | \|left elbow y - right elbow y\| |
| `hand_spread` | \|left wrist x - right wrist x\| |
| `body_symmetry` | Average symmetry across limb pairs |

## File Structure

```
.
├── scripts/
│   ├── cluster_openapepose.py    # Step 1: Clustering
│   ├── visualize_clusters.py      # Step 2: Sample images
│   ├── plot_clusters.py           # Step 3: 2D projections
│   └── pipeline_openapepose.py    # Wrapper for steps 1+2
├── data/
│   └── openapepose_clustered/     # Output: cluster_*.json files
└── scripts/data/
    └── openapepose_visual_check/   # Output: sample images by cluster
```

## Notes

- Coordinates are normalized to [0,1] using the bounding box before feature extraction
- Only annotations with ≥8 visible keypoints are used for clustering
- The `--seed` parameter ensures reproducible results across runs
