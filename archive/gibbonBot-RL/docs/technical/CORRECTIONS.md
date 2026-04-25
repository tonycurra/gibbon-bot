# ⚠️ Important Corrections

## GPU Training Limitations

**Issue**: `train_mjx.py` does NOT support the `reach` task (tree-seeking) yet.

**Current Status**:
- ✅ `train_mjx.py` supports **stand** task only
- ❌ `train_mjx.py` does NOT support **reach** task
- ✅ `train.py` supports both **stand** and **reach** tasks

## Correct Commands

### For Tree-Seeking (Reach Task)

**CPU Training (✅ Working)**:
```bash
python train.py --task reach --watch --envs 2
```

**GPU Training (❌ Not Available)**:
```bash
# This will FAIL - MJX version not implemented yet
python train_mjx.py --task reach
```

### For Standing Task

**CPU Training**:
```bash
python train.py --task stand --watch --envs 2
```

**GPU Training (✅ Working)**:
```bash
python train_mjx.py --envs 1024
```

## Why This Happened

The `train_mjx.py` script is hardcoded to use `stand_env_mjx.py`:

```python
# From train_mjx.py line 27-32
from envs.stand_env_mjx import (
    ACT_DIM,
    MAX_STEPS,
    OBS_DIM,
    load_model,
    reset,
    step,
)
```

There is no `ape_env_mjx.py` file yet for GPU-accelerated tree-seeking.

## Workaround

Use CPU training for tree-seeking:
```bash
python train.py --task reach --watch --envs 2
```

## Future Plan

To enable GPU training for reach task, we need to:
1. Create `envs/ape_env_mjx.py` (MJX version of tree-seeking)
2. Update `train_mjx.py` to support task selection
3. Add `--task` argument to `train_mjx.py`

## Updated Documentation

All documentation has been corrected to reflect this limitation:
- [`CHEATSHEET.md`](CHEATSHEET.md)
- [`QUICKSTART.md`](QUICKSTART.md)
- [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md)

## Quick Reference

| Task | CPU Training | GPU Training |
|------|-------------|--------------|
| **reach** (tree-seeking) | ✅ `train.py --task reach` | ❌ Not available |
| **stand** (standing) | ✅ `train.py --task stand` | ✅ `train_mjx.py` |

**Use CPU for tree-seeking, GPU for standing only.**

## Error You Saw

```bash
$ python train_mjx.py --task reach
Failed to import warp: No module named 'warp'
usage: train_mjx.py [-h] [--envs ENVS] [--inherit] [--watch]
train_mjx.py: error: unrecognized arguments: --task reach
```

**This is expected** - the script doesn't accept `--task` argument yet.

## Correct Usage

```bash
# For tree-seeking (CPU only)
python train.py --task reach --watch --envs 2

# For standing (CPU)
python train.py --task stand --watch --envs 2

# For standing (GPU)
python train_mjx.py --envs 1024
```

## Documentation Updates

- Removed all references to `train_mjx.py --task reach`
- Added clear warnings about GPU limitations
- Provided CPU alternatives for tree-seeking
- Updated cheatsheet and quickstart guides

## Summary

**Use CPU training for tree-seeking tasks** until the MJX version is implemented. The optimized CPU version still provides excellent performance with the new reward system, curriculum learning, and adaptive physics.