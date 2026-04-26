
📊 Evaluation Report
===================
Generated: Sun Apr 26 19:00:27 2026
Test duration: 50,000 steps per script

### train_simple
**Description**: Simplified training with direct rewards
**Command**: python scripts/train_simple.py --steps 50000 --no-inherit
**Status**: ✅ COMPLETED
**Training Time**: 43.0s
**Errors Detected**: 3

### train_fixed
**Description**: Fixed training with inheritance options
**Command**: python scripts/train_fixed.py --task stand --envs 1
**Status**: ❌ FAILED - Timeout

### train_unified
**Description**: Unified training script (CPU mode)
**Command**: python scripts/train.py stand --cpu --no-watch
**Status**: ❌ FAILED - Timeout

## Comparative Analysis

### Performance Comparison
**Fastest**: train_simple (43.0s)
**Slowest**: train_simple (43.0s)

### Recommendations
- For fastest training: Use train_simple
