# gibbonBot-RL

A reinforcement learning framework for training a 4-limb robotic ape.

## Quick Start

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Development/testing (fast iteration)
python scripts/train_simple.py --steps 50000 --no-inherit

# Production training (recommended for final models)
python scripts/train_fixed.py --task stand --envs 8 --inherit

# End-user interface (automatic CPU/GPU detection)
python scripts/train.py stand --cpu --no-watch
```

**See our comprehensive training evaluation for detailed recommendations:**
- [`test/TRAINING_EVALUATION_REPORT.md`](test/TRAINING_EVALUATION_REPORT.md)
- Includes performance benchmarks, feature comparisons, and use case recommendations

## Training Options

### 1. Simple Training (Best for Development)
```bash
python scripts/train_simple.py --steps 100000
```
- ✅ **Best for**: Rapid development, debugging, quick testing
- Direct height rewards with simplified structure
- Step control for controlled experiments
- Fastest for iteration (43s for 50K steps)
- Limited checkpointing (use for testing, not production)

### 2. Fixed Training (Best for Production)
```bash
python scripts/train_fixed.py --task stand --envs 8
```
- ✅ **Best for**: Production training, final models, long runs
- Full checkpointing and lineage tracking
- Inheritance from past training (`--inherit`)
- Evaluation callbacks and monitoring
- Designed for 10M+ step training runs

### 3. Unified Training (Best for End Users)
```bash
python scripts/train.py stand --cpu --no-watch
```
- ✅ **Best for**: End-user interface, automatic mode detection
- Unified interface for CPU/GPU training
- Automatic GPU detection and fallback
- Wrapper around other training systems
- Integrated viewer support

### 4. GPU Training (Best for Speed)
```bash
python src/gibbonbot/config/train_mjx.py --envs 1024 --inherit
```
- ✅ **Best for**: Maximum speed with GPU acceleration
- MJX backend with massive parallelization
- 1024+ parallel environments
- Requires GPU and additional dependencies
- 50-100x speedup over CPU training

### 4. Evaluation
```bash
python scripts/evaluate_training.py --task stand
```
- Real-time progress monitoring
- Stagnation detection
- Performance metrics

## Training Script Comparison

| Script | Best For | Speed | Features | Checkpointing | Inheritance |
|--------|----------|-------|----------|--------------|-------------|
| `train_simple.py` | Development | ⭐⭐⭐⭐⭐ | Basic | Limited | Basic |
| `train_fixed.py` | Production | ⭐⭐⭐ | Advanced | Full | Advanced |
| `train.py` | End Users | ⭐⭐⭐ | Unified | Full | Advanced |
| `train_mjx.py` | GPU Speed | ⭐⭐⭐⭐⭐ | GPU | Full | Advanced |

**See our comprehensive evaluation for detailed benchmarks:** [`test/TRAINING_EVALUATION_REPORT.md`](test/TRAINING_EVALUATION_REPORT.md)

## Data Management

All training data is saved in `runs/`:
- `runs/simple_stand/` - Simple training
- `runs/unified_stand/` - Unified training
- `runs/ape_stand/` - CPU training
- `runs/ape_stand_mjx/` - GPU training

Best models are preserved in `best/` subdirectories.

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for full documentation.
