# gibbonBot-RL

A reinforcement learning framework for training a 4-limb robotic ape.

## Quick Start

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Simple training (recommended)
python scripts/train_simple.py --steps 50000

# Unified training (considers past data)
python scripts/train_unified.py --steps 100000

# GPU training
python scripts/train.py stand
```

## Training Options

### 1. Simple Training (Recommended)
```bash
python scripts/train_simple.py --steps 100000
```
- Direct height rewards
- Positive reinforcement
- Fast convergence

### 2. Unified Training
```bash
python scripts/train_unified.py --steps 100000
```
- Merges all past training data
- Considers best models automatically
- Saves to unified location

### 3. GPU Training
```bash
python scripts/train.py stand
```
- GPU acceleration (MJX)
- 512 parallel environments
- Automatic viewer

### 4. Evaluation
```bash
python scripts/evaluate_training.py --task stand
```
- Real-time progress monitoring
- Stagnation detection
- Performance metrics

## Data Management

All training data is saved in `runs/`:
- `runs/simple_stand/` - Simple training
- `runs/unified_stand/` - Unified training
- `runs/ape_stand/` - CPU training
- `runs/ape_stand_mjx/` - GPU training

Best models are preserved in `best/` subdirectories.

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for full documentation.
