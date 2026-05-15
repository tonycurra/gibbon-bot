# GibbonBot-RL: Quick Command Reference

> **Purpose**: Simple, copy-paste commands for training, visualization, and monitoring

## 🚀 Training

### Start Training
```bash
# Start training (detached background process)
scripts/train_bg.sh start

# Start specific task
scripts/train_bg.sh start stand
scripts/train_bg.sh start walk

# Start in Docker
scripts/train_bg.sh docker stand
scripts/train_bg.sh docker walk
```

### Monitor Training
```bash
# Check if trainer is running + last 20 log lines
scripts/train_bg.sh status

# View last N lines of training log
scripts/train_bg.sh tail 40
scripts/train_bg.sh tail 100

# Follow training log in real-time
tail -f train.log
```

### Stop Training
```bash
# Stop the trainer
scripts/train_bg.sh stop

# Stop Docker trainer
docker compose stop trainer
```

### Full Training Control
```bash
# View all options
scripts/train_bg.sh

# Usage: scripts/train_bg.sh {start [task]|docker [task]|status|tail [N]|stop}
```

---

## 👁️ Visualization

### Live Policy Viewer
```bash
# View robot with trained policy (default: latest checkpoint)
python view.py

# View with specific task
python view.py --task stand
python view.py --task walk

# View with zero actions (robot falls/settles)
python view.py --policy zero

# View with random actions
python view.py --policy random

# View with best checkpoint
python view.py --best

# View checkpoints since specific timestamp
python view.py --since 1715000000
```

### Static Scene Viewer
```bash
# Inspect robot in canonical scene
python look.py

# Inspect with custom scene file
python look.py data/scene.xml
```

---

## 📊 Monitoring & Metrics

### Local Dashboard
```bash
# Start MLflow metrics dashboard
scripts/dashboard.sh start

# Stop dashboard
scripts/dashboard.sh stop

# Check dashboard status
scripts/dashboard.sh status

# Access: http://127.0.0.1:8765
```

### System Status
```bash
# Full status report (GPU, training, sync, checkpoints)
scripts/status.sh

# Quick status (one line)
scripts/status.sh quick
```

### Metrics Comparison
```bash
# Compare recent runs side-by-side
scripts/compare_runs.py

# Export metrics to CSV
scripts/export_metrics.py
```

---

## 🧠 Brain State Management

### Save Brain State
```bash
python -c "
from brain.ape_brain import ApeBrain
from brain.brain_state_persistence import BrainStatePersistence

brain = ApeBrain()
# ... run some steps ...
persistence = BrainStatePersistence('brain_states.db')
persistence.save_complete_state(brain, episode_id=1, metadata={'note': 'first save'})
"
```

### Load Brain State
```bash
python -c "
from brain.ape_brain import ApeBrain
from brain.brain_state_persistence import BrainStatePersistence

brain = ApeBrain()
persistence = BrainStatePersistence('brain_states.db')
persistence.load_complete_state(brain, episode_id=1)
"
```

### List Saved States
```bash
python -c "
from brain.brain_state_persistence import BrainStatePersistence
persistence = BrainStatePersistence('brain_states.db')
states = persistence.list_episodes()
print(states)
"
```

---

## 📁 File Locations

### Important Directories
```
runs/                    # Training runs and checkpoints
  .train.pid             # Current trainer PID
  .dashboard.pid         # Dashboard PID
  .watch_remote.pid      # Remote watcher PID
  ape_<task>_mjx/        # Task-specific checkpoints
data/                   # Scene files and assets
docs/                   # Documentation
brain/                  # Brain modules
scripts/                # Utility scripts
```

### Checkpoints
```
runs/ape_stand_mjx/mjx_*.npz    # Stand task checkpoints
runs/ape_walk_mjx/mjx_*.npz     # Walk task checkpoints
```

---

## 🐳 Docker

### Build & Run
```bash
# Build the Docker image
docker compose build

# Run training in Docker
docker compose run trainer python train_mjx.py --task stand

# View logs
docker compose logs -f trainer

# Stop container
docker compose stop trainer

# Clean up
docker compose down
```

---

## 🔍 Quick Debugging

### Check GPU Usage
```bash
nvidia-smi
```

### Test Brain Modules
```bash
# Test individual brain modules
python -m brain.neurotransmitters
python -m brain.brainstem
python -m brain.basal_ganglia
python -m brain.cerebellum
python -m brain.amygdala
python -m brain.hippocampus
python -m brain.prefrontal_cortex
python -m brain.motor_cortex
```

### Test ApeBrain
```bash
python -c "
from brain.ape_brain import ApeBrain
import jax.numpy as jnp

brain = ApeBrain()
obs = jnp.zeros(88)
action = brain.step(obs, reward=0.0, done=False)
print(f'Action shape: {action.shape}')
"
```

---

## 📚 Documentation

### Open Documentation
```bash
# View main README
cat README.md

# View brain architecture docs
cat docs/APEBRAIN.md

# View biology priors
cat docs/BIOLOGY.md

# View quantum interface
cat docs/QUANTUM.md

# View this commands reference
cat docs/COMMANDS.md
```

---

## ⚡ Common Workflows

### Start training, monitor, then stop
```bash
scripts/train_bg.sh start stand
tail -f train.log          # Monitor in another terminal
# ... wait for training ...
scripts/train_bg.sh stop   # When done
```

### Train in Docker
```bash
scripts/train_bg.sh docker stand
# Logs: docker compose logs -f trainer
# Stop: docker compose stop trainer
```

### View trained policy
```bash
scripts/train_bg.sh status  # Check training progress
python view.py              # View live policy
```

### Compare runs
```bash
scripts/compare_runs.py     # Side-by-side comparison
scripts/status.sh           # Full system status
```

---

**Last updated**: 2026-05-14  
**Branch**: brain-modeling
