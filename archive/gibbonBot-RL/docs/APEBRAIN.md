# ApeBrain - Bioinspired Neural Architecture

> **Status**: Under development (see `brain/` module)
> **Branch**: `ape_brain`
> **Author**: Mistral Vibe + Tony

ApeBrain is a biologically-plausible neural architecture for reinforcement learning,
inspired by ape and human neuroanatomy. It replaces the standard monolithic neural
network policy with a modular, brain-like system that includes neurotransmitter
modulation, reflexes, action selection, and memory systems.

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEOCORTEX (Planning Layer)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Prefrontal Cortex (PFC)                             │   │
│  │  - Goal setting: "Stand on ankles, maintain symmetry"                 │   │
│  │  - Working memory: current pose, terrain assessment                    │   │
│  │  - Output: High-level intent → Basal Ganglia                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Motor Cortex (M1)                                  │   │
│  │  - Executes selected motor program                                   │   │
│  │  - Receives from: Basal Ganglia + Cerebellum                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SUBORTICAL STRUCTURES                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │  BASAL GANGLIA    │    │   CEREBELLUM      │    │    AMYGDALA      │    │
│  │  (Action Selection)│    │  (Error Correction)│    │  (Threat/Fear)   │    │
│  │  - D1 MSNs: Go    │    │  - Climbing fibers:│    │  - Punishment:   │    │
│  │  - D2 MSNs: No-go │    │    error signals   │    │    flip penalty  │    │
│  │  - TD learning    │    │  - Purkinje:      │    │  - Fear memory:  │    │
│  │                    │    │    motor correction│    │    unstable poses│    │
│  └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘    │
└────────────┼────────────────────────┼─────────────────────────┼────────────┘
             │                          │                           │
             └──────────────┬───────────┘                           │
                            │                                       │
                            ▼                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BRAINSTEM & SPINAL CORD                            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │   Vestibular      │    │   Reticular       │    │   Spinal Cord    │    │
│  │   System          │    │   Formation       │    │   (Reflexes)      │    │
│  │  - Balance        │    │  - Arousal        │    │  - Withdrawal    │    │
│  │  - Orientation    │    │  - Attention       │    │  - Stretch       │    │
│  │  - Head position  │    │  - Sleep/wake     │    │  - Joint limits  │    │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEUROTRANSMITTER SYSTEMS                            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │   DOPAMINE        │    │   SEROTONIN       │    │   GABA           │    │
│  │   (Reward)        │    │   (Patience)      │    │   (Inhibition)   │    │
│  │  - RPE: δ = r +    │    │  - Long-term      │    │  - Stabilizes    │    │
│  │    γ*V(s') - V(s)  │    │    planning       │    │    networks     │    │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Module Structure

```
brain/
├── __init__.py              # Lazy imports, module exports
├── neurotransmitters.py     # Dopamine, Serotonin, GABA, Noradrenaline, Acetylcholine
├── brainstem.py             # Vestibular, Reflexes, Attention, Arousal
├── basal_ganglia.py         # D1/D2 MSNs, Action Selection, TD Learning
├── cerebellum.py            # Motor Error Correction (TODO)
├── hippocampus.py            # Episodic Memory (TODO)
├── amygdala.py              # Fear/Punishment System (TODO)
├── prefrontal_cortex.py      # High-Level Planning (TODO)
├── motor_cortex.py          # Motor Command Execution (TODO)
└── ape_brain.py             # Complete Brain Integration (TODO)
```

## 🧬 Implemented Modules

### 1. Neurotransmitters (`brain/neurotransmitters.py`)

**Biological Role**: Modulate synaptic plasticity, learning rate, exploration, and reward perception.

**Implemented Systems**:

| Neurotransmitter | Role | Range | Modulates |
|-----------------|------|-------|-----------|
| Dopamine | Reward Prediction Error (RPE) | -1 to +1 | Learning rate, exploration, reward perception |
| Serotonin | Patience / Risk Aversion | 0.1 to 2.0 | Long-term planning, impulsivity |
| GABA | Inhibition / Stabilization | 0.1 to 0.9 | Network stability, prevents runaway excitation |
| Noradrenaline | Arousal / Novelty Detection | 0 to 1 | Attention, memory consolidation |
| Acetylcholine | Learning Rate Modulation | 0.1 to 2.0 | Synaptic plasticity, memory |

**Key Methods**:
- `update(reward, predicted_reward, state, done)` - Update all neurotransmitter levels
- `get_learning_rate()` - Adaptive learning rate based on dopamine + acetylcholine
- `get_exploration_rate()` - Adaptive exploration based on dopamine + noradrenaline
- `get_reward_modulation()` - Reward scaling based on dopamine

**Example**:
```python
from brain import Neurotransmitters
import jax.numpy as jnp

neuro = Neurotransmitters()
state = jnp.zeros(88)

# Update with reward
neuro.update(reward=10.0, predicted_reward=0.0, state=state)
print(f"Dopamine: {neuro.state.dopamine}")  # ~1.0 (positive RPE)
print(f"Learning Rate: {neuro.get_learning_rate()}")
```

### 2. Brainstem (`brain/brainstem.py`)

**Biological Role**: Low-level reflexes, balance, and sensory preprocessing.

**Components**:

#### Vestibular System
- Extracts roll, pitch, yaw from torso quaternion
- Estimates gravity direction in body frame
- Detects balance state (upright vs tilted)
- Computes balance confidence (0-1)

#### Spinal Cord Reflexes
- **Withdrawal Reflex**: Pull away from pain/nociceptive input
- **Stretch Reflex**: Resist rapid joint stretching
- **Joint Limit Protection**: Push away from joint limits
- **Pain Signal Processing**: Nociceptive input handling

#### Reticular Formation
- **Attention Filtering**: Prioritizes novel sensory inputs
- **Arousal Regulation**: Modulates wakefulness level (0-1)
- **Novelty Detection**: Tracks state frequency for noradrenaline triggering

**Key Methods**:
- `process_vestibular(torso_quat, torso_pos, torso_vel, torso_ang_vel)` - Balance processing
- `process_reflexes(joint_positions, joint_velocities, joint_torques, pain_input)` - Reflex generation
- `apply_reflex_correction(action)` - Apply reflex corrections to motor commands
- `update_attention(sensory_input, novelty)` - Attention filtering
- `preprocess_observation(...)` - Complete brainstem preprocessing pipeline

**Example**:
```python
from brain import Brainstem
import jax.numpy as jnp

brainstem = Brainstem(n_joints=24, n_obs=88)

# Upright torso
vest = brainstem.process_vestibular(
    torso_quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
    torso_pos=jnp.array([0.0, 0.0, 0.5]),
    torso_vel=jnp.zeros(3),
    torso_ang_vel=jnp.zeros(3)
)
print(f"Balanced: {vest.is_balanced}")  # True
print(f"Confidence: {vest.balance_confidence}")  # ~1.0
```

### 3. Basal Ganglia (`brain/basal_ganglia.py`)

**Biological Role**: Action selection, habit formation, reinforcement learning.

**Components**:

#### Pathways
- **D1 MSNs (Direct/Go Pathway)**: Dopamine excites → promotes action
- **D2 MSNs (Indirect/No-Go Pathway)**: Dopamine inhibits → suppresses action
- **Thalamus Output**: D1 - D2 = center-surround selection

#### Learning
- **TD Learning**: Temporal difference learning for action values
- **Dopamine Modulation**: Dopamine scales learning rate
- **Hebbian Learning**: Cortical weights updated proportionally to activation

**Key Methods**:
- `process_cortical_input(cortical_input, dopamine)` - Process input through BG pathways
- `select_action_deterministic(cortical_input, dopamine)` - Deterministic action selection
- `learn(reward, next_cortical_input, selected_action)` - TD learning update
- `get_action_probs(cortical_input, dopamine)` - Get action probability distribution
- `set_n_actions(n_actions)` - Dynamically change action space

**Example**:
```python
from brain import BasalGanglia
import jax.numpy as jnp

bg = BasalGanglia(n_actions=4, n_inputs=64)

# Set weights to favor action 2
bg.d1_weights = bg.d1_weights.at[0, 2].set(2.0)

# Select action
cortical_input = jnp.zeros(64).at[0].set(1.0)
action, state = bg.select_action_deterministic(cortical_input, dopamine=0.0)
print(f"Selected action: {action}")

# Learn from reward
bg.learn(reward=10.0, next_cortical_input=cortical_input, selected_action=action)
print(f"Action values: {bg.action_values}")
```

## 🧩 TODO Modules

| Module | Status | Description |
|--------|--------|-------------|
| Cerebellum | ⏳ TODO | Motor error correction, fine-tuning |
| Hippocampus | ⏳ TODO | Episodic memory storage and retrieval |
| Amygdala | ⏳ TODO | Fear conditioning, threat detection |
| Prefrontal Cortex | ⏳ TODO | High-level planning, working memory |
| Motor Cortex | ⏳ TODO | Motor command execution |
| ApeBrain (Integrator) | ⏳ TODO | Complete brain integration |

## 🎯 Design Principles

### Biological Plausibility
Each module is designed to mimic its biological counterpart:
- **Neurotransmitters**: Realistic firing patterns and modulation effects
- **Brainstem**: Reflex arcs, vestibular processing, attention filtering
- **Basal Ganglia**: D1/D2 pathway competition, dopamine modulation

### Modularity
- Each module can be developed and tested independently
- Modules communicate via well-defined interfaces
- Easy to swap out individual components (e.g., replace with quantum equivalents)

### JAX Compatibility
- All modules use JAX for GPU acceleration
- Functional programming style where possible
- Support for JIT compilation

### Quantum-Ready
The architecture is designed with quantum computing in mind:
- Modular components can be replaced with quantum circuits
- Probabilistic representations (softmax, sampling)
- Amplitude-encoding friendly

**Quantum Integration Path**:

| Classical Component | Quantum Equivalent | Status |
|---------------------|-------------------|--------|
| Dense Layers | Quantum Neural Network (QNN) | ⏳ |
| Activation Functions | Quantum Activation Circuits | ⏳ |
| Attention | Quantum Attention | ⏳ |
| Memory (Hippocampus) | Quantum RAM (QRAM) | ⏳ |
| Sampling | Quantum Measurement | ⏳ |

## 🚀 Usage

### Training with ApeBrain

```python
from brain import ApeBrain
import jax.numpy as jnp

# Initialize brain
brain = ApeBrain(obs_dim=88, act_dim=24, n_poses=4)

# Environment step
obs = env.reset()
for _ in range(1000):
    # Step through brain
    action = brain.step(obs)
    
    # Environment step
    next_obs, reward, done, info = env.step(action)
    
    # Learn from experience
    brain.learn(obs, action, reward, next_obs, done)
    
    obs = next_obs
    if done:
        obs = env.reset()
```

### Connecting to Existing Training

The ApeBrain can be integrated with the existing PPO trainer:

```python
# In train_mjx.py or similar
def make_actor_critic(act_dim):
    # Option 1: Use ApeBrain as policy
    from brain import ApeBrain
    brain = ApeBrain(obs_dim=88, act_dim=act_dim)
    
    class BrainPolicy(nn.Module):
        def __call__(self, obs):
            # Preprocess with brain
            action = brain.step(obs)
            return action, log_prob, value
    
    return BrainPolicy()
```

## 🔬 Testing

Each module has comprehensive self-tests:

```bash
# Test individual modules
python -m brain.neurotransmitters
python -m brain.brainstem
python -m brain.basal_ganglia

# Run all tests
python -m pytest tests_brain/ -v
```

## 📊 Performance Characteristics

| Module | Computational Complexity | Memory | GPU Compatible |
|--------|------------------------|--------|----------------|
| Neurotransmitters | O(1) | O(1) | ✅ |
| Brainstem | O(n_joints + n_obs) | O(n_joints + n_obs) | ✅ |
| Basal Ganglia | O(n_actions × n_inputs) | O(n_actions × n_inputs) | ✅ |

## 📊 Current Status (Live Tracker)

### ✅ Completed (Deployed & Tested)

| Module | File | Status | Tests | LOC | Commit |
|--------|------|--------|-------|-----|--------|
| Neurotransmitters | `brain/neurotransmitters.py` | ✅ **Done** | ✅ All pass | 448 | 423d329 |
| Brainstem | `brain/brainstem.py` | ✅ **Done** | ✅ All pass | 575 | fb9c3dc |
| Basal Ganglia | `brain/basal_ganglia.py` | ✅ **Done** | ✅ All pass | 520 | 437b85d |
| Documentation | `docs/APEBRAIN.md` | ✅ **Done** | - | 500+ | 367ce4c |
| Stand Task Upgrade | `envs/params_stand_mjx.py` | ✅ **Done** | ✅ Working | 67 | 9a00450 |

### 🚧 In Development (Next Priority)

| Module | File | Status | Priority | Dependencies |
|--------|------|--------|----------|--------------|
| Cerebellum | `brain/cerebellum.py` | ⏳ TODO | **HIGH** | None |
| Amygdala | `brain/amygdala.py` | ⏳ TODO | **HIGH** | None |
| Hippocampus | `brain/hippocampus.py` | ⏳ TODO | **HIGH** | None |
| Prefrontal Cortex | `brain/prefrontal_cortex.py` | ⏳ TODO | **HIGH** | Neurotransmitters |
| Motor Cortex | `brain/motor_cortex.py` | ⏳ TODO | **HIGH** | Brainstem |
| ApeBrain Integrator | `brain/ape_brain.py` | ⏳ TODO | **HIGH** | All above |

### 📋 Backlog

| Module | Status | Priority | Notes |
|--------|--------|----------|-------|
| Integration with train_mjx.py | ⏳ TODO | Medium | Connect to existing PPO trainer |
| Quantum interface stubs | ⏳ TODO | Medium | Quantum-ready abstractions |
| Training scripts for poses | ⏳ TODO | Medium | Multi-pose curriculum |
| Full brain validation | ⏳ TODO | Medium | End-to-end testing |
| Biological validation | ⏳ TODO | Low | Compare with neuroscience |
| Performance benchmarking | ⏳ TODO | Low | Compare with standard RL |

---

## 🎯 Implementation Plan (Next Steps)

### Phase 1: Complete Core Modules (This Week)

#### Step 1: Cerebellum Module
- **Purpose**: Motor error correction and fine-tuning
- **Biological analogy**: Purkinje cells, climbing fibers, inferior olive
- **Function**: Predicts motor errors, applies corrections to actions
- **Inputs**: State, action, next_state
- **Outputs**: Error prediction, corrected action
- **Status**: ⏳ NOT STARTED

#### Step 2: Amygdala Module  
- **Purpose**: Fear conditioning and threat detection
- **Biological analogy**: Fear memory, punishment system
- **Function**: Detects dangerous states, stores fear memories, modulates avoidance
- **Inputs**: State, punishment signals
- **Outputs**: Fear level, avoidance bias
- **Status**: ⏳ NOT STARTED

#### Step 3: Hippocampus Module
- **Purpose**: Episodic memory storage and retrieval
- **Biological analogy**: Place cells, memory consolidation
- **Function**: Stores successful episodes, retrieves similar past experiences
- **Inputs**: State, action, reward, context
- **Outputs**: Retrieved memories, similarity scores
- **Status**: ⏳ NOT STARTED

#### Step 4: Prefrontal Cortex Module
- **Purpose**: High-level planning and working memory
- **Biological analogy**: Goal setting, executive function
- **Function**: Sets goals, maintains working memory, selects motor programs
- **Inputs**: Task description, current state
- **Outputs**: High-level intent, cortical input to Basal Ganglia
- **Status**: ⏳ NOT STARTED

#### Step 5: Motor Cortex Module
- **Purpose**: Motor command execution
- **Biological analogy**: Primary motor cortex (M1)
- **Function**: Executes selected motor program, interfaces with muscles
- **Inputs**: Intent from PFC, corrected action from Cerebellum
- **Outputs**: Final motor commands
- **Status**: ⏳ NOT STARTED

#### Step 6: ApeBrain Integrator
- **Purpose**: Complete brain integration
- **Function**: Orchestrates all modules, handles data flow
- **Inputs**: Observation from environment
- **Outputs**: Action to environment
- **Status**: ⏳ NOT STARTED

### Phase 2: Integration & Testing

#### Step 7: Unit Tests for New Modules
- Each module gets comprehensive self-tests
- Test biological plausibility
- Test numerical stability
- Test JAX compatibility

#### Step 8: Integration Tests
- Test module interactions
- Test complete brain pipeline
- Test with MuJoCo environment

#### Step 9: Connect to Training Loop
- Integrate with `train_mjx.py`
- Option: Use ApeBrain as policy
- Option: Use ApeBrain alongside existing policy
- Maintain backward compatibility

### Phase 3: Multi-Pose Learning

#### Step 10: Pose Curriculum
- Define pose space: upright, crouched, left_lean, right_lean, etc.
- Create training curriculum
- Train brain to handle multiple poses

#### Step 11: Pose Memory
- Store learned poses in Hippocampus
- Retrieve poses based on context
- Switch between poses dynamically

### Phase 4: Quantum Readiness

#### Step 12: Quantum Interface
- Add quantum-ready abstractions
- Create stubs for quantum components
- Design for hybrid classical-quantum training

---

## 📅 Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 (Core) | 1-2 weeks | Cerebellum, Amygdala, Hippocampus, PFC, Motor Cortex, ApeBrain |
| Phase 2 (Integration) | 1 week | Unit tests, integration tests, training connection |
| Phase 3 (Multi-Pose) | 1 week | Pose curriculum, pose memory, dynamic switching |
| Phase 4 (Quantum) | 2 weeks | Quantum interface, hybrid training |
| **Total** | **5-6 weeks** | Complete bioinspired brain |

---

## 📚 References

### Neuroscience
- **Dopamine and Reward**: Schultz, W. (2002). "Getting Formal with Dopamine and Reward"
- **Basal Ganglia**: Mink, J. W. (1996). "The Basal Ganglia: Focused Selection and Inhibition of Competing Motor Programs"
- **Cerebellum**: Eccles, J. C. (1967). "The Cerebellum as a Neuronal Machine"
- **Hippocampus**: O'Keefe, J., & Nadel, L. (1978). "The Hippocampus as a Cognitive Map"
- **Amygdala**: LeDoux, J. E. (2000). "Emotion Circuits in the Brain"

### Reinforcement Learning
- **TD Learning**: Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
- **Actor-Critic**: Konda, V. R., & Tsitsiklis, J. N. (2000). "Actor-Critic Algorithms"
- **PPO**: Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"

### Quantum Machine Learning
- **QNN**: Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces"
- **Quantum RL**: Dong, D., et al. (2008). "Quantum Algorithm for Training Binary Perceptrons on Quantum Computers"

## 🤝 Contributing

1. **Branch**: Create a feature branch from `ape_brain`
2. **Develop**: Implement a module or feature
3. **Test**: Add comprehensive tests
4. **Document**: Update documentation
5. **Commit**: Follow conventional commits
6. **Push**: Push to your fork and create a PR

### Module Template

```python
"""
Module Name - Brief Description

Biological Background:
    [Explain the biological system]

Implementation:
    [Explain how it's implemented]
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class ModuleState:
    """State of the module."""
    pass

class Module:
    """Module implementation."""
    
    def __init__(self, ...):
        """Initialize module."""
        pass
    
    def step(...):
        """Process one step."""
        pass
    
    def learn(...):
        """Learn from experience."""
        pass
    
    def reset(self):
        """Reset module state."""
        pass

def test_module():
    """Self-tests for the module."""
    pass

if __name__ == "__main__":
    test_module()
```

## 📝 Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-11 | 0.1.0 | Initial implementation: Neurotransmitters, Brainstem, Basal Ganglia |

## 🎉 Roadmap

### Phase 1: Core Brain (Current)
- [x] Neurotransmitters
- [x] Brainstem
- [x] Basal Ganglia
- [ ] Cerebellum
- [ ] Hippocampus
- [ ] Amygdala

### Phase 2: Cortex
- [ ] Prefrontal Cortex
- [ ] Motor Cortex
- [ ] Sensory Cortex

### Phase 3: Integration
- [ ] ApeBrain integrator
- [ ] Training loop integration
- [ ] Multi-pose learning

### Phase 4: Quantum
- [ ] Quantum circuit stubs
- [ ] Hybrid classical-quantum training
- [ ] Full quantum implementation

### Phase 5: Validation
- [ ] Biological validation
- [ ] Performance benchmarking
- [ ] Comparison with standard RL

---

**Maintainer**: Tony + Mistral Vibe  
**License**: MIT  
**Branch**: `ape_brain`  
**Status**: Active Development
