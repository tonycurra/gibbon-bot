# Quantum-Ready Interface for ApeBrain

> **Status**: Interface defined, stubs implemented, classical fallback working
> **Branch**: `ape_brain`
> **Module**: `brain/quantum.py`

## 🎯 Overview

This document describes the quantum-ready interface for the ApeBrain bioinspired neural architecture. The design allows each brain module to be independently replaced with a quantum equivalent while maintaining full backward compatibility with the existing classical RL training infrastructure.

## 📋 Design Principles

### 1. **Consistent Interface**
Every quantum module maintains the exact same method signatures as its classical counterpart. This allows seamless swapping between classical and quantum implementations.

```python
# Classical
from brain.basal_ganglia import BasalGanglia
bg = BasalGanglia(n_actions=24, n_inputs=88)
action_idx, state = bg.select_action_deterministic(obs, dopamine=0.5)

# Quantum (same interface)
from brain.quantum import QuantumBasalGanglia
q_bg = QuantumBasalGanglia(bg)  # Wraps classical module
action_idx, state = q_bg.select_action_deterministic(obs, dopamine=0.5)
```

### 2. **Graceful Fallback**
All quantum modules gracefully fall back to classical computation when:
- Quantum backend is not available
- Quantum hardware is not accessible
- Running in classical-only mode

### 3. **Hybrid Computation**
Mix classical and quantum modules freely:
```python
brain.modules["basal_ganglia"] = QuantumBasalGanglia(bg)
brain.modules["cerebellum"] = Cerebellum()  # Classical
```

### 4. **JAX Compatibility**
All quantum modules use JAX arrays (`jnp.ndarray`) for:
- Automatic differentiation
- GPU acceleration
- Batch processing
- Just-in-time compilation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ApeBrain (Classical)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Basal Ganglia│ │ Cerebellum  │ │  Amygdala   │ │ Hippocampus │ │
│  │ (Classical) │ │ (Classical) │ │ (Classical) │ │ (Classical) │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     QuantumApeBrain                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Quantum Backend Configuration               │   │
│  │  backend: CLASSICAL | PENNYLANE | QISKIT | CIRQ | JAX_Q   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Quantum BG   │ │Quantum CB   │ │Quantum AM   │ │Quantum HP   │ │
│  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │
│  │ │Classical│◄─┤ │ │Classical│◄─┤ │ │Classical│◄─┤ │ │Classical│◄─┤ │
│  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │
│  │   Fallback  │ │   Fallback  │ │   Fallback  │ │   Fallback  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Module Reference

### Core Classes

#### `QuantumBackend` (Enum)
Supported quantum computing backends:
- `CLASSICAL` - Classical simulation (default)
- `PENNYLANE` - PennyLane (Xanadu)
- `QISKIT` - Qiskit (IBM Quantum)
- `CIRQ` - Cirq (Google Quantum)
- `JAX_QUANTUM` - JAX-based quantum simulation

#### `QuantumConfig` (dataclass)
Configuration for quantum modules:

```python
@dataclass
class QuantumConfig:
    backend: QuantumBackend = QuantumBackend.CLASSICAL
    n_qubits: int = 8              # Number of qubits
    n_shots: int = 1024            # Measurement shots
    n_layers: int = 3              # Circuit layers
    param_scale: float = 0.1      # Parameter initialization scale
    deterministic: bool = False   # Use expected values
    simulate_noise: bool = False   # Simulate hardware noise
    noise_model: str = "depolarizing"
    noise_level: float = 0.01
    optimizer: str = "adam"
    learning_rate: float = 0.01
    device_name: str = "default.qubit"  # PennyLane device
```

#### `QuantumModule` (ABC)
Base class for all quantum modules:

```python
class QuantumModule(ABC):
    backend: QuantumBackend
    config: QuantumConfig
    classical_module: Any
    
    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None)
    
    def reset(self) -> None
    
    def get_classical(self) -> Any  # Get classical counterpart
    def is_quantum(self) -> bool      # Check if using quantum backend
    def set_backend(self, backend: QuantumBackend) -> None
```

### Quantum Circuit Library (`QuantumCircuits`)

All circuits are **STUBS** - currently use classical computation, designed to be replaced with actual quantum circuits.

#### `amplitude_encoding(x, n_qubits) → state`
Encode classical vector into quantum state amplitudes.

**Classical stub**: Normalizes input, pads to 2^n_qubits dimensions.

**Quantum target**: Use `QRAM` or state preparation circuits:
```python
# PennyLane example (future)
import pennylane as qml

def amplitude_encoding(x, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(x, wires=range(n_qubits))
        return qml.state()
    
    return circuit()
```

#### `angle_encoding(x, n_qubits) → angles`
Encode classical vector into rotation angles.

**Classical stub**: Maps input to angles via sigmoid.

**Quantum target**: Use rotation gates:
```python
# PennyLane example (future)
@qml.qnode(dev)
def circuit(x, angles):
    for i in range(n_qubits):
        qml.RY(angles[i], wires=i)
    return qml.state()
```

#### `variational_circuit(x, params, n_qubits, n_layers) → state`
Parameterized quantum circuit for state processing.

**Classical stub**: 2-layer neural network.

**Quantum target**: Variational quantum circuit:
```python
# PennyLane example (future)
@qml.qnode(dev)
def circuit(x, params):
    # Encode input
    for i in range(n_qubits):
        qml.RY(x[i % len(x)], wires=i)
    
    # Variational layers
    for layer in range(n_layers):
        # Entangling gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        # Rotation gates
        for i in range(n_qubits):
            qml.RY(params[layer, i], wires=i)
    
    return qml.state()
```

#### `measure_expectation(state, observable, n_shots, deterministic) → float`
Measure expectation value of an observable.

**Classical stub**: `state^T @ observable @ state`

**Quantum target**:
```python
@qml.qnode(dev)
def circuit(state, observable):
    # Prepare state (assume already prepared)
    return qml.expval(observable)
```

#### `measure_probabilities(state, n_shots) → probs`
Measure probability distribution.

**Classical stub**: Born rule: `|amplitudes|^2`

**Quantum target**:
```python
@qml.qnode(dev)
def circuit(state):
    return qml.probs()
```

### Quantum Modules

Each brain module has a quantum-ready counterpart:

| Classical Module | Quantum Module | Key Quantum Advantage |
|-----------------|----------------|----------------------|
| Neurotransmitters | `QuantumNeurotransmitters` | Interference for modulation |
| Brainstem | `QuantumBrainstem` | Quantum sensors for vestibular |
| Basal Ganglia | `QuantumBasalGanglia` | Grover search: O(√N) vs O(N) |
| Cerebellum | `QuantumCerebellum` | Quantum kernels for error patterns |
| Amygdala | `QuantumAmygdala` | Quantum pattern recognition |
| Hippocampus | `QuantumHippocampus` | QRAM: O(1) vs O(log N) lookup |
| Prefrontal Cortex | `QuantumPrefrontalCortex` | Superposition of goals |
| Motor Cortex | `QuantumMotorCortex` | Quantum control optimization |

#### QuantumBasalGanglia
Special methods:

```python
q_bg = QuantumBasalGanglia(bg_classical)

# Grover-based action search (PLACEHOLDER)
action_idx = q_bg.quantum_action_search(cortical_input, n_actions)
```

**Grover's Algorithm**: Searches N actions in O(√N) time vs O(N) classical.
For 24 actions: ~5 quantum evaluations vs 24 classical.

**Future implementation**:
```python
def quantum_action_search(self, cortical_input, n_actions):
    # 1. Encode all actions as quantum states
    # 2. Prepare oracle that marks "good" actions
    # 3. Apply Grover diffusion
    # 4. Measure to find best action
    pass
```

#### QuantumHippocampus
Special methods:

```python
q_hipp = QuantumHippocampus(hippocampus_classical)

# Grover-based memory search (PLACEHOLDER)
memory = q_hipp.quantum_search_memory(observation)
```

**QRAM (Quantum Random Access Memory)**: O(1) lookup vs O(log N) classical.

**Future implementation**:
```python
def quantum_search_memory(self, observation):
    # 1. Encode query as quantum state
    # 2. Use QRAM to retrieve matching memories
    # 3. Apply amplitude amplification
    # 4. Measure best match
    pass
```

### Factory Function

```python
from brain.quantum import create_quantum_module, QuantumBackend, QuantumConfig

# Create quantum module
q_bg = create_quantum_module(
    "basal_ganglia",
    classical_module=bg_classical,
    config=QuantumConfig(
        backend=QuantumBackend.CLASSICAL,
        n_qubits=8,
        n_shots=1024,
    )
)
```

### QuantumApeBrain

Integrated quantum brain that can switch between backends:

```python
from brain.quantum import QuantumApeBrain, QuantumBackend

# Create with classical backend (default)
q_brain = QuantumApeBrain(
    backend=QuantumBackend.CLASSICAL,
    n_qubits_per_module=8,
    obs_dim=88,
    act_dim=24,
    training=True,
)

# Use like classical brain
action = q_brain.step(observation, reward=reward)

# Switch to quantum backend (when hardware available)
q_brain.set_backend(QuantumBackend.PENNYLANE)

# Get info
info = q_brain.get_info()
# {'step_count': 100, 'backend': 'pennylane', 'modules': [...], ...}

# Access individual modules
q_bg = q_brain.get_module("basal_ganglia")
```

## 🚀 Usage Examples

### Example 1: Classical Only (Default)
```python
from brain.ape_brain import ApeBrain

# This uses classical modules with quantum-ready interface
brain = ApeBrain()
action = brain.step(observation, reward=reward)
```

### Example 2: Mixed Classical and Quantum
```python
from brain.ape_brain import ApeBrain
from brain.quantum import QuantumBasalGanglia, QuantumCerebellum

# Create classical brain
brain = ApeBrain()

# Replace specific modules with quantum versions
q_bg = QuantumBasalGanglia(brain.modules["basal_ganglia"])
q_cb = QuantumCerebellum(brain.modules["cerebellum"])

brain.modules["basal_ganglia"] = q_bg
brain.modules["cerebellum"] = q_cb

# Now these modules use quantum computation
action = brain.step(observation, reward=reward)
```

### Example 3: Full Quantum Brain (Future)
```python
from brain.quantum import QuantumApeBrain, QuantumBackend

# Create quantum brain
q_brain = QuantumApeBrain(
    backend=QuantumBackend.PENNYLANE,  # Or QISKIT, CIRQ
    n_qubits_per_module=8,
)

# Connect to quantum hardware (future)
# q_brain.set_device("ibmq_5_yorktown")  # IBM Quantum

# Use normally
action = q_brain.step(observation, reward=reward)
```

### Example 4: Grover-Accelerated Action Selection
```python
from brain.basal_ganglia import BasalGanglia
from brain.quantum import QuantumBasalGanglia

# Create classical BG
bg = BasalGanglia(n_actions=24, n_inputs=88)

# Wrap with quantum
q_bg = QuantumBasalGanglia(bg)

# Use Grover search (currently falls back to classical)
action_idx = q_bg.quantum_action_search(cortical_input, n_actions=24)

# TODO: When quantum hardware available, this will use Grover's algorithm
# for O(√24) = O(4.9) evaluations instead of O(24)
```

## 🔬 Quantum Algorithms for Brain Functions

### 1. Grover's Algorithm for Action Selection

**Problem**: Search N actions for the one with highest value.

**Classical**: O(N) evaluations

**Quantum**: O(√N) evaluations using Grover's algorithm

**Implementation sketch**:
```python
import pennylane as qml
from pennylane import numpy as np

def grover_action_search(value_oracle, n_actions, n_iterations=None):
    """
    Use Grover's algorithm to find action with highest value.
    
    Args:
        value_oracle: Function that marks actions above threshold
        n_actions: Number of actions (must be power of 2)
        n_iterations: Number of Grover iterations (default: π/4 * √N)
    
    Returns:
        action_idx: Index of best action
    """
    n_qubits = int(np.log2(n_actions))
    
    if n_iterations is None:
        n_iterations = int(np.pi / 4 * np.sqrt(n_actions))
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def grover_circuit():
        # Initialize superposition
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Grover iterations
        for _ in range(n_iterations):
            # Oracle
            value_oracle()
            
            # Diffusion
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.X(wires=i)
            qml.Hadamard(wires=n_qubits-1)
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, n_qubits-1])
            qml.Hadamard(wires=n_qubits-1)
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, n_qubits-1])
            for i in range(n_qubits):
                qml.X(wires=i)
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
        
        return qml.probs()
    
    probs = grover_circuit()
    action_idx = np.argmax(probs)
    return action_idx
```

### 2. QRAM for Hippocampus Memory

**Problem**: Store and retrieve N memories.

**Classical**: O(log N) lookup (binary search)

**Quantum**: O(1) lookup using Quantum RAM

**Reference**: Giovannetti, V., Lloyd, S., & Maccone, L. (2008). "Quantum random access memory"

### 3. Variational Quantum Classifier for Threat Detection

**Problem**: Classify states as threatening or safe.

**Classical**: Neural network classifier

**Quantum**: Variational quantum circuit with trainable parameters

**Reference**: Schulman, J., et al. (2019). "Variational Quantum Eigensolver"

### 4. Quantum Neural Networks for Error Prediction

**Problem**: Predict motor errors from state-action pairs.

**Classical**: 2-layer neural network

**Quantum**: Quantum neural network with parameterized gates

### 5. Quantum Boltzmann Machines for Reward Prediction

**Problem**: Predict rewards from states.

**Classical**: Neural network value function

**Quantum**: Quantum Boltzmann machine for probability distributions

**Reference**: Amin, M. H., et al. (2018). "Quantum Boltzmann Machine"

## 📊 Performance Comparison (Theoretical)

| Function | Classical | Quantum | Speedup | Notes |
|----------|-----------|---------|---------|-------|
| Action Selection (N=24) | O(24) | O(√24) ≈ O(4.9) | ~5x | Grover's algorithm |
| Memory Lookup (N=1000) | O(log 1000) ≈ O(10) | O(1) | ~10x | QRAM |
| Threat Detection (N features) | O(N²) | O(N) | Nx | Quantum kernel |
| Error Prediction | O(N²) | O(N) | Nx | QNN |
| Value Estimation | O(N) | O(log N) | N/log N | QBM |

**Note**: These are theoretical speedups. Actual performance depends on:
- Quantum hardware quality
- Circuit compilation efficiency
- Classical-quantum communication overhead
- Problem-specific constraints

## 🔌 Connecting to Real Quantum Hardware

### PennyLane (Xanadu)

```python
from brain.quantum import QuantumApeBrain, QuantumBackend
import pennylane as qml

# Create quantum brain
q_brain = QuantumApeBrain(backend=QuantumBackend.PENNYLANE)

# Use PennyLane device
dev = qml.device('qiskit.ibmq', wires=8, backend='ibmq_qasm_simulator')
# Or real hardware:
# dev = qml.device('qiskit.ibmq', wires=8, backend='ibmq_5_yorktown')

# TODO: Set device for all quantum modules
# q_brain.set_device(dev)

# Run
action = q_brain.step(observation, reward=reward)
```

### Qiskit (IBM Quantum)

```python
from qiskit import IBMQ, Aer
from brain.quantum import QuantumApeBrain, QuantumBackend

# Load IBMQ account
IBMQ.load_account()

# Create quantum brain
q_brain = QuantumApeBrain(backend=QuantumBackend.QISKIT)

# TODO: Connect to backend
# provider = IBMQ.get_provider('ibm-q')
# backend = provider.get_backend('ibmq_5_yorktown')
# q_brain.set_backend(backend)
```

### Cirq (Google Quantum)

```python
import cirq
from brain.quantum import QuantumApeBrain, QuantumBackend

# Create quantum brain
q_brain = QuantumApeBrain(backend=QuantumBackend.CIRQ)

# TODO: Connect to Sycamore processor
# processor = cirq.get_engine_processor('rainbow')
# q_brain.set_processor(processor)
```

## 🛠️ Implementation Roadmap

### Phase 1: Interface Definition ✅ COMPLETE
- [x] Define `QuantumModule` base class
- [x] Create `QuantumConfig` dataclass
- [x] Define `QuantumBackend` enum
- [x] Create quantum stubs for all 8 modules
- [x] Create `QuantumApeBrain` integrator
- [x] Implement factory function
- [x] Add comprehensive tests

### Phase 2: Quantum Circuit Implementation (TODO)
- [ ] Implement actual amplitude encoding circuits
- [ ] Implement angle encoding circuits
- [ ] Implement variational quantum circuits
- [ ] Implement Grover-based search for Basal Ganglia
- [ ] Implement QRAM for Hippocampus
- [ ] Implement quantum kernels for Cerebellum
- [ ] Implement quantum neural networks for all modules

### Phase 3: Hardware Integration (TODO)
- [ ] PennyLane backend support
- [ ] Qiskit backend support
- [ ] Cirq backend support
- [ ] JAX Quantum backend support
- [ ] Hardware noise modeling
- [ ] Circuit optimization
- [ ] Benchmarking framework

### Phase 4: Hybrid Training (TODO)
- [ ] Hybrid classical-quantum training loops
- [ ] Parameter shift rule for gradients
- [ ] Quantum natural gradient
- [ ] Variational quantum eigensolver integration
- [ ] Quantum approximate optimization algorithm (QAOA)

### Phase 5: Production Deployment (TODO)
- [ ] Connect to IBM Quantum
- [ ] Connect to Rigetti
- [ ] Connect to IonQ
- [ ] Connect to Xanadu
- [ ] Cloud-based quantum inference
- [ ] Local simulator optimization

## 📚 References

### Quantum Machine Learning
1. Biamonte, J., et al. (2017). "Quantum machine learning." Nature.
2. Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature Hilbert spaces." PRX Quantum.
3. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." Nature.

### Quantum Reinforcement Learning
1. Dong, D., et al. (2008). "Quantum algorithm for training a quantum neural network." Physical Review A.
2. Chen, J., et al. (2020). "Quantum reinforcement learning: A comprehensive review." IEEE Access.
3. Jerbi, S., et al. (2021). "Quantum reinforcement learning with variational circuits." arXiv.

### Quantum Algorithms
1. Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search." arXiv:quant-ph/9605043.
2. Shor, P. W. (1994). "Algorithms for quantum computation: Discrete logarithms and factoring." arXiv:quant-ph/9508027.
3. Giovannetti, V., Lloyd, S., & Maccone, L. (2008). "Quantum random access memory." Physical Review Letters.

### Frameworks
1. PennyLane: https://pennylane.ai/
2. Qiskit: https://qiskit.org/
3. Cirq: https://cirq.readthedocs.io/
4. JAX Quantum: https://github.com/google/qsim

## 📝 Changelog

### v0.1.0 (Current)
- Initial quantum interface definition
- All 8 quantum module stubs implemented
- QuantumApeBrain integrator
- QuantumCircuits library (classical stubs)
- Comprehensive documentation
- Full test suite

### v0.2.0 (Planned)
- Actual quantum circuit implementations
- PennyLane backend integration
- Basic quantum training support

### v0.3.0 (Planned)
- Qiskit backend integration
- Cirq backend integration
- Hardware noise simulation

### v1.0.0 (Planned)
- Production-ready quantum modules
- Real quantum hardware integration
- Performance benchmarks
- Hybrid training optimization

## 🎓 How to Contribute

### For Quantum Computing Experts
1. Pick a TODO item from Phase 2 or 3
2. Implement the actual quantum circuit
3. Test with quantum simulators
4. Document the implementation
5. Submit PR to `ape_brain` branch

### For RL Experts
1. Test classical modules with existing RL tasks
2. Identify bottlenecks suitable for quantum speedup
3. Define quantum oracles for specific problems
4. Collaborate with quantum experts on implementation

### For Everyone
1. Report bugs in classical or quantum modules
2. Improve documentation
3. Add more tests
4. Suggest new quantum algorithms for brain functions

## 💬 Support

- **Issues**: Report on Codeberg: https://codeberg.org/tony_olivabot/gibbonBot-RL/issues
- **Discussions**: Start a discussion on Codeberg
- **Quantum Questions**: Ask in quantum computing forums
- **RL Questions**: Ask in reinforcement learning communities

## 📄 License

This quantum interface is part of the GibbonBot-RL project and is licensed under the same terms.

---

**Generated by Mistral Vibe**  
**Co-Authored-By: Mistral Vibe <vibe@mistral.ai>**
