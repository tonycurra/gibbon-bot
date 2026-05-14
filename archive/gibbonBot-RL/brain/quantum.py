"""
Quantum-Ready Interface for ApeBrain

This module provides quantum-compatible abstractions and stubs that allow
the bioinspired brain modules to be replaced with quantum equivalents
in the future.

Design Principles:
==================

1. **Quantum-Classical Interface**: Each quantum module should have the same
   interface as its classical counterpart (same method signatures, same inputs/outputs)

2. **Fallback to Classical**: Quantum modules should gracefully fall back to
   classical computation when quantum hardware is not available

3. **Hybrid Computation**: Support mixing classical and quantum components

4. **State Management**: Quantum state must be properly managed across steps

5. **Measurement**: Quantum measurements are stochastic - provide options
   for deterministic vs probabilistic behavior

Current Status:
===============

This module provides:
- QuantumModule base class with common quantum operations
- Quantum-ready stubs for each brain module
- Circuit definitions for key operations (using PennyLane-style API)
- Documentation for future quantum implementation

The actual quantum circuits are STUBS/PLACEHOLDERS that currently use
classical computation. They are designed to be replaced with real quantum
circuits when quantum hardware becomes available.

Future Work:
============

1. Replace stub circuits with actual quantum circuits
2. Add quantum circuit optimization and compilation
3. Implement quantum state persistence across steps
4. Add noise modeling for quantum hardware simulation
5. Benchmark quantum vs classical performance
6. Connect to real quantum hardware (IBM Q, Rigetti, IonQ, etc.)

Usage:
======

# Use quantum-ready modules
from brain.quantum import QuantumNeurotransmitters, QuantumBasalGanglia

# Or use the factory to create quantum versions
from brain.quantum import create_quantum_module

q_bg = create_quantum_module("basal_ganglia", n_qubits=8)
action = q_bg.step(observation, reward=1.0)

# Or use the quantum brain
from brain.quantum import QuantumApeBrain
q_brain = QuantumApeBrain(n_qubits_per_module=8)
action = q_brain.step(observation, reward=reward)

References:
===========
- PennyLane: https://pennylane.ai/
- JAX Quantum: https://github.com/google/qsim
- Cirq: https://cirq.readthedocs.io/
- Qiskit: https://qiskit.org/
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp


class QuantumBackend(Enum):
    """Supported quantum computing backends."""

    CLASSICAL = "classical"  # Classical simulation (default)
    PENNYLANE = "pennylane"  # PennyLane quantum computing
    QISKIT = "qiskit"  # Qiskit (IBM Quantum)
    CIRQ = "cirq"  # Cirq (Google Quantum)
    JAX_QUANTUM = "jax_quantum"  # JAX-based quantum simulation


@dataclass
class QuantumConfig:
    """Configuration for quantum modules."""

    # Backend selection
    backend: QuantumBackend = QuantumBackend.CLASSICAL

    # Quantum resource settings
    n_qubits: int = 8  # Number of qubits to use
    n_shots: int = 1024  # Number of measurements (shots) for probabilistic circuits

    # Circuit parameters
    n_layers: int = 3  # Number of layers in variational circuits
    param_scale: float = 0.1  # Scale for parameter initialization

    # Execution
    deterministic: bool = False  # If True, use expected values instead of sampling
    use_mid_circuit_measurement: bool = False  # Allow mid-circuit measurements

    # Hardware simulation
    simulate_noise: bool = False  # Simulate quantum noise
    noise_model: str = "depolarizing"  # Type of noise to simulate
    noise_level: float = 0.01  # Noise probability

    # Optimization
    optimizer: str = "adam"  # Classical optimizer for hybrid training
    learning_rate: float = 0.01

    # Quantum-specific
    device_name: str = "default.qubit"  # PennyLane device name

    def __post_init__(self):
        """Validate configuration."""
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if self.n_shots < 1:
            raise ValueError("n_shots must be at least 1")
        if not 0 <= self.noise_level <= 1:
            raise ValueError("noise_level must be in [0, 1]")


@dataclass
class QuantumState:
    """State of a quantum module."""

    # Quantum circuit parameters (trainable)
    params: jnp.ndarray = field(default_factory=lambda: jnp.array([]))

    # Classical state (non-quantum)
    classical_state: Dict[str, Any] = field(default_factory=dict)

    # Measurement history
    measurement_history: List[jnp.ndarray] = field(default_factory=list)

    # Step counter
    step_count: int = 0

    # Quantum-specific state
    qubit_states: Optional[jnp.ndarray] = None  # Simulated qubit states
    entanglement_map: Optional[jnp.ndarray] = None  # Which qubits are entangled


class QuantumModule(ABC):
    """
    Base class for quantum-ready brain modules.

    All quantum modules should inherit from this class to ensure
    a consistent interface with classical modules.

    Key Requirements:
    ================
    1. All inputs and outputs must be JAX arrays (jnp.ndarray)
    2. Must support both classical and quantum backends
    3. Must be differentiable for training
    4. Must handle state properly across steps
    """

    backend: QuantumBackend
    config: QuantumConfig
    classical_module: Any  # Reference to classical counterpart

    @abstractmethod
    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        """
        Initialize quantum module.

        Args:
            classical_module: The classical counterpart module
            config: Quantum configuration
        """
        self.classical_module = classical_module
        self.config = config if config is not None else QuantumConfig()
        self.backend = self.config.backend
        self._initialize_quantum()

    @abstractmethod
    def _initialize_quantum(self) -> None:
        """Initialize quantum-specific resources."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the module state."""
        pass

    def get_classical(self) -> Any:
        """Get the classical counterpart module."""
        return self.classical_module

    def is_quantum(self) -> bool:
        """Check if using quantum backend."""
        return self.backend != QuantumBackend.CLASSICAL

    def set_backend(self, backend: QuantumBackend) -> None:
        """Switch backend."""
        self.backend = backend


# =============================================================================
# QUANTUM CIRCUIT DEFINITIONS
# =============================================================================
# These are stub implementations that use classical computation.
# They are designed to be replaced with actual quantum circuits.
# =============================================================================


class QuantumCircuits:
    """
    Library of parameterized quantum circuits for brain operations.

    Each circuit is designed to perform a specific computation relevant
    to brain function. Currently implemented as classical neural networks
    (stubs) to be replaced with actual quantum circuits.

    Circuit Types:
    =============
    1. Feature Encoding: Encode classical data into quantum state
    2. State Processing: Process quantum information
    3. Measurement: Extract classical information from quantum state
    """

    @staticmethod
    def amplitude_encoding(x: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """
        Encode classical vector into quantum state amplitudes.

        This is a stub that normalizes the input and pads/truncates to n_qubits.
        In a real quantum implementation, this would prepare a quantum state
        where the amplitudes are proportional to the input vector.

        Args:
            x: (d,) - classical input vector
            n_qubits: Number of qubits (state dimension = 2^n_qubits)

        Returns:
            (2^n_qubits,) - quantum state amplitudes (normalized)
        """
        # Normalize input
        x = x / (jnp.linalg.norm(x) + 1e-8)

        # Pad or truncate to 2^n_qubits
        dim = 2**n_qubits
        if x.shape[0] < dim:
            x = jnp.pad(x, (0, dim - x.shape[0]))
        else:
            x = x[:dim]

        # Renormalize
        x = x / (jnp.linalg.norm(x) + 1e-8)

        return x

    @staticmethod
    def angle_encoding(x: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """
        Encode classical vector into rotation angles for qubits.

        This is a stub that maps input values to rotation angles.
        In a real quantum implementation, each angle would control a rotation
        gate on a qubit.

        Args:
            x: (d,) - classical input vector
            n_qubits: Number of qubits

        Returns:
            (n_qubits,) - rotation angles
        """
        # Map to n_qubits dimensions
        if x.shape[0] < n_qubits:
            x = jnp.pad(x, (0, n_qubits - x.shape[0]))
        else:
            x = x[:n_qubits]

        # Scale to reasonable angle range
        angles = jnp.pi * jax.nn.sigmoid(x)

        return angles

    @staticmethod
    def variational_circuit(
        x: jnp.ndarray,
        params: jnp.ndarray,
        n_qubits: int,
        n_layers: int = 3,
    ) -> jnp.ndarray:
        """
        Variational quantum circuit for processing quantum states.

        This is a stub that implements a classical neural network.
        In a real quantum implementation, this would be a parameterized
        quantum circuit (PQC) with alternating layers of entangling and
        rotation gates.

        Args:
            x: (d,) - input features
            params: (n_params,) - trainable parameters
            n_qubits: Number of qubits
            n_layers: Number of layers

        Returns:
            (2^n_qubits,) - output quantum state
        """
        # Stub: classical neural network
        # In practice: apply parameterized quantum gates

        # Reshape params for NN
        input_dim = x.shape[0]
        hidden_dim = 2**n_qubits

        # Simple 2-layer NN
        if params.shape[0] < input_dim * hidden_dim + hidden_dim:
            params = jnp.pad(
                params, (0, input_dim * hidden_dim + hidden_dim - params.shape[0])
            )

        w1 = params[: input_dim * hidden_dim].reshape(input_dim, hidden_dim)
        b1 = params[input_dim * hidden_dim : input_dim * hidden_dim + hidden_dim]

        h = jnp.tanh(jnp.dot(x, w1) + b1)
        return h / (jnp.linalg.norm(h) + 1e-8)

    @staticmethod
    def measure_expectation(
        state: jnp.ndarray,
        observable: Optional[jnp.ndarray] = None,
        n_shots: int = 1024,
        deterministic: bool = False,
    ) -> float:
        """
        Measure expectation value of an observable.

        This is a stub that computes a classical expectation.
        In a real quantum implementation, this would perform quantum
        measurements and compute the expectation.

        Args:
            state: (2^n,) - quantum state
            observable: (2^n, 2^n) - Hermitian observable matrix
            n_shots: Number of measurement shots
            deterministic: If True, return exact expectation

        Returns:
            float - expectation value
        """
        if observable is None:
            # Default: measure in computational basis
            observable = jnp.eye(state.shape[0])

        # Classical expectation: state^T @ observable @ state
        expectation = jnp.dot(state, jnp.dot(observable, state)).real

        if not deterministic:
            # Add simulated sampling noise
            key = jax.random.PRNGKey(42)
            noise = jax.random.normal(key, ()) * 0.01
            expectation = expectation + noise

        return expectation.item()

    @staticmethod
    def measure_probabilities(
        state: jnp.ndarray,
        n_shots: int = 1024,
    ) -> jnp.ndarray:
        """
        Measure probabilities of each basis state.

        Args:
            state: (2^n,) - quantum state
            n_shots: Number of measurement shots

        Returns:
            (2^n,) - probability distribution
        """
        # Born rule: probabilities = |amplitudes|^2
        probs = jnp.abs(state) ** 2
        probs = probs / (jnp.sum(probs) + 1e-8)
        return probs


# =============================================================================
# QUANTUM MODULE STUBS
# =============================================================================
# These provide quantum-ready versions of each brain module.
# Currently they wrap the classical modules but are designed to be
# replaced with pure quantum implementations.
# =============================================================================


class QuantumNeurotransmitters(QuantumModule):
    """
    Quantum-ready neurotransmitter system.

    Classical counterpart: Neurotransmitters

    Quantum Approach:
    - Use quantum circuits to model chemical diffusion processes
    - Encode neurotransmitter concentrations as quantum state amplitudes
    - Use interference for modulation effects

    TODO: Implement actual quantum circuits
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        """Initialize quantum resources."""
        # For now, just use classical module
        pass

    def reset(self):
        """Reset state."""
        self.classical_module.reset()

    def step(
        self,
        observation: jnp.ndarray,
        reward: float,
        punishment: float = 0.0,
    ) -> Any:
        """
        Step the neurotransmitter system.

        Args:
            observation: Current observation
            reward: Reward signal
            punishment: Punishment signal

        Returns:
            Neurotransmitter state
        """
        # For now, use classical implementation
        return self.classical_module.step(observation, reward, punishment)


class QuantumBrainstem(QuantumModule):
    """
    Quantum-ready brainstem.

    Classical counterpart: Brainstem

    Quantum Approach:
    - Use quantum sensors for vestibular processing
    - Quantum circuits for reflex arc modeling
    - Entanglement for coordinated reflexes

    TODO: Implement actual quantum circuits
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        pass

    def reset(self):
        self.classical_module.reset()

    def preprocess_observation(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Preprocess observation."""
        return self.classical_module.preprocess_observation(observation)

    def process_reflexes(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Process reflexes."""
        return self.classical_module.process_reflexes(observation)

    def process_vestibular(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Process vestibular input."""
        return self.classical_module.process_vestibular(observation)


class QuantumBasalGanglia(QuantumModule):
    """
    Quantum-ready basal ganglia with simulated quantum circuits.

    Classical counterpart: BasalGanglia

    Quantum Approach:
    - Amplitude encoding of cortical input into quantum state
    - Variational quantum circuit for D1/D2 pathway simulation
    - Grover-inspired amplitude amplification for action search
    - Quantum interference between Go and No-Go pathways

    Quantum Advantage:
    - Grover's algorithm can search action space in O(sqrt(N)) vs O(N)
    - Superposition allows evaluating all actions simultaneously
    - Interference enables precise action selection

    Implementation:
    This implementation uses JAX to simulate quantum circuits classically.
    It can be switched to actual quantum backends (PennyLane, Qiskit) by
    replacing the circuit implementations.

    References:
    - Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search"
    - Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)
        self.classical = classical_module

        # Determine number of actions from classical module
        self.n_actions = self.classical.n_actions
        self.n_inputs = self.classical.n_inputs

        # Calculate required qubits: enough to represent all actions
        # For n_actions, we need ceil(log2(n_actions)) qubits for action encoding
        # Plus additional qubits for cortical input encoding
        self.n_action_qubits = max(4, int(jnp.ceil(jnp.log2(self.n_actions))))
        self.n_cortical_qubits = (
            max(4, config.n_qubits - self.n_action_qubits) if config else 4
        )
        self.n_qubits = self.n_action_qubits + self.n_cortical_qubits

        # Quantum circuit parameters (trainable)
        # D1 pathway parameters
        self.d1_params = (
            jax.random.normal(
                jax.random.PRNGKey(42), (self.n_cortical_qubits * self.n_action_qubits,)
            )
            * 0.1
        )

        # D2 pathway parameters
        self.d2_params = (
            jax.random.normal(
                jax.random.PRNGKey(123),
                (self.n_cortical_qubits * self.n_action_qubits,),
            )
            * 0.1
        )

        # Value estimation parameters
        self.value_params = (
            jax.random.normal(jax.random.PRNGKey(456), (self.n_qubits,)) * 0.1
        )

        # Doppler modulation parameter
        self.dopamine_mod = 0.0

    def _initialize_quantum(self):
        """Initialize quantum circuit parameters."""
        # Parameters are already initialized in __init__
        pass

    def reset(self):
        self.classical.reset()
        # Re-initialize quantum parameters
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        self.d1_params = (
            jax.random.normal(keys[0], (self.n_cortical_qubits * self.n_action_qubits,))
            * 0.1
        )
        self.d2_params = (
            jax.random.normal(keys[1], (self.n_cortical_qubits * self.n_action_qubits,))
            * 0.1
        )
        self.value_params = jax.random.normal(keys[2], (self.n_qubits,)) * 0.1
        self.dopamine_mod = 0.0

    def _encode_cortical_input(self, cortical_input: jnp.ndarray) -> jnp.ndarray:
        """
        Encode cortical input into quantum state using amplitude encoding.

        Args:
            cortical_input: (n_inputs,) - classical cortical input

        Returns:
            (2^n_cortical_qubits,) - quantum state amplitudes
        """
        # Truncate or pad input to match encoding dimension
        target_dim = 2**self.n_cortical_qubits
        if cortical_input.shape[0] > self.n_cortical_qubits:
            # Use PCA-like dimensionality reduction
            cortical_input = cortical_input[: self.n_cortical_qubits]

        # Normalize and encode
        cortical_input = cortical_input / (jnp.linalg.norm(cortical_input) + 1e-8)

        # Create amplitude-encoded state
        state = QuantumCircuits.amplitude_encoding(
            cortical_input, self.n_cortical_qubits
        )
        return state

    def _d1_pathway(self, cortical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Simulate D1 (Go) pathway as a parameterized quantum circuit.

        Args:
            cortical_state: (2^n_cortical_qubits,) - encoded cortical input

        Returns:
            (n_actions,) - D1 activations for each action
        """
        # Apply variational circuit
        processed = QuantumCircuits.variational_circuit(
            cortical_state, self.d1_params, self.n_cortical_qubits, n_layers=2
        )

        # Project to action space
        # Reshape params to (n_cortical_qubits, n_action_qubits)
        d1_w = self.d1_params.reshape((self.n_cortical_qubits, self.n_action_qubits))

        # Matrix multiplication-like operation
        d1_activations = jnp.einsum(
            "i,ij->j", processed[: self.n_cortical_qubits], d1_w
        )
        d1_activations = jax.nn.tanh(d1_activations)

        return d1_activations

    def _d2_pathway(self, cortical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Simulate D2 (No-Go) pathway as a parameterized quantum circuit.

        Args:
            cortical_state: (2^n_cortical_qubits,) - encoded cortical input

        Returns:
            (n_actions,) - D2 activations for each action
        """
        # Apply variational circuit with different parameters
        processed = QuantumCircuits.variational_circuit(
            cortical_state, self.d2_params, self.n_cortical_qubits, n_layers=2
        )

        # Project to action space
        d2_w = self.d2_params.reshape((self.n_cortical_qubits, self.n_action_qubits))
        d2_activations = jnp.einsum(
            "i,ij->j", processed[: self.n_cortical_qubits], d2_w
        )
        d2_activations = jax.nn.tanh(d2_activations)

        return d2_activations

    def _grover_amplification(
        self, d1_activations: jnp.ndarray, d2_activations: jnp.ndarray, dopamine: float
    ) -> jnp.ndarray:
        """
        Apply Grover-inspired amplitude amplification to action selection.

        In Grover's algorithm, we amplify the amplitude of the correct answer.
        Here, we use a similar approach to amplify the best action based on
        D1-D2 difference and dopamine modulation.

        Args:
            d1_activations: (n_actions,) - D1 (Go) pathway activations
            d2_activations: (n_actions,) - D2 (No-Go) pathway activations
            dopamine: Dopamine level for modulation

        Returns:
            (n_actions,) - Amplified action probabilities
        """
        # Basal ganglia output: D1 - D2 (center-surround selection)
        thalamus_output = d1_activations - d2_activations

        # Dopamine modulation: amplifies good actions, suppresses bad ones
        # Positive dopamine = more amplification of high-value actions
        amplified = thalamus_output * (1.0 + dopamine * 0.5)

        # Grover-like iteration: apply oracle and diffusion
        # For simplicity, we do one iteration
        # Oracle: mark good actions (positive values)
        oracle = jax.nn.sigmoid(amplified * 5.0)  # Sharp threshold

        # Diffusion: invert about mean
        mean_val = jnp.mean(amplified)
        diffused = 2.0 * mean_val - amplified

        # Combine
        grover_output = oracle * amplified + (1 - oracle) * diffused

        return grover_output

    def process_cortical_input(
        self,
        cortical_input: jnp.ndarray,
        dopamine: float,
    ) -> Any:
        """
        Process cortical input using quantum-inspired circuits.

        Args:
            cortical_input: (n_inputs,) - input from cortex
            dopamine: Dopamine modulation level

        Returns:
            BGState - basal ganglia state with action selection
        """
        import jax

        from brain.basal_ganglia import BGState

        # Store dopamine modulation
        self.dopamine_mod = dopamine

        # Encode cortical input
        cortical_state = self._encode_cortical_input(cortical_input)

        # Process through D1 and D2 pathways
        d1_activations = self._d1_pathway(cortical_state)
        d2_activations = self._d2_pathway(cortical_state)

        # Apply Grover-inspired amplification
        thalamus_output = self._grover_amplification(
            d1_activations, d2_activations, dopamine
        )

        # Add learned action values (classical component)
        biased_output = thalamus_output + 0.3 * self.classical.action_values

        # Softmax selection
        temperature = 0.5
        exp_output = jnp.exp(biased_output / temperature)
        probs = exp_output / jnp.sum(exp_output)

        # Selected action (stochastic sampling)
        key = jax.random.PRNGKey(int(dopamine * 1000))
        selected_action = jax.random.categorical(
            key, jnp.log(probs + 1e-8), shape=(1,)
        )[0]

        # Return BGState
        return BGState(
            d1_activations=d1_activations,
            d2_activations=d2_activations,
            thalamus_output=thalamus_output,
            selected_action=int(selected_action),
            action_values=self.classical.action_values,
        )

    def select_action_deterministic(
        self,
        cortical_input: jnp.ndarray,
        dopamine: float,
    ) -> Tuple[int, Any]:
        """
        Select action deterministically using quantum circuits.

        Args:
            cortical_input: (n_inputs,) - input from cortex
            dopamine: Dopamine modulation level

        Returns:
            (selected_action, state) - action index and BG state
        """
        state = self.process_cortical_input(cortical_input, dopamine)

        # Get probabilities from softmax
        exp_output = jnp.exp(state.thalamus_output / 0.5)
        probs = exp_output / jnp.sum(exp_output)

        # Select action with highest probability
        selected_action = int(jnp.argmax(probs))

        # Update state with deterministic selection
        state.selected_action = selected_action

        return selected_action, state

    def learn(
        self,
        reward: float,
        next_cortical_input: jnp.ndarray,
        selected_action: int,
    ) -> None:
        """
        Learn from experience using quantum-enhanced RL.

        Args:
            reward: Immediate reward
            next_cortical_input: Next cortical input
            selected_action: Action that was taken
        """
        # Classical learning (TD learning)
        self.classical.learn(reward, next_cortical_input, selected_action)

        # Quantum learning: update circuit parameters based on reward
        # This is a simplified version - in a full implementation,
        # we would use quantum gradient descent

        # Calculate TD error
        max_next_value = jnp.max(self.classical.action_values)
        current_value = self.classical.action_values[selected_action]
        td_error = reward + self.classical.gamma * max_next_value - current_value

        # Update quantum parameters based on TD error
        learning_rate = 0.01 * (1.0 + jnp.abs(td_error) * 0.5)

        # Simple gradient update (in practice: use quantum gradient descent)
        key = jax.random.PRNGKey(42)
        grad_d1 = (
            jax.random.normal(key, self.d1_params.shape) * td_error * learning_rate
        )
        grad_d2 = (
            jax.random.normal(key, self.d2_params.shape)
            * td_error
            * learning_rate
            * -1.0
        )

        self.d1_params = self.d1_params + grad_d1
        self.d2_params = self.d2_params + grad_d2

    def quantum_action_search(
        self,
        cortical_input: jnp.ndarray,
        n_actions: int,
    ) -> int:
        """
        Quantum-enhanced action search using amplitude amplification.

        This implements a simplified version of Grover's algorithm for
        searching the action space. The oracle is based on the current
        action values, and we perform a single amplification iteration.

        Args:
            cortical_input: Input from cortex
            n_actions: Number of possible actions

        Returns:
            Selected action index
        """
        # Process input
        state = self.process_cortical_input(cortical_input, self.dopamine_mod)

        # Get action probabilities
        exp_output = jnp.exp(state.thalamus_output / 0.3)
        probs = exp_output / jnp.sum(exp_output)

        # Sample from quantum-inspired distribution
        key = jax.random.PRNGKey(42)
        action_idx = jax.random.categorical(key, jnp.log(probs + 1e-8), shape=(1,))[0]

        return int(action_idx)

    def get_quantum_advantage(self) -> float:
        """
        Estimate the quantum advantage (speedup) for current configuration.

        Returns:
            Estimated speedup factor (1.0 = no advantage)
        """
        # Theoretical Grover speedup: sqrt(N/M) where N=search space, M=solutions
        n_actions = self.n_actions

        # Estimate number of good solutions (actions with high value)
        mean_value = jnp.mean(self.classical.action_values)
        good_actions = jnp.sum(self.classical.action_values > mean_value + 0.1)

        if good_actions == 0:
            return 1.0

        # Grover speedup
        speedup = jnp.sqrt(n_actions / good_actions)
        return float(jnp.clip(speedup, 1.0, n_actions))


class QuantumCerebellum(QuantumModule):
    """
    Quantum-ready cerebellum.

    Classical counterpart: Cerebellum

    Quantum Approach:
    - Quantum neural networks for error prediction
    - Quantum memory for storing error patterns
    - Quantum interference for precise error correction

    Quantum Advantage:
    - Quantum kernels can capture complex error patterns
    - Quantum memory can store exponentially many error cases

    TODO: Implement quantum error prediction circuits
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        pass

    def reset(self):
        self.classical_module.reset()

    def predict_error(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict motor error."""
        return self.classical_module.predict_error(observation, action)

    def correct_action(
        self,
        action: jnp.ndarray,
        predicted_error: jnp.ndarray,
    ) -> jnp.ndarray:
        """Correct action."""
        return self.classical_module.correct_action(action, predicted_error)

    def step(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: Optional[jnp.ndarray] = None,
        learning_rate: Optional[float] = None,
    ) -> jnp.ndarray:
        """Complete step."""
        return self.classical_module.step(
            observation, action, next_observation, learning_rate
        )


class QuantumAmygdala(QuantumModule):
    """
    Quantum-ready amygdala.

    Classical counterpart: Amygdala

    Quantum Approach:
    - Quantum pattern recognition for threat detection
    - Quantum associative memory for fear conditioning
    - Quantum amplitude amplification for threat signal enhancement

    Quantum Advantage:
    - Can detect complex threat patterns in high-dimensional space
    - Quantum memory allows storing many fear associations

    TODO: Implement quantum threat detection circuits
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        pass

    def reset(self):
        self.classical_module.reset()

    def detect_threat(self, observation: jnp.ndarray) -> Tuple[float, bool]:
        """Detect threat."""
        return self.classical_module.detect_threat(observation)

    def step(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        learning_rate: Optional[float] = None,
        punishment: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, float]:
        """Complete step."""
        return self.classical_module.step(
            observation, action, reward, learning_rate, punishment
        )


class QuantumHippocampus(QuantumModule):
    """
    Quantum-ready hippocampus.

    Classical counterpart: Hippocampus

    Quantum Approach:
    - Quantum RAM (QRAM) for memory storage and retrieval
    - Quantum pattern completion for partial memory recall
    - Quantum spatial encoding for navigation

    Quantum Advantage:
    - QRAM allows O(1) memory lookup vs O(log N) classical
    - Quantum superposition allows searching all memories simultaneously
    - Quantum entanglement enables rich spatial representations

    References:
    - Giovannetti, V., Lloyd, S., & Maccone, L. (2008). "Quantum random access memory"

    TODO: Implement QRAM-based memory system
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        pass

    def reset(self):
        self.classical_module.reset()

    def encode_spatial(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Encode spatial information."""
        return self.classical_module.encode_spatial(observation)

    def store_memory(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
        reward: float,
        punishment: float = 0.0,
    ) -> None:
        """Store memory."""
        self.classical_module.store_memory(
            observation, action, next_observation, reward, punishment
        )

    def retrieve_memory(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
    ) -> Tuple[Optional[Any], float]:
        """Retrieve memory."""
        return self.classical_module.retrieve_memory(observation, action)

    def quantum_search_memory(
        self,
        observation: jnp.ndarray,
    ) -> Optional[Any]:
        """
        Quantum-enhanced memory search.

        This is a PLACEHOLDER for a Grover-based memory search.
        In a real implementation, this would use Grover's algorithm
        to search the memory space in O(sqrt(N)) time.

        Args:
            observation: Query observation

        Returns:
            Matching memory or None
        """
        # For now, use classical search
        memory, _ = self.retrieve_memory(observation)
        return memory


class QuantumPrefrontalCortex(QuantumModule):
    """
    Quantum-ready prefrontal cortex.

    Classical counterpart: PrefrontalCortex

    Quantum Approach:
    - Quantum circuits for working memory maintenance
    - Quantum planning algorithms for decision trees
    - Quantum amplitude amplification for goal prioritization

    Quantum Advantage:
    - Can maintain superposition of multiple goals/hypotheses
    - Quantum parallelism allows evaluating many plans simultaneously
    - Quantum interference can resolve conflicts optimally

    TODO: Implement quantum planning circuits
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        pass

    def reset(self):
        self.classical_module.reset()

    def set_goal(
        self,
        description: str = "",
        target_observation: Optional[jnp.ndarray] = None,
        reward_bonus: float = 1.0,
        priority: float = 1.0,
    ) -> None:
        """Set goal."""
        self.classical_module.set_goal(
            description, target_observation, reward_bonus, priority
        )

    def step(
        self,
        observation: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        action_values: Optional[jnp.ndarray] = None,
        generate_new_plan: bool = True,
    ) -> Tuple[jnp.ndarray, float, float]:
        """Complete step."""
        return self.classical_module.step(
            observation, context, action_values, generate_new_plan
        )


class QuantumMotorCortex(QuantumModule):
    """
    Quantum-ready motor cortex.

    Classical counterpart: MotorCortex

    Quantum Approach:
    - Quantum control theory for motor command generation
    - Quantum circuits for motor primitive composition
    - Quantum annealing for optimizing movement trajectories

    Quantum Advantage:
    - Can represent complex motor patterns compactly
    - Quantum parallelism allows trying multiple motor commands
    - Quantum optimization for smooth, efficient movements

    TODO: Implement quantum motor control circuits
    """

    def __init__(self, classical_module: Any, config: Optional[QuantumConfig] = None):
        super().__init__(classical_module, config)

    def _initialize_quantum(self):
        pass

    def reset(self):
        self.classical_module.reset()

    def generate_action(
        self,
        intention: jnp.ndarray,
        use_primitives: bool = True,
    ) -> jnp.ndarray:
        """Generate action from intention."""
        return self.classical_module.generate_action(intention, use_primitives)

    def step(
        self,
        intention: jnp.ndarray,
        correction: Optional[jnp.ndarray] = None,
        inhibition: Optional[float] = None,
        use_primitives: bool = True,
    ) -> jnp.ndarray:
        """Complete step."""
        return self.classical_module.step(
            intention, correction, inhibition, use_primitives
        )


# =============================================================================
# QUANTUM BRAIN FACTORY
# =============================================================================


def create_quantum_module(
    module_name: str,
    classical_module: Any = None,
    config: Optional[QuantumConfig] = None,
) -> QuantumModule:
    """
    Factory function to create quantum-ready modules.

    Args:
        module_name: Name of the module to create
        classical_module: Optional classical module to wrap
        config: Quantum configuration

    Returns:
        Quantum module instance

    Raises:
        ValueError: If module_name is not recognized
    """
    module_map = {
        "neurotransmitters": QuantumNeurotransmitters,
        "brainstem": QuantumBrainstem,
        "basal_ganglia": QuantumBasalGanglia,
        "cerebellum": QuantumCerebellum,
        "amygdala": QuantumAmygdala,
        "hippocampus": QuantumHippocampus,
        "prefrontal_cortex": QuantumPrefrontalCortex,
        "motor_cortex": QuantumMotorCortex,
    }

    if module_name not in module_map:
        raise ValueError(
            f"Unknown module: {module_name}. Available: {list(module_map.keys())}"
        )

    if classical_module is None:
        # Import and create classical module
        classical_module = _create_classical_module(module_name)

    return module_map[module_name](classical_module, config)


def _create_classical_module(module_name: str) -> Any:
    """Create a classical module instance."""
    module_map = {
        "neurotransmitters": lambda: __import__(
            "brain.neurotransmitters", fromlist=["Neurotransmitters"]
        ).Neurotransmitters(),
        "brainstem": lambda: __import__(
            "brain.brainstem", fromlist=["Brainstem"]
        ).Brainstem(),
        "basal_ganglia": lambda: __import__(
            "brain.basal_ganglia", fromlist=["BasalGanglia"]
        ).BasalGanglia(),
        "cerebellum": lambda: __import__(
            "brain.cerebellum", fromlist=["Cerebellum"]
        ).Cerebellum(),
        "amygdala": lambda: __import__(
            "brain.amygdala", fromlist=["Amygdala"]
        ).Amygdala(),
        "hippocampus": lambda: __import__(
            "brain.hippocampus", fromlist=["Hippocampus"]
        ).Hippocampus(),
        "prefrontal_cortex": lambda: __import__(
            "brain.prefrontal_cortex", fromlist=["PrefrontalCortex"]
        ).PrefrontalCortex(),
        "motor_cortex": lambda: __import__(
            "brain.motor_cortex", fromlist=["MotorCortex"]
        ).MotorCortex(),
    }

    if module_name not in module_map:
        raise ValueError(f"Unknown module: {module_name}")

    return module_map[module_name]()


class QuantumApeBrain:
    """
    Quantum-ready integrated brain.

    This class provides a unified interface for using quantum brain modules.
    It can operate in three modes:
    1. CLASSICAL: All modules use classical computation (default)
    2. HYBRID: Some modules use quantum, others classical
    3. QUANTUM: All modules use quantum computation (future)

    Usage:
        # Create quantum brain
        q_brain = QuantumApeBrain(
            backend=QuantumBackend.CLASSICAL,  # Start with classical
            n_qubits_per_module=8,
        )

        # Step through environment
        action = q_brain.step(observation, reward=reward)

        # Switch to quantum backend (when available)
        q_brain.set_backend(QuantumBackend.PENNYLANE)
    """

    def __init__(
        self,
        backend: QuantumBackend = QuantumBackend.CLASSICAL,
        n_qubits_per_module: int = 8,
        obs_dim: int = 88,
        act_dim: int = 24,
        training: bool = True,
    ):
        """
        Initialize QuantumApeBrain.

        Args:
            backend: Quantum backend to use
            n_qubits_per_module: Number of qubits per module
            obs_dim: Observation dimension
            act_dim: Action dimension
            training: Whether in training mode
        """
        self.backend = backend
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.training = training
        self._step_count = 0

        # Create quantum config
        self.quantum_config = QuantumConfig(
            backend=backend,
            n_qubits=n_qubits_per_module,
        )

        # Initialize modules
        self.modules = {}
        self._initialize_modules()

    def _initialize_modules(self):
        """Initialize all brain modules."""
        from brain.amygdala import Amygdala
        from brain.basal_ganglia import BasalGanglia
        from brain.brainstem import Brainstem
        from brain.cerebellum import Cerebellum
        from brain.hippocampus import Hippocampus
        from brain.motor_cortex import MotorCortex
        from brain.neurotransmitters import Neurotransmitters
        from brain.prefrontal_cortex import PrefrontalCortex

        # Create classical modules
        classical_modules = {
            "basal_ganglia": BasalGanglia(
                n_actions=self.act_dim, n_inputs=self.obs_dim
            ),
            "cerebellum": Cerebellum(obs_dim=self.obs_dim, act_dim=self.act_dim),
            "amygdala": Amygdala(obs_dim=self.obs_dim, act_dim=self.act_dim),
            "hippocampus": Hippocampus(obs_dim=self.obs_dim, act_dim=self.act_dim),
            "prefrontal_cortex": PrefrontalCortex(
                obs_dim=self.obs_dim, act_dim=self.act_dim
            ),
            "motor_cortex": MotorCortex(obs_dim=self.obs_dim, act_dim=self.act_dim),
            "brainstem": Brainstem(n_joints=self.act_dim, n_obs=self.obs_dim),
            "neurotransmitters": Neurotransmitters(),
        }

        # Create quantum modules
        for name, classical in classical_modules.items():
            if self.backend == QuantumBackend.CLASSICAL:
                # Use classical module directly
                self.modules[name] = classical
            else:
                # Use quantum module
                self.modules[name] = create_quantum_module(
                    name, classical, self.quantum_config
                )

    def reset(self):
        """Reset all modules."""
        for module in self.modules.values():
            if hasattr(module, "reset"):
                module.reset()
        self._step_count = 0

    def set_backend(self, backend: QuantumBackend):
        """Switch quantum backend."""
        self.backend = backend
        self.quantum_config.backend = backend
        # Reinitialize modules with new backend
        self._initialize_modules()

    def get_module(self, name: str) -> Any:
        """Get a specific module."""
        return self.modules.get(name)

    def step(
        self,
        observation: jnp.ndarray,
        reward: Optional[float] = None,
        next_observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Perform one brain step.

        Args:
            observation: Current observation
            reward: Optional reward signal
            next_observation: Optional next observation

        Returns:
            Action vector
        """
        self._step_count += 1

        # For now, use Basal Ganglia as primary controller
        bg = self.modules["basal_ganglia"]

        # Check if quantum or classical
        if hasattr(bg, "get_classical"):
            # Quantum module - get classical for now (TODO: use quantum)
            bg_classical = bg.get_classical()
        else:
            bg_classical = bg

        dopamine = 0.0
        if reward is not None:
            dopamine = min(max(reward, 0) * 2, 1.0)

        state = bg_classical.process_cortical_input(observation, dopamine)
        action_idx, _ = bg_classical.select_action_deterministic(observation, dopamine)

        # Convert to action vector
        action = jnp.zeros(self.act_dim)
        if action_idx < self.act_dim:
            action = action.at[action_idx].set(1.0)
        action = action * 2.0 - 1.0

        # Learn
        if reward is not None and next_observation is not None and self.training:
            bg_classical.learn(reward, next_observation, action_idx)

        # Apply cerebellum
        cerebellum = self.modules["cerebellum"]
        if hasattr(cerebellum, "get_classical"):
            cerebellum = cerebellum.get_classical()
        if next_observation is not None:
            action = cerebellum.step(observation, action, next_observation)
        else:
            action = cerebellum.step(observation, action)

        # Apply motor cortex
        motor_cortex = self.modules["motor_cortex"]
        if hasattr(motor_cortex, "get_classical"):
            motor_cortex = motor_cortex.get_classical()
        intention = jnp.zeros(motor_cortex.config.primitive_dim)
        mc_action = motor_cortex.step(intention, use_primitives=False)
        action = action * 0.7 + mc_action * 0.3

        return jnp.clip(action, -1.0, 1.0)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the brain."""
        return {
            "step_count": self._step_count,
            "backend": self.backend.value,
            "modules": list(self.modules.keys()),
            "quantum_config": {
                "n_qubits": self.quantum_config.n_qubits,
                "n_shots": self.quantum_config.n_shots,
                "n_layers": self.quantum_config.n_layers,
            },
        }


# =============================================================================
# TESTS
# =============================================================================


def test_quantum():
    """Test quantum module."""
    print("Testing Quantum module...")

    import jax.random as jr

    # Test QuantumConfig
    config = QuantumConfig(n_qubits=8, n_shots=1024, n_layers=3)
    assert config.n_qubits == 8
    assert config.n_shots == 1024
    print("  ✓ QuantumConfig")

    # Test QuantumCircuits (stubs)
    circuits = QuantumCircuits()

    # Test amplitude encoding
    x = jnp.ones(8)
    state = circuits.amplitude_encoding(x, n_qubits=4)
    assert state.shape == (16,)  # 2^4 = 16
    print("  ✓ Amplitude encoding")

    # Test angle encoding
    angles = circuits.angle_encoding(x, n_qubits=4)
    assert angles.shape == (4,)
    print("  ✓ Angle encoding")

    # Test variational circuit
    params = jnp.ones(10)
    output = circuits.variational_circuit(x, params, n_qubits=4, n_layers=3)
    assert output.shape == (16,)
    print("  ✓ Variational circuit")

    # Test measurement
    expectation = circuits.measure_expectation(state)
    assert isinstance(expectation, float)
    print("  ✓ Measurement")

    # Test QuantumApeBrain
    q_brain = QuantumApeBrain(
        backend=QuantumBackend.CLASSICAL,
        n_qubits_per_module=8,
        obs_dim=88,
        act_dim=24,
    )
    assert "basal_ganglia" in q_brain.modules
    assert q_brain.backend == QuantumBackend.CLASSICAL
    print("  ✓ QuantumApeBrain initialization")

    # Test step
    key = jr.PRNGKey(42)
    obs = jr.normal(key, (88,)) * 0.1
    action = q_brain.step(obs)
    assert action.shape == (24,)
    print("  ✓ QuantumApeBrain step")

    # Test step with reward
    next_obs = obs + jr.normal(jr.PRNGKey(100), (88,)) * 0.01
    action = q_brain.step(obs, reward=1.0, next_observation=next_obs)
    assert action.shape == (24,)
    print("  ✓ QuantumApeBrain step with reward")

    # Test get_info
    info = q_brain.get_info()
    assert "step_count" in info
    assert "backend" in info
    print("  ✓ QuantumApeBrain get_info")

    # Test set_backend
    q_brain.set_backend(QuantumBackend.PENNYLANE)
    assert q_brain.backend == QuantumBackend.PENNYLANE
    print("  ✓ QuantumApeBrain set_backend")

    # Test create_quantum_module
    from brain.basal_ganglia import BasalGanglia

    bg = BasalGanglia(n_actions=24, n_inputs=88)
    q_bg = create_quantum_module("basal_ganglia", bg)
    assert q_bg is not None
    assert q_bg.classical_module == bg
    print("  ✓ create_quantum_module")

    print("✅ All Quantum module tests passed!")


if __name__ == "__main__":
    test_quantum()
