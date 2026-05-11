"""
Neurotransmitter Systems - Biological Modulation Signals

This module implements biologically-plausible neurotransmitter systems that
modulate learning and behavior in the ApeBrain architecture.

Neurotransmitters Implemented:
- Dopamine: Reward prediction error (RPE) - modulates synaptic plasticity
- Serotonin: Patience, risk aversion, long-term planning
- GABA: Inhibition, stabilization, prevents runaway excitation
- Noradrenaline: Arousal, attention, novelty detection
- Acetylcholine: Learning rate modulation, memory consolidation

Biological Background:
    Dopamine neurons in the ventral tegmental area (VTA) and substantia nigra
    fire in response to unexpected rewards (positive RPE) or the omission
    of expected rewards (negative RPE). This dopamine signal modulates
    synaptic plasticity throughout the brain, particularly in the basal ganglia
    and prefrontal cortex.

    Serotonin from the raphe nuclei promotes patience and risk aversion.
    Low serotonin is associated with impulsivity.

    GABA is the primary inhibitory neurotransmitter, providing balance
    to excitatory glutamate signals.

    Noradrenaline from the locus coeruleus signals arousal, novelty, and
    uncertainty, enhancing attention and memory consolidation.

    Acetylcholine from the basal forebrain modulates learning rate and
    memory consolidation, particularly in the hippocampus.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class NeurotransmitterState:
    """Current state of all neurotransmitter levels."""

    dopamine: float = 0.0  # Reward prediction error
    serotonin: float = 1.0  # Patience (0-2 scale, 1=baseline)
    gaba: float = 0.5  # Inhibition (0-1 scale)
    noradrenaline: float = 0.0  # Arousal/novelty (0-1 scale)
    acetylcholine: float = 1.0  # Learning modulation (0-2 scale)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "gaba": self.gaba,
            "noradrenaline": self.noradrenaline,
            "acetylcholine": self.acetylcholine,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeurotransmitterState":
        """Create from dictionary."""
        return cls(**data)


class Neurotransmitters:
    """
    Biologically-plausible neurotransmitter modulation system.

    This class simulates the major neurotransmitter systems that modulate
    learning and behavior in biological brains. Each neurotransmitter has
    a specific role in the learning process:

    Attributes:
        state: Current neurotransmitter levels
        dopamine_baseline: Baseline dopamine firing rate
        novelty_threshold: Threshold for noradrenaline response
        memory_buffer: Recent states for novelty detection
    """

    def __init__(
        self,
        dopamine_baseline: float = 0.0,
        serotonin_baseline: float = 1.0,
        gaba_baseline: float = 0.5,
        novelty_threshold: float = 0.15,
        memory_size: int = 1000,
    ):
        """
        Initialize neurotransmitter system.

        Args:
            dopamine_baseline: Baseline dopamine level (biological: ~3-8 Hz firing)
            serotonin_baseline: Baseline serotonin tone
            gaba_baseline: Baseline inhibition level
            novelty_threshold: State change threshold to trigger noradrenaline
            memory_size: Number of recent states to track for novelty detection
        """
        self.state = NeurotransmitterState(
            dopamine=dopamine_baseline,
            serotonin=serotonin_baseline,
            gaba=gaba_baseline,
            noradrenaline=0.0,
            acetylcholine=1.0,
        )

        self.dopamine_baseline = dopamine_baseline
        self.serotonin_baseline = serotonin_baseline
        self.novelty_threshold = novelty_threshold
        self.memory_size = memory_size

        # Ring buffer for recent states (for novelty detection)
        self._state_memory = []

        # Previous reward prediction for TD error calculation
        self._prev_value: Optional[float] = None
        self._prev_state_hash: Optional[int] = None

        # Learning parameters
        self.gamma = 0.99  # Discount factor (biological: ~0.9-0.99)
        self.lr_base = 0.001  # Base learning rate

    def reset(self) -> None:
        """Reset neurotransmitter system to baseline states."""
        self.state = NeurotransmitterState(
            dopamine=self.dopamine_baseline,
            serotonin=self.serotonin_baseline,
            gaba=self.dopamine_baseline,  # Use dopamine baseline for gaba too
            noradrenaline=0.0,
            acetylcholine=1.0,
        )
        self._prev_value = None
        self._prev_state_hash = None
        self._state_memory = []

    def update(
        self,
        reward: float,
        predicted_reward: float,
        state: jnp.ndarray,
        done: bool = False,
    ) -> NeurotransmitterState:
        """
        Update neurotransmitter levels based on current experience.

        This implements biologically-plausible modulation:
        - Dopamine: TD error (reward - expected_reward)
        - Serotonin: Inversely related to reward volatility
        - Noradrenaline: Novelty/uncertainty detection
        - GABA: Adaptive inhibition based on activity
        - Acetylcholine: Modulates learning rate

        Args:
            reward: Immediate reward received
            predicted_reward: Expected reward (value function estimate)
            state: Current observation/state
            done: Whether episode terminated

        Returns:
            Updated NeurotransmitterState
        """
        # 1. Dopamine: Reward Prediction Error (RPE)
        # Biological: δ = r + γ*V(s') - V(s)
        # For simplicity, we use: δ = r + γ*V_next - V_current
        # In practice, predicted_reward is V(s) and we estimate V(s') from next state
        if self._prev_value is not None:
            # TD error: actual + discounted future - expected
            td_error = reward + self.gamma * predicted_reward - self._prev_value
        else:
            # First step: just the reward itself
            td_error = reward

        # Dopamine firing rate (Hz equivalent, normalized)
        # Positive RPE -> burst firing (~20-30 Hz)
        # Negative RPE -> pause in firing (~0-3 Hz)
        # We normalize to -1 to +1 range
        self.state.dopamine = jnp.tanh(td_error * 5.0)  # Scale for sensitivity

        # 2. Serotonin: Patience and risk aversion
        # Low when rewards are volatile or unexpected
        # High when stable and predictable
        reward_volatility = jnp.abs(td_error)
        self.state.serotonin = jnp.clip(
            self.serotonin_baseline * (1.0 - 0.5 * reward_volatility), 0.1, 2.0
        )

        # 3. Noradrenaline: Novelty detection
        state_hash = self._hash_state(state)
        novelty = self._detect_novelty(state_hash)

        # Novelfy triggers noradrenaline release
        # Biological: phasic response to novel stimuli
        self.state.noradrenaline = jnp.clip(novelty * 2.0, 0.0, 1.0)

        # 4. GABA: Adaptive inhibition
        # Higher when system is over-active (high dopamine + noradrenaline)
        excitation = jnp.abs(self.state.dopamine) + self.state.noradrenaline
        self.state.gaba = jnp.clip(0.1 + 0.4 * excitation, 0.1, 0.9)

        # 5. Acetylcholine: Learning rate modulation
        # Higher when novelty is high or rewards are unexpected
        # Biological: ACh release during learning and memory consolidation
        self.state.acetylcholine = jnp.clip(
            0.5 + 0.5 * novelty + 0.3 * jnp.abs(self.state.dopamine), 0.1, 2.0
        )

        # Update memory
        self._prev_value = predicted_reward
        self._prev_state_hash = state_hash
        self._add_to_memory(state_hash)

        # If episode done, reset some state
        if done:
            self._prev_value = None

        return self.state

    def _hash_state(self, state: jnp.ndarray) -> int:
        """Create a simple hash of the state for novelty detection."""
        # Use a simple hash based on the first few dimensions
        # For JAX compatibility, we use a deterministic hash
        state_flat = state.ravel()
        # Take first 8 dimensions for hashing
        truncated = state_flat[:8] if len(state_flat) > 8 else state_flat
        # Quantize and create hash
        quantized = (truncated * 100).astype(jnp.int32)
        hash_val = jnp.sum(
            quantized * jnp.array([1, 7, 49, 343, 2401, 16807, 117649, 823543])
        )
        return hash_val.item()  # Convert to Python int

    def _detect_novelty(self, state_hash: int) -> float:
        """Detect novelty of current state."""
        if not self._state_memory:
            return 1.0  # First state is always novel

        # Count occurrences in recent memory
        occurrences = self._state_memory.count(state_hash)

        # Novelty = inverse of frequency
        # Normalize by memory size
        frequency = occurrences / len(self._state_memory)
        novelty = 1.0 - frequency

        # Only trigger if above threshold
        if novelty > self.novelty_threshold:
            return novelty
        return 0.0

    def _add_to_memory(self, state_hash: int) -> None:
        """Add current state to memory ring buffer."""
        self._state_memory.append(state_hash)
        if len(self._state_memory) > self.memory_size:
            self._state_memory.pop(0)

    def get_learning_rate(self) -> float:
        """
        Get adaptive learning rate based on neurotransmitter state.

        Biological insight: Acetylcholine and dopamine both facilitate
        synaptic plasticity. GABA inhibits it.

        Returns:
            Adaptive learning rate
        """
        # Base learning rate modulated by neurotransmitters
        lr = self.lr_base

        # Dopamine enhances learning (especially for positive RPE)
        dopamine_factor = 1.0 + 0.5 * jnp.clip(self.state.dopamine, -1.0, 1.0)

        # Acetylcholine enhances memory consolidation
        ach_factor = 1.0 + 0.3 * (self.state.acetylcholine - 1.0)

        # GABA inhibits learning (prevents over-updating)
        gaba_factor = 1.0 - 0.7 * self.state.gaba

        # Combined modulation
        modulated_lr = lr * dopamine_factor * ach_factor * gaba_factor

        return jnp.clip(modulated_lr, 0.0001, 0.1)

    def get_exploration_rate(self) -> float:
        """
        Get adaptive exploration rate based on neurotransmitter state.

        Biological insight: High dopamine and noradrenaline increase
        exploration. High GABA decreases it.

        Returns:
            Exploration rate (0-1)
        """
        # Base exploration
        base_exploration = 0.1

        # Dopamine: positive RPE -> more exploration
        dopamine_effect = 0.1 * jnp.clip(self.state.dopamine, 0.0, 1.0)

        # Noradrenaline: novelty -> more exploration
        na_effect = 0.2 * self.state.noradrenaline

        # GABA: inhibition -> less exploration
        gaba_effect = -0.15 * self.state.gaba

        exploration = base_exploration + dopamine_effect + na_effect + gaba_effect

        return jnp.clip(exploration, 0.01, 0.5)

    def get_reward_modulation(self) -> float:
        """
        Get reward scaling based on neurotransmitter state.

        Biological insight: Dopamine amplifies reward signals.

        Returns:
            Reward scaling factor
        """
        # Dopamine amplifies reward perception
        return 1.0 + 0.5 * jnp.clip(self.state.dopamine, 0.0, 1.0)


# -- Test Function --
def test_neurotransmitters():
    """Test the Neurotransmitters module."""
    print("Testing Neurotransmitters module...")

    neuro = Neurotransmitters()

    # Test 1: Initial state
    assert abs(neuro.state.dopamine) < 0.01, "Initial dopamine should be near baseline"
    assert neuro.state.serotonin == 1.0, "Initial serotonin should be baseline"

    # Test 2: Positive reward (unexpected)
    state = jnp.zeros(10)
    neuro.update(reward=10.0, predicted_reward=0.0, state=state)
    assert neuro.state.dopamine > 0, "Positive RPE should increase dopamine"
    print(f"  ✓ Positive RPE: dopamine={neuro.state.dopamine:.4f}")

    # Test 3: Negative reward (unexpected)
    neuro.reset()
    neuro.update(reward=-10.0, predicted_reward=0.0, state=state)
    assert neuro.state.dopamine < 0, "Negative RPE should decrease dopamine"
    print(f"  ✓ Negative RPE: dopamine={neuro.state.dopamine:.4f}")

    # Test 4: Novel state detection
    neuro.reset()
    neuro.update(reward=0.0, predicted_reward=0.0, state=jnp.ones(10) * 999)
    assert neuro.state.noradrenaline > 0, "Novel state should trigger noradrenaline"
    print(f"  ✓ Novelty detection: noradrenaline={neuro.state.noradrenaline:.4f}")

    # Test 5: Learning rate modulation
    neuro.reset()
    neuro.update(reward=5.0, predicted_reward=0.0, state=state)
    lr = neuro.get_learning_rate()
    assert lr > 0, "Learning rate should be positive"
    assert lr != neuro.lr_base, "Learning rate should be modulated from baseline"
    print(f"  ✓ Adaptive learning rate: lr={lr:.6f} (base={neuro.lr_base})")

    # Test 6: Exploration rate
    neuro.reset()
    neuro.update(reward=0.0, predicted_reward=0.0, state=state)
    neuro.state.noradrenaline = 0.8  # Simulate novelty
    neuro.state.dopamine = 0.5  # Simulate positive expectation
    exp_rate = neuro.get_exploration_rate()
    assert exp_rate > 0.1, "Exploration should increase with novelty"
    print(f"  ✓ Adaptive exploration: rate={exp_rate:.4f}")

    print("✅ All Neurotransmitters tests passed!")


if __name__ == "__main__":
    test_neurotransmitters()
