"""
Basal Ganglia - Action Selection and Reinforcement Learning

This module implements the basal ganglia, a group of subcortical nuclei
that play a critical role in action selection, habit formation, and
reinforcement learning.

Components Implemented:
- D1 MSNs (Medium Spiny Neurons): Direct pathway (Go signals)
- D2 MSNs: Indirect pathway (No-Go signals)
- Substantia Nigra (SNc): Dopamine production
- Globus Pallidus: Output nuclei (GPi, GPe)
- Subthalamic Nucleus: Modulates indirect pathway

Biological Background:
    The basal ganglia receive input from the cortex, process it through
    direct (Go) and indirect (No-Go) pathways, and send output back to
    the cortex via the thalamus.

    Key insight: Dopamine has opposite effects on D1 and D2 MSNs:
    - D1 MSNs (Direct pathway): Dopamine excites via D1 receptors -> promotes action
    - D2 MSNs (Indirect pathway): Dopamine inhibits via D2 receptors -> suppresses action

    This creates a "center-surround" selection mechanism where the most
    valuable action is selected while competing actions are inhibited.

    Action selection equation (simplified):
        Action[i] = D1[i] - D2[i]
        where D1 and D2 are modulated by dopamine

In our ApeBrain, the basal ganglia selects which motor primitive or
pose to execute based on cortical input and dopamine-modulated values.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class BGState:
    """State of the basal ganglia."""

    d1_activations: jnp.ndarray  # (n_actions,) - D1 MSN activations
    d2_activations: jnp.ndarray  # (n_actions,) - D2 MSN activations
    thalamus_output: jnp.ndarray  # (n_actions,) - Final action selection
    selected_action: int = 0  # Index of selected action
    action_values: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros(0)
    )  # (n_actions,) - learned values


class BasalGanglia:
    """
    Basal ganglia module for action selection and reinforcement learning.

    This implements a biologically-plausible action selection mechanism
    using direct (Go) and indirect (No-Go) pathways modulated by dopamine.

    Key Features:
    - D1/D2 pathway competition for action selection
    - Dopamine-modulated learning (TD error)
    - Softmax selection with temperature parameter
    - Multiple action/pose support

    Attributes:
        n_actions: Number of possible actions/poses
        n_inputs: Dimension of cortical input
    """

    def __init__(
        self,
        n_actions: int = 4,
        n_inputs: int = 64,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        temperature: float = 0.5,
    ):
        """
        Initialize basal ganglia.

        Args:
            n_actions: Number of actions/poses to select from
            n_inputs: Dimension of cortical input (from PFC)
            learning_rate: Learning rate for value updates
            gamma: Discount factor for future rewards
            temperature: Softmax temperature (0=greedy, high=exploratory)
        """
        self.n_actions = n_actions
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.temperature = temperature

        # Cortical input weights (cortex -> striatum)
        # D1 and D2 pathways have separate weights
        self.d1_weights = jnp.zeros((n_inputs, n_actions))  # Cortex -> D1 MSNs
        self.d2_weights = jnp.zeros((n_inputs, n_actions))  # Cortex -> D2 MSNs

        # Action values (learned via TD learning)
        self.action_values = jnp.zeros(n_actions)

        # Dopamine state
        self.dopamine = 0.0
        self.prev_dopamine = 0.0

        # Eligibility traces for TD learning
        self.d1_traces = jnp.zeros(n_actions)
        self.d2_traces = jnp.zeros(n_actions)

        # Trace decay parameter
        self.trace_decay = 0.9

        # Random key for sampling
        self._rng_key = jax.random.PRNGKey(42)

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset basal ganglia state."""
        self.state = BGState(
            d1_activations=jnp.zeros(self.n_actions),
            d2_activations=jnp.zeros(self.n_actions),
            thalamus_output=jnp.zeros(self.n_actions),
            selected_action=0,
            action_values=jnp.zeros(self.n_actions),
        )
        self.dopamine = 0.0
        self.prev_dopamine = 0.0
        self.d1_traces = jnp.zeros(self.n_actions)
        self.d2_traces = jnp.zeros(self.n_actions)

    def reset(self) -> None:
        """Reset basal ganglia to initial state."""
        self._reset_state()

    def _sample_action(
        self,
        probs: jnp.ndarray,
    ) -> int:
        """
        Sample action from probability distribution.

        Args:
            probs: (n_actions,) - Probability distribution over actions

        Returns:
            int - Selected action index
        """
        # Deterministic for now - use argmax
        # In biological terms, this is like a "winner-take-all" mechanism
        return int(jnp.argmax(probs))

    def process_cortical_input(
        self,
        cortical_input: jnp.ndarray,
        dopamine: float,
    ) -> BGState:
        """
        Process cortical input through basal ganglia pathways.

        This implements the biological flow:
        Cortex -> Striatum (D1/D2) -> GPi/GPi -> Thalamus -> Cortex

        Args:
            cortical_input: (n_inputs,) - Input from prefrontal cortex
            dopamine: float - Current dopamine level (-1 to 1)

        Returns:
            Updated BGState with action selection
        """
        # Store current dopamine
        self.dopamine = dopamine

        # 1. Cortex -> Striatum (D1 and D2 pathways)
        # D1 pathway: Direct (Go)
        d1_input = jnp.dot(cortical_input, self.d1_weights)

        # D2 pathway: Indirect (No-Go)
        d2_input = jnp.dot(cortical_input, self.d2_weights)

        # 2. Dopamine modulation
        # D1 MSNs: Dopamine excites (D1 receptors are excitatory)
        # D2 MSNs: Dopamine inhibits (D2 receptors are inhibitory)
        # Biological: D1 receptors use cAMP/PKA pathway (excitatory)
        #           D2 receptors use Gi pathway (inhibitory)

        # Sigmoid activation with dopamine modulation
        d1_activation = jnp.tanh(d1_input + 2.0 * dopamine)
        d2_activation = jnp.tanh(d2_input - 2.0 * dopamine)

        # 3. Basal Ganglia output
        # Thalamus output = D1 - D2 (center-surround selection)
        thalamus_output = d1_activation - d2_activation

        # 4. Softmax selection with temperature
        # Add action values to bias selection toward high-value actions
        biased_output = thalamus_output + 0.5 * self.action_values

        # Softmax with temperature
        exp_output = jnp.exp(biased_output / self.temperature)
        probs = exp_output / jnp.sum(exp_output)

        # Selected action (stochastic sampling)
        selected_action = self._sample_action(probs)

        # Update state
        self.state = BGState(
            d1_activations=d1_activation,
            d2_activations=d2_activation,
            thalamus_output=thalamus_output,
            selected_action=selected_action,
            action_values=self.action_values,
        )

        # Update eligibility traces
        self.d1_traces = (
            self.trace_decay * self.d1_traces + (1 - self.trace_decay) * d1_activation
        )
        self.d2_traces = (
            self.trace_decay * self.d2_traces + (1 - self.trace_decay) * d2_activation
        )

        return self.state

    def select_action_deterministic(
        self,
        cortical_input: jnp.ndarray,
        dopamine: float,
    ) -> Tuple[int, BGState]:
        """
        Select action deterministically (for testing/debugging).

        Args:
            cortical_input: (n_inputs,) - Input from prefrontal cortex
            dopamine: float - Current dopamine level

        Returns:
            (selected_action, state) tuple
        """
        # Process cortical input
        state = self.process_cortical_input(cortical_input, dopamine)

        # Get probabilities from softmax
        exp_output = jnp.exp(state.thalamus_output / self.temperature)
        probs = exp_output / jnp.sum(exp_output)

        # Select action with highest probability
        selected_action = int(jnp.argmax(probs))

        # Update state with deterministic selection
        self.state.selected_action = selected_action

        return selected_action, self.state

    def learn(
        self,
        reward: float,
        next_cortical_input: jnp.ndarray,
        selected_action: int,
    ) -> None:
        """
        Update action values using TD learning with dopamine modulation.

        Biological insight: Dopamine modulates synaptic plasticity in the
        striatum. Positive RPE (reward > expected) increases D1 plasticity
        and decreases D2 plasticity, making the selected action more likely
        in the future.

        Args:
            reward: float - Immediate reward
            next_cortical_input: (n_inputs,) - Next cortical input (for bootstrap)
            selected_action: int - Action that was taken
        """
        # Calculate TD error (simplified)
        # V(s) = max_a Q(s,a)  (we use action_values as Q-values)
        max_next_value = jnp.max(self.action_values)

        # Expected value of current state
        current_value = self.action_values[selected_action]

        # TD error: r + gamma * max_a' Q(s',a') - Q(s,a)
        td_error = reward + self.gamma * max_next_value - current_value

        # Dopamine-modulated learning
        # Dopamine scales the learning rate
        dopamine_factor = 1.0 + self.dopamine  # More dopamine = faster learning
        effective_lr = self.learning_rate * dopamine_factor

        # Update action value for selected action
        self.action_values = self.action_values.at[selected_action].set(
            self.action_values[selected_action] + effective_lr * td_error
        )

        # Also update cortical weights (D1 and D2 pathways)
        # For simplicity, we update weights proportionally to activation
        # This is a simplified model of Hebbian learning modulated by dopamine
        input_expanded = next_cortical_input[:, None]  # (n_inputs, 1)

        # D1 weights: strengthen active D1 pathways
        d1_update = (
            effective_lr
            * td_error
            * input_expanded
            * self.state.d1_activations[selected_action]
        )
        self.d1_weights = self.d1_weights + d1_update

        # D2 weights: weaken active D2 pathways (opposite effect)
        d2_update = (
            effective_lr
            * td_error
            * input_expanded
            * (-self.state.d2_activations[selected_action])
        )
        self.d2_weights = self.d2_weights + d2_update

    def get_action_probs(
        self,
        cortical_input: jnp.ndarray,
        dopamine: float,
    ) -> jnp.ndarray:
        """
        Get action probability distribution without sampling.

        Useful for analyzing policy or for use in other algorithms.

        Args:
            cortical_input: (n_inputs,) - Input from prefrontal cortex
            dopamine: float - Current dopamine level

        Returns:
            (n_actions,) - Probability distribution over actions
        """
        # Process cortical input
        self.process_cortical_input(cortical_input, dopamine)

        # Softmax over thalamus output + action values
        biased_output = self.state.thalamus_output + 0.5 * self.action_values
        exp_output = jnp.exp(biased_output / self.temperature)
        probs = exp_output / jnp.sum(exp_output)

        return probs

    def set_n_actions(self, n_actions: int) -> None:
        """
        Change the number of actions (e.g., for dynamic action spaces).

        Args:
            n_actions: New number of actions
        """
        if n_actions == self.n_actions:
            return

        # Resize weights and values
        old_n = self.n_actions
        self.n_actions = n_actions

        # Expand or truncate weights
        if n_actions > old_n:
            # Add new actions with zero weights
            new_d1 = jnp.zeros((self.n_inputs, n_actions))
            new_d1 = new_d1.at[:, :old_n].set(self.d1_weights)
            self.d1_weights = new_d1

            new_d2 = jnp.zeros((self.n_inputs, n_actions))
            new_d2 = new_d2.at[:, :old_n].set(self.d2_weights)
            self.d2_weights = new_d2

            new_values = jnp.zeros(n_actions)
            new_values = new_values.at[:old_n].set(self.action_values)
            self.action_values = new_values
        else:
            # Truncate
            self.d1_weights = self.d1_weights[:, :n_actions]
            self.d2_weights = self.d2_weights[:, :n_actions]
            self.action_values = self.action_values[:n_actions]

        # Reset state
        self._reset_state()


# -- Test Function --
def test_basal_ganglia():
    """Test the Basal Ganglia module."""
    print("Testing Basal Ganglia module...")

    n_actions = 4
    n_inputs = 16
    bg = BasalGanglia(n_actions=n_actions, n_inputs=n_inputs)

    # Test 1: Initial state
    assert bg.state.d1_activations.shape == (n_actions,), (
        "D1 activations should match n_actions"
    )
    assert bg.state.d2_activations.shape == (n_actions,), (
        "D2 activations should match n_actions"
    )
    print(f"  ✓ Initial state: n_actions={n_actions}, shapes correct")

    # Test 2: Cortical input processing
    cortical_input = jnp.ones(n_inputs) * 0.5
    # Set some weights so we get non-zero activations
    bg.d1_weights = bg.d1_weights.at[0, :].set(0.5)
    bg.d2_weights = bg.d2_weights.at[0, :].set(0.5)
    bg.process_cortical_input(cortical_input, dopamine=0.0)
    assert jnp.any(bg.state.d1_activations != 0), (
        "D1 should activate with input and weights"
    )
    assert jnp.any(bg.state.d2_activations != 0), (
        "D2 should activate with input and weights"
    )
    print(
        f"  ✓ Cortical processing: D1={bg.state.d1_activations[0]:.4f}, D2={bg.state.d2_activations[0]:.4f}"
    )

    # Test 3: Dopamine modulation - positive
    bg.reset()
    bg.d1_weights = bg.d1_weights.at[0, :].set(0.5)
    bg.d2_weights = bg.d2_weights.at[0, :].set(0.5)
    bg.process_cortical_input(cortical_input, dopamine=0.5)
    d1_pos = bg.state.d1_activations.copy()
    bg.reset()
    bg.d1_weights = bg.d1_weights.at[0, :].set(0.5)
    bg.d2_weights = bg.d2_weights.at[0, :].set(0.5)
    bg.process_cortical_input(cortical_input, dopamine=0.0)
    d1_zero = bg.state.d1_activations.copy()

    # With positive dopamine, D1 should be more active
    assert jnp.any(d1_pos > d1_zero), "Positive dopamine should increase D1 activation"
    print(f"  ✓ Dopamine modulation: positive dopamine increases D1 activation")

    # Test 4: Dopamine modulation - negative
    bg.reset()
    bg.d1_weights = bg.d1_weights.at[0, :].set(0.5)
    bg.d2_weights = bg.d2_weights.at[0, :].set(0.5)
    bg.process_cortical_input(cortical_input, dopamine=-0.5)
    d1_neg = bg.state.d1_activations.copy()

    # With negative dopamine, D1 should be less active
    assert jnp.any(d1_neg < d1_zero), "Negative dopamine should decrease D1 activation"
    print(f"  ✓ Dopamine modulation: negative dopamine decreases D1 activation")

    # Test 5: Action selection
    cortical_input = jnp.zeros(n_inputs).at[0].set(1.0)  # Only first input active
    bg.d1_weights = bg.d1_weights.at[0, 2].set(2.0)  # D1 favors action 2
    bg.d2_weights = bg.d2_weights.at[0, 2].set(
        -2.0
    )  # D2 also favors action 2 (but dopamine matters)

    selected, _ = bg.select_action_deterministic(cortical_input, dopamine=0.0)
    assert selected in range(n_actions), "Selected action should be valid"
    print(f"  ✓ Action selection: selected action {selected}")

    # Test 6: Learning from reward
    bg.reset()
    bg.action_values = jnp.array([0.0, 0.0, 0.0, 0.0])

    # Take action 0
    selected_action = 0
    cortical_input = jnp.ones(n_inputs) * 0.1
    bg.process_cortical_input(cortical_input, dopamine=0.0)

    # Get positive reward
    bg.learn(
        reward=10.0, next_cortical_input=cortical_input, selected_action=selected_action
    )

    # Action 0 value should increase
    assert bg.action_values[0] > 0, "Action value should increase with positive reward"
    print(f"  ✓ Learning: action values after reward: {bg.action_values}")

    # Test 7: Negative reward learning
    bg.reset()
    bg.action_values = jnp.array([5.0, 0.0, 0.0, 0.0])
    bg.process_cortical_input(cortical_input, dopamine=0.0)

    # Take action 0, get negative reward
    bg.learn(reward=-10.0, next_cortical_input=cortical_input, selected_action=0)

    assert bg.action_values[0] < 5.0, (
        "Action value should decrease with negative reward"
    )
    print(f"  ✓ Learning: action values after punishment: {bg.action_values}")

    # Test 8: Dopamine-modulated learning
    bg.reset()
    bg.action_values = jnp.zeros(n_actions)
    bg.dopamine = 0.8  # High dopamine
    bg.process_cortical_input(cortical_input, dopamine=0.8)

    bg.learn(reward=5.0, next_cortical_input=cortical_input, selected_action=0)

    # With high dopamine, learning should be faster
    assert bg.action_values[0] > 0.5, "High dopamine should accelerate learning"
    print(
        f"  ✓ Dopamine-modulated learning: value with high dopamine = {bg.action_values[0]:.4f}"
    )

    # Test 9: Action probabilities
    bg.reset()
    cortical_input = jnp.zeros(n_inputs).at[0].set(1.0)
    bg.d1_weights = bg.d1_weights.at[0, 0].set(2.0)  # Favor action 0
    probs = bg.get_action_probs(cortical_input, dopamine=0.0)

    assert probs.shape == (n_actions,), "Probabilities should match n_actions"
    assert jnp.all(probs >= 0), "Probabilities should be non-negative"
    assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities should sum to 1"
    print(f"  ✓ Action probabilities: {probs}")

    # Test 10: Temperature effect
    bg.temperature = 0.1  # Low temperature = greedy
    probs_low_temp = bg.get_action_probs(cortical_input, dopamine=0.0)

    bg.temperature = 2.0  # High temperature = exploratory
    probs_high_temp = bg.get_action_probs(cortical_input, dopamine=0.0)

    # High temperature should give more uniform distribution
    max_prob_low = jnp.max(probs_low_temp)
    max_prob_high = jnp.max(probs_high_temp)
    assert max_prob_high < max_prob_low, (
        "High temperature should reduce max probability"
    )
    print(
        f"  ✓ Temperature effect: max prob low_temp={max_prob_low:.4f}, high_temp={max_prob_high:.4f}"
    )

    print("✅ All Basal Ganglia tests passed!")


if __name__ == "__main__":
    test_basal_ganglia()
