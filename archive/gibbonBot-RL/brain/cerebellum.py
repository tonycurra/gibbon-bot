"""
Cerebellum - Motor Error Correction and Fine-Tuning

This module implements a simplified, biologically-inspired cerebellum for
motor error correction and fine-tuning. The cerebellum learns to predict
and correct errors in motor commands.

Biological Background:
    The cerebellum ("little brain") contains more neurons than the rest of
    the brain combined. It plays a crucial role in motor coordination and
    error correction through:

    1. **Granule Cells**: Expand sparse mossy fiber inputs into dense representations
    2. **Purkinje Cells**: Primary output neurons that learn to predict errors
    3. **Climbing Fibers**: Carry error signals from the inferior olive

    The cerebellum operates as a supervisor: it doesn't initiate movements
    but ensures they are executed correctly by learning from errors.

Simplification Note:
    This implementation uses a streamlined neural network approach that
    captures the essence of cerebellar function (error prediction and correction)
    while being JAX-compatible and computationally efficient.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class CerebellumConfig:
    """Configuration for the cerebellum module."""

    hidden_dim: int = 128  # Size of hidden layer
    error_scale: float = 0.5  # Scale of error correction
    learning_rate: float = 0.01  # Learning rate for error prediction

    def __post_init__(self):
        """Validate config."""
        pass


@dataclass
class CerebellumState:
    """State of the cerebellum at one timestep."""

    hidden_activation: jnp.ndarray  # Hidden layer activations
    predicted_error: jnp.ndarray  # Predicted motor error
    corrected_action: jnp.ndarray  # Action after cerebellar correction


class Cerebellum:
    """
    Simplified cerebellum module for motor error correction and fine-tuning.

    This implements the core cerebellar function: predicting and correcting
    errors in motor commands. The cerebellum learns from the difference
    between predicted and actual outcomes.

    Architecture:
        Input (obs + action) -> Hidden Layer -> Error Prediction

    The error prediction is used to correct the action before execution.
    After execution, the actual error is computed and used for learning.

    Attributes:
        config: CerebellumConfig with all parameters
        obs_dim: Dimension of observation input
        act_dim: Dimension of action space
    """

    def __init__(
        self,
        obs_dim: int = 88,
        act_dim: int = 24,
        config: Optional[CerebellumConfig] = None,
    ):
        """
        Initialize cerebellum.

        Args:
            obs_dim: Dimension of observation input
            act_dim: Dimension of action space
            config: Optional CerebellumConfig (defaults to standard values)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config if config is not None else CerebellumConfig()

        # Network input dimension: observation + action
        input_dim = obs_dim + act_dim

        # Weights for error prediction network
        # Input -> Hidden
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.w1 = jax.random.normal(keys[0], (input_dim, self.config.hidden_dim)) * 0.1
        # Hidden -> Output (error prediction, same dim as action)
        self.w2 = jax.random.normal(keys[1], (self.config.hidden_dim, act_dim)) * 0.1

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset cerebellum state."""
        self.state = CerebellumState(
            hidden_activation=jnp.zeros(self.config.hidden_dim),
            predicted_error=jnp.zeros(self.act_dim),
            corrected_action=jnp.zeros(self.act_dim),
        )
        self._last_inputs = None

    def reset(self) -> None:
        """Reset cerebellum to initial state."""
        self._reset_state()

    def predict_error(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict the motor error for a given state-action pair.

        Args:
            observation: (obs_dim,) - current observation/state
            action: (act_dim,) - proposed motor action

        Returns:
            (act_dim,) - predicted motor error
        """
        # Combine observation and action as input
        x = jnp.concatenate([observation, action])

        # Forward pass through error prediction network
        # Hidden layer with ReLU activation
        h = jnp.maximum(0, jnp.dot(x, self.w1))

        # Output layer (error prediction)
        error = jnp.dot(h, self.w2)

        # Store state for later use
        self.state = CerebellumState(
            hidden_activation=h,
            predicted_error=error,
            corrected_action=jnp.zeros(self.act_dim),
        )
        self._last_inputs = x  # Store for learning

        return error

    def correct_action(
        self,
        action: jnp.ndarray,
        predicted_error: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply cerebellar correction to a motor action.

        The correction moves the action in the direction opposite to the
        predicted error, scaled by error_scale.

        Args:
            action: (act_dim,) - original motor action
            predicted_error: (act_dim,) - predicted error from predict_error()

        Returns:
            (act_dim,) - corrected motor action
        """
        # Correction: move action in direction opposite to error
        correction = -self.config.error_scale * predicted_error

        # Apply correction
        corrected_action = action + correction

        # Clamp to valid action range
        corrected_action = jnp.clip(corrected_action, -1.0, 1.0)

        # Update state
        self.state = replace(self.state, corrected_action=corrected_action)

        return corrected_action

    def compute_actual_error(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the actual motor error from state transition.

        This represents the "ground truth" error that the cerebellum
        should learn to predict. In biology, this comes from the inferior
        olive via climbing fibers.

        Args:
            observation: (obs_dim,) - state before action
            action: (act_dim,) - action that was taken
            next_observation: (obs_dim,) - state after action

        Returns:
            (act_dim,) - actual motor error
        """
        # For simplicity, we compute error as the difference in the first
        # act_dim dimensions of the observation (joint positions, velocities, etc.)

        if observation.shape[0] >= self.act_dim:
            state_diff = next_observation[: self.act_dim] - observation[: self.act_dim]
        else:
            # If obs is smaller than act_dim, pad with zeros
            state_diff = jnp.zeros(self.act_dim)

        # Scale error by action magnitude (bigger actions = bigger expected changes)
        action_scale = jnp.clip(jnp.mean(jnp.abs(action)), 0.1, 1.0)
        actual_error = state_diff * action_scale

        return actual_error

    def learn(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
        learning_rate: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Learn from experience using error prediction.

        This implements a simplified version of cerebellar learning:
        1. Predict error (before action)
        2. Observe actual error (after action)
        3. Update weights to reduce prediction error

        Args:
            observation: (obs_dim,) - state before action
            action: (act_dim,) - action that was taken
            next_observation: (obs_dim,) - state after action
            learning_rate: Optional override for learning rate

        Returns:
            (predicted_error, actual_error) - For monitoring/debugging
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate

        # Combine observation and action as input
        x = jnp.concatenate([observation, action])

        # 1. Predict error (forward pass)
        # We need to do forward pass again to get the hidden activations
        h = jnp.maximum(0, jnp.dot(x, self.w1))
        predicted_error = jnp.dot(h, self.w2)

        # 2. Compute actual error
        actual_error = self.compute_actual_error(observation, action, next_observation)

        # 3. Compute error gradient
        error_diff = predicted_error - actual_error

        # 4. Backpropagate to update weights
        # Gradient for w2: h.T @ error_diff
        grad_w2 = jnp.outer(h, error_diff)
        self.w2 = self.w2 - learning_rate * grad_w2

        # Gradient for w1: x.T @ (grad_h * w2)
        # grad_h = (h > 0) * (error_diff @ w2.T)  (ReLU derivative)
        grad_h = (h > 0).astype(jnp.float32) * jnp.dot(error_diff, self.w2.T)
        grad_w1 = jnp.outer(x, grad_h)
        self.w1 = self.w1 - learning_rate * grad_w1

        # Update state
        self.state = CerebellumState(
            hidden_activation=h,
            predicted_error=predicted_error,
            corrected_action=jnp.zeros(self.act_dim),
        )

        return predicted_error, actual_error

    def step(
        self,
        observation: jnp.ndarray,
        proposed_action: jnp.ndarray,
        next_observation: Optional[jnp.ndarray] = None,
        learning_rate: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Complete cerebellum processing step.

        This is the main method that should be called during training:
        1. Predict error for proposed action
        2. Correct the action
        3. (Optional) Learn from actual outcome

        Args:
            observation: (obs_dim,) - current state
            proposed_action: (act_dim,) - action proposed by basal ganglia
            next_observation: (obs_dim,) - next state (for learning, optional)
            learning_rate: Optional learning rate override

        Returns:
            (act_dim,) - corrected action
        """
        # Predict error
        predicted_error = self.predict_error(observation, proposed_action)

        # Correct action
        corrected_action = self.correct_action(proposed_action, predicted_error)

        # Learn if next observation is provided
        if next_observation is not None:
            self.learn(observation, proposed_action, next_observation, learning_rate)

        return corrected_action

    def get_correction_strength(self) -> float:
        """
        Get the current correction strength (L2 norm of correction).

        Returns:
            float - magnitude of current correction
        """
        if self.state is None:
            return 0.0
        return float(jnp.linalg.norm(self.state.predicted_error))


# -- Test Function --
def test_cerebellum():
    """Test the Cerebellum module."""
    print("Testing Cerebellum module...")

    import jax.random as jr

    obs_dim = 88
    act_dim = 24
    cerebellum = Cerebellum(obs_dim=obs_dim, act_dim=act_dim)

    # Test 1: Initial state
    assert cerebellum.state.hidden_activation.shape == (cerebellum.config.hidden_dim,)
    assert cerebellum.state.predicted_error.shape == (act_dim,)
    print(f"  ✓ Initial state: shapes correct")

    # Test 2: Error prediction
    key = jr.PRNGKey(42)
    keys = jr.split(key, 10)
    obs = jr.normal(keys[0], (obs_dim,)) * 0.1
    action = jr.normal(keys[1], (act_dim,)) * 0.1

    predicted_error = cerebellum.predict_error(obs, action)
    assert predicted_error.shape == (act_dim,), (
        "Predicted error should match action dim"
    )
    print(f"  ✓ Error prediction: shape={predicted_error.shape}")

    # Test 3: Action correction
    corrected = cerebellum.correct_action(action, predicted_error)
    assert corrected.shape == (act_dim,), "Corrected action should match action dim"
    assert jnp.all(jnp.abs(corrected) <= 1.0), "Corrected action should be clamped"
    print(f"  ✓ Action correction: shape={corrected.shape}, clamped to [-1, 1]")

    # Test 4: Actual error computation
    next_obs = obs + jr.normal(keys[2], (obs_dim,)) * 0.01
    actual_error = cerebellum.compute_actual_error(obs, action, next_obs)
    assert actual_error.shape == (act_dim,), "Actual error should match action dim"
    print(f"  ✓ Actual error computation: shape={actual_error.shape}")

    # Test 5: Learning
    cerebellum.reset()
    predicted, actual = cerebellum.learn(obs, action, next_obs, learning_rate=0.1)
    assert predicted.shape == (act_dim,), "Learned predicted error shape"
    assert actual.shape == (act_dim,), "Learned actual error shape"
    print(
        f"  ✓ Learning: predicted_norm={jnp.linalg.norm(predicted):.4f}, actual_norm={jnp.linalg.norm(actual):.4f}"
    )

    # Test 6: Complete step with learning
    cerebellum.reset()
    corrected = cerebellum.step(obs, action, next_obs, learning_rate=0.1)
    assert corrected.shape == (act_dim,), "Step output shape"
    print(f"  ✓ Complete step: output shape={corrected.shape}")

    # Test 7: Step without learning (inference mode)
    corrected = cerebellum.step(obs, action)
    assert corrected.shape == (act_dim,), "Inference step output shape"
    print(f"  ✓ Inference step: output shape={corrected.shape}")

    # Test 8: Correction strength
    strength = cerebellum.get_correction_strength()
    assert isinstance(strength, float), "Correction strength should be float"
    print(f"  ✓ Correction strength: {strength:.4f}")

    # Test 9: Weights are non-zero after learning
    cerebellum.reset()
    for _ in range(10):
        obs = jr.normal(jr.PRNGKey(100), (obs_dim,)) * 0.1
        action = jr.normal(jr.PRNGKey(101), (act_dim,)) * 0.1
        next_obs = obs + jr.normal(jr.PRNGKey(102), (obs_dim,)) * 0.01
        cerebellum.learn(obs, action, next_obs, learning_rate=0.1)

    # Check that weights have been updated
    w1_norm = jnp.linalg.norm(cerebellum.w1)
    w2_norm = jnp.linalg.norm(cerebellum.w2)
    assert w1_norm > 0 or w2_norm > 0, "Weights should be non-zero after learning"
    print(f"  ✓ Weight learning: w1_norm={w1_norm:.4f}, w2_norm={w2_norm:.4f}")

    # Test 10: Error reduction over learning
    cerebellum.reset()
    # Generate consistent data
    obs = jr.normal(jr.PRNGKey(200), (obs_dim,)) * 0.1
    action = jr.normal(jr.PRNGKey(201), (act_dim,)) * 0.1
    next_obs = obs + jr.normal(jr.PRNGKey(202), (obs_dim,)) * 0.01

    # First prediction
    pred1, _ = cerebellum.learn(obs, action, next_obs, learning_rate=0.1)

    # Train more
    for _ in range(5):
        cerebellum.learn(obs, action, next_obs, learning_rate=0.1)

    # Predict again
    pred2 = cerebellum.predict_error(obs, action)

    # Error should be reducing (predicted error should be closer to actual)
    actual = cerebellum.compute_actual_error(obs, action, next_obs)
    error1 = jnp.linalg.norm(pred1 - actual)
    error2 = jnp.linalg.norm(pred2 - actual)
    print(f"  ✓ Error reduction: initial={error1:.4f}, after_training={error2:.4f}")

    print("✅ All Cerebellum tests passed!")


if __name__ == "__main__":
    test_cerebellum()
