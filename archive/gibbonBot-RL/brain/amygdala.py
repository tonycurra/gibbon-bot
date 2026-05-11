"""
Amygdala - Fear, Punishment, and Threat Detection

This module implements a biologically-inspired amygdala for processing
fear, punishment signals, and threat detection. The amygdala plays a
critical role in emotional regulation and risk assessment.

Biological Background:
    The amygdala is a set of almond-shaped nuclei deep in the temporal lobe
    that processes:

    1. **Fear Conditioning**: Learning associations between neutral stimuli
       and aversive outcomes (punishment)
    2. **Threat Detection**: Rapid identification of potential dangers
    3. **Emotional Memory**: Storing memories of emotionally significant events
    4. **Punishment Signal**: Generating internal punishment signals to
       guide behavior away from harmful actions

    In RL terms, the amygdala helps the agent avoid states/actions that
    lead to negative outcomes (punishment, pain, failure).

Architecture:
    - Input: observations, actions, reward signals
    - Processes: threat detection, fear memory, punishment signal
    - Output: punishment/avoidance signals for action selection
"""

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class AmygdalaConfig:
    """Configuration for the amygdala module."""

    hidden_dim: int = 64  # Size of hidden layer for threat detection
    fear_threshold: float = 0.5  # Threshold for fear activation
    punishment_strength: float = 1.0  # Multiplier for punishment signals
    learning_rate: float = 0.05  # Learning rate for fear conditioning
    fear_decay: float = 0.95  # Decay rate for fear memory

    def __post_init__(self):
        """Validate config."""
        pass


@dataclass
class AmygdalaState:
    """State of the amygdala at one timestep."""

    fear_level: float  # Current fear level (0-1)
    punishment_signal: float  # Current punishment signal
    threat_detected: bool  # Whether threat is currently detected
    fear_memory: jnp.ndarray  # Long-term fear associations (obs_dim,)


class Amygdala:
    """
    Amygdala module for fear processing and punishment signals.

    The amygdala:
    1. Detects threats in the current state
    2. Maintains fear memories of dangerous states
    3. Generates punishment signals to guide action selection
    4. Modulates learning based on emotional significance

    This module works closely with the Basal Ganglia (which processes
    reward via dopamine) to provide a balanced reward/punishment system.

    Attributes:
        config: AmygdalaConfig with all parameters
        obs_dim: Dimension of observation input
        act_dim: Dimension of action space
    """

    def __init__(
        self,
        obs_dim: int = 88,
        act_dim: int = 24,
        config: Optional[AmygdalaConfig] = None,
    ):
        """
        Initialize amygdala.

        Args:
            obs_dim: Dimension of observation input
            act_dim: Dimension of action space
            config: Optional AmygdalaConfig (defaults to standard values)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config if config is not None else AmygdalaConfig()

        # Threat detection network: observation -> threat score
        # Single layer for simplicity
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.w_threat = jax.random.normal(keys[0], (obs_dim, 1)) * 0.1
        self.b_threat = jnp.array(
            [-10.0]
        )  # Strong negative bias: zeros = near-zero threat

        # Fear memory: stores long-term fear associations
        # Initially negative so that dot(zeros, fear_memory) = 0 gives sigmoid(0) = 0.5
        # But we want neutral to be 0 contribution, so we subtract 0.5
        # Actually, let's just initialize to zeros and adjust the formula
        self.fear_memory = jnp.zeros(obs_dim)

        # Punishment predictor: predicts punishment from state-action
        self.w_punish = jax.random.normal(keys[1], (obs_dim + act_dim, 1)) * 0.1
        self.b_punish = jnp.zeros(1)

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset amygdala state."""
        self.state = AmygdalaState(
            fear_level=0.0,
            punishment_signal=0.0,
            threat_detected=False,
            fear_memory=self.fear_memory,
        )
        self._last_obs = None
        self._last_action = None

    def reset(self) -> None:
        """Reset amygdala to initial state."""
        self._reset_state()

    def detect_threat(
        self,
        observation: jnp.ndarray,
    ) -> Tuple[float, bool]:
        """
        Detect threats in the current observation.

        Args:
            observation: (obs_dim,) - current observation/state

        Returns:
            (threat_score, is_threat) - threat score (0-1) and boolean
        """
        # Compute threat score using threat detection network
        threat_score = jnp.dot(observation, self.w_threat) + self.b_threat
        threat_score = jax.nn.sigmoid(threat_score).item()  # Convert to 0-1

        # Add contribution from fear memory (states similar to feared states)
        # Use dot product directly - positive values indicate similarity to feared states
        memory_contribution = jnp.dot(observation, self.fear_memory)
        # Normalize and clamp
        memory_contribution = jnp.clip(memory_contribution, 0.0, 1.0).item()

        # Combined threat score
        combined_threat = 0.7 * threat_score + 0.3 * memory_contribution

        # Determine if threat is detected
        is_threat = combined_threat > self.config.fear_threshold

        return combined_threat, is_threat

    def update_fear_level(
        self,
        observation: jnp.ndarray,
        reward: float,
        punishment: Optional[float] = None,
    ) -> float:
        """
        Update the current fear level based on observations and outcomes.

        Fear increases with:
        - Detected threats
        - Negative rewards (punishment)
        - States similar to previously punished states

        Fear decreases with:
        - Safe observations
        - Positive rewards
        - Time (decay)

        Args:
            observation: (obs_dim,) - current observation
            reward: Current reward signal
            punishment: Optional explicit punishment signal

        Returns:
            float - updated fear level (0-1)
        """
        # Detect threat
        threat_score, is_threat = self.detect_threat(observation)

        # Base fear from threat detection
        fear = threat_score

        # Add fear from negative reward
        if reward < 0:
            fear += jnp.abs(reward) * 0.5

        # Reduce fear from positive reward (safety signal)
        if reward > 0:
            fear = fear * (1.0 - reward * 0.5)

        # Add explicit punishment
        if punishment is not None:
            fear += punishment * 0.3

        # Clamp and normalize fear
        fear = jnp.clip(fear, 0.0, 1.0).item()

        # Apply decay to existing fear
        # Fear decays exponentially toward 0, then new fear is added
        current_fear = self.state.fear_level
        decayed_fear = current_fear * self.config.fear_decay
        # New fear input replaces some portion of the decayed fear
        new_fear = decayed_fear + fear * 0.1
        new_fear = jnp.clip(new_fear, 0.0, 1.0).item()

        # Update state
        self.state = replace(
            self.state,
            fear_level=new_fear,
            threat_detected=is_threat,
        )

        return new_fear

    def compute_punishment_signal(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
    ) -> float:
        """
        Compute internal punishment signal based on current state and action.

        The punishment signal is used to:
        1. Bias action selection away from dangerous actions
        2. Modulate learning (punishment-based learning)
        3. Create avoidance behavior

        Args:
            observation: (obs_dim,) - current state
            action: (act_dim,) - action being considered
            reward: Current reward signal

        Returns:
            float - punishment signal (higher = more punishment)
        """
        # Predict punishment from state-action
        x = jnp.concatenate([observation, action])
        predicted_punishment = jnp.dot(x, self.w_punish) + self.b_punish
        predicted_punishment = jax.nn.sigmoid(predicted_punishment).item()

        # Fear-based punishment
        fear_punishment = self.state.fear_level * self.config.punishment_strength

        # Reward-based punishment (negative rewards)
        reward_punishment = max(0, -reward) * 0.5

        # Combined punishment signal
        punishment = (
            0.4 * predicted_punishment + 0.4 * fear_punishment + 0.2 * reward_punishment
        )

        # Update state
        self.state = replace(self.state, punishment_signal=punishment)

        return punishment

    def learn_from_punishment(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        punishment: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Learn from punishment experiences (fear conditioning).

        When a negative outcome occurs, the amygdala:
        1. Strengthens associations between the state and punishment
        2. Updates fear memory to remember this dangerous state
        3. Updates punishment predictor

        Args:
            observation: (obs_dim,) - state that led to punishment
            action: (act_dim,) - action that led to punishment
            reward: The reward received (negative = punishment)
            punishment: Optional explicit punishment signal
            learning_rate: Optional learning rate override
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate

        # Compute punishment magnitude
        if punishment is not None:
            punishment_magnitude = punishment
        else:
            punishment_magnitude = max(0, -reward)

        if punishment_magnitude <= 0:
            return  # No punishment to learn from

        # Update fear memory: increase association with this state
        # Hebbian learning: neurons that fire together, wire together
        update = observation * punishment_magnitude * learning_rate
        self.fear_memory = self.fear_memory + update

        # Update threat detection network
        threat_score, _ = self.detect_threat(observation)
        error = punishment_magnitude - threat_score
        grad_w = jnp.outer(observation, jnp.array([error]))
        self.w_threat = self.w_threat + learning_rate * grad_w
        self.b_threat = self.b_threat + learning_rate * error

        # Update punishment predictor
        x = jnp.concatenate([observation, action])
        predicted = jnp.dot(x, self.w_punish) + self.b_punish
        predicted = jax.nn.sigmoid(predicted)
        error_punish = punishment_magnitude - predicted
        # Gradient through sigmoid derivative
        sigmoid_deriv = predicted * (1 - predicted)
        grad_w_punish = jnp.outer(x, jnp.array([error_punish * sigmoid_deriv]))
        self.w_punish = self.w_punish + learning_rate * grad_w_punish
        self.b_punish = self.b_punish + learning_rate * error_punish * sigmoid_deriv

    def step(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        learning_rate: Optional[float] = None,
        punishment: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Complete amygdala processing step.

        This is the main method to call during training/inference:
        1. Detect threats in current state
        2. Update fear level
        3. Compute punishment signal
        4. Learn from punishment if applicable

        Args:
            observation: (obs_dim,) - current state
            action: (act_dim,) - current action
            reward: Current reward signal
            learning_rate: Optional learning rate override
            punishment: Optional explicit punishment signal

        Returns:
            (action_penalty, punishment_signal) -
                action_penalty: (act_dim,) vector to subtract from action values
                punishment_signal: scalar fear/punishment level
        """
        # Update fear level
        self.update_fear_level(observation, reward, punishment)

        # Compute punishment signal
        punishment_signal = self.compute_punishment_signal(observation, action, reward)

        # Learn from punishment
        self.learn_from_punishment(
            observation, action, reward, punishment, learning_rate
        )

        # Generate action penalty: punish actions that lead to dangerous states
        # Higher punishment = more suppression of all actions (freeze response)
        # Also penalize specific actions that led to punishment
        action_penalty = jnp.ones(self.act_dim) * punishment_signal * 0.1

        # If we have a specific punishment association with this action,
        # penalize it more (but we don't track per-action fear memory yet)

        return action_penalty, punishment_signal

    def get_fear_level(self) -> float:
        """Get current fear level."""
        return self.state.fear_level

    def get_punishment_signal(self) -> float:
        """Get current punishment signal."""
        return self.state.punishment_signal

    def is_threat_detected(self) -> bool:
        """Check if threat is currently detected."""
        return self.state.threat_detected


# -- Test Function --
def test_amygdala():
    """Test the Amygdala module."""
    print("Testing Amygdala module...")

    import jax.random as jr

    obs_dim = 88
    act_dim = 24
    amygdala = Amygdala(obs_dim=obs_dim, act_dim=act_dim)

    # Test 1: Initial state
    assert amygdala.state.fear_level == 0.0, "Initial fear should be 0"
    assert amygdala.state.punishment_signal == 0.0, "Initial punishment should be 0"
    assert amygdala.state.threat_detected == False, "No threat initially"
    print("  ✓ Initial state: all zeros, no threat")

    # Test 2: Threat detection
    key = jr.PRNGKey(42)
    obs = jr.normal(key, (obs_dim,)) * 0.1
    threat_score, is_threat = amygdala.detect_threat(obs)
    assert 0 <= threat_score <= 1, "Threat score should be in [0, 1]"
    print(f"  ✓ Threat detection: score={threat_score:.4f}, is_threat={is_threat}")

    # Test 3: Fear level update
    fear = amygdala.update_fear_level(obs, reward=-0.5)
    assert 0 <= fear <= 1, "Fear level should be in [0, 1]"
    assert fear > 0, "Fear should increase with negative reward"
    print(f"  ✓ Fear level update: fear={fear:.4f}")

    # Test 4: Punishment signal
    action = jr.normal(jr.PRNGKey(100), (act_dim,)) * 0.1
    punishment = amygdala.compute_punishment_signal(obs, action, reward=-0.5)
    assert punishment >= 0, "Punishment should be non-negative"
    print(f"  ✓ Punishment signal: {punishment:.4f}")

    # Test 5: Learning from punishment
    amygdala.learn_from_punishment(obs, action, reward=-1.0, punishment=0.5)
    # Fear memory should have been updated
    fear_mem_norm = jnp.linalg.norm(amygdala.fear_memory)
    assert fear_mem_norm > 0, "Fear memory should be non-zero after learning"
    print(f"  ✓ Learning from punishment: fear_memory_norm={fear_mem_norm:.4f}")

    # Test 6: Complete step
    amygdala.reset()
    action_penalty, punishment_signal = amygdala.step(
        obs, action, reward=-0.3, learning_rate=0.1
    )
    assert action_penalty.shape == (act_dim,), "Action penalty shape"
    assert action_penalty.shape == (act_dim,), "Action penalty should match action dim"
    assert punishment_signal >= 0, "Punishment signal should be non-negative"
    print(
        f"  ✓ Complete step: penalty_shape={action_penalty.shape}, punishment={punishment_signal:.4f}"
    )

    # Test 7: Fear increases with repeated punishment
    amygdala.reset()
    fear_before = amygdala.state.fear_level
    for _ in range(5):
        obs = jr.normal(jr.PRNGKey(200), (obs_dim,)) * 0.1
        action = jr.normal(jr.PRNGKey(201), (act_dim,)) * 0.1
        amygdala.step(obs, action, reward=-0.5, learning_rate=0.1)
    fear_after = amygdala.state.fear_level
    assert fear_after > fear_before, "Fear should increase with repeated punishment"
    print(f"  ✓ Fear accumulation: {fear_before:.4f} -> {fear_after:.4f}")

    # Test 8: Fear decay
    amygdala_decay = Amygdala(obs_dim=obs_dim, act_dim=act_dim)  # Fresh instance
    amygdala_decay.update_fear_level(obs, reward=-2.0)  # High fear
    fear1 = amygdala_decay.state.fear_level
    # Use a safe observation (zeros) with positive reward for decay test
    safe_obs = jnp.zeros(obs_dim)
    amygdala_decay.update_fear_level(safe_obs, reward=0.5)  # Positive reward = safety
    fear2 = amygdala_decay.state.fear_level
    assert fear2 < fear1, (
        f"Fear should decay with safe observations: {fear1:.4f} -> {fear2:.4f}"
    )
    print(f"  ✓ Fear decay: {fear1:.4f} -> {fear2:.4f}")

    # Test 9: Getters
    assert isinstance(amygdala.get_fear_level(), float), "Fear level getter"
    assert isinstance(amygdala.get_punishment_signal(), float), "Punishment getter"
    assert isinstance(amygdala.is_threat_detected(), bool), "Threat getter"
    print("  ✓ Getter methods: all working")

    # Test 10: Fear conditioning - threat detection improves
    amygdala.reset()
    dangerous_obs = jr.normal(jr.PRNGKey(300), (obs_dim,)) * 0.5  # A specific state

    # Before conditioning
    threat_before, _ = amygdala.detect_threat(dangerous_obs)

    # Condition with punishment
    for _ in range(10):
        amygdala.learn_from_punishment(
            dangerous_obs, action, reward=-1.0, learning_rate=0.1
        )

    # After conditioning
    threat_after, _ = amygdala.detect_threat(dangerous_obs)
    assert threat_after > threat_before, (
        "Threat detection should improve with conditioning"
    )
    print(f"  ✓ Fear conditioning: threat {threat_before:.4f} -> {threat_after:.4f}")

    print("✅ All Amygdala tests passed!")


if __name__ == "__main__":
    test_amygdala()
