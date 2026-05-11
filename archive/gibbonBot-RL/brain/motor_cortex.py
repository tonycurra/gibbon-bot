"""
Motor Cortex - Motor Command Generation and Execution

This module implements a biologically-inspired motor cortex for generating
and executing motor commands. The motor cortex is the primary output system
of the brain, translating high-level intentions into specific muscle activations.

Biological Background:
    The motor cortex consists of several regions:

    1. **Primary Motor Cortex (M1)**: Direct motor command output
    2. **Premotor Cortex**: Movement planning and sequencing
    3. **Supplementary Motor Area (SMA)**: Complex movement coordination
    4. **Somatosensory Cortex**: Sensory feedback for motor control

    The motor cortex:
    - Generates precise motor commands for voluntary movements
    - Organizes movements in space and time
    - Integrates sensory feedback for fine motor control
    - Works with the cerebellum for error correction
    - Receives input from basal ganglia for action selection

Simplification:
    This implementation focuses on the core motor functions:
    1. Motor command generation from high-level intentions
    2. Action execution with timing and coordination
    3. Integration of corrections from the cerebellum
    4. Motor primitive composition (building complex movements from simples)
    5. Force/velocity control for smooth movements
"""

from dataclasses import dataclass, replace
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class MotorCortexConfig:
    """Configuration for the motor cortex module."""

    motor_primitives: int = 8  # Number of learned motor primitives
    primitive_dim: int = 16  # Dimension of motor primitive encoding
    output_gain: float = 1.0  # Gain for motor output
    smoothing_factor: float = 0.3  # Smoothing factor for action sequences
    min_activation: float = 0.05  # Minimum activation threshold
    max_activation: float = 1.0  # Maximum activation
    fatigue_decay: float = 0.99  # Decay rate for motor fatigue

    def __post_init__(self):
        """Validate config."""
        pass


@dataclass
class MotorPrimitive:
    """A learned motor primitive (basic movement pattern)."""

    name: str  # Name/description of the primitive
    activation_pattern: jnp.ndarray  # Pattern of muscle activations
    duration: int  # Typical duration in timesteps
    amplitude: float  # Typical amplitude


@dataclass
class MotorCortexState:
    """State of the motor cortex at one timestep."""

    current_action: jnp.ndarray  # Current motor command being executed
    previous_action: jnp.ndarray  # Previous motor command (for smoothing)
    motor_activation: jnp.ndarray  # Activation levels for each motor primitive
    fatigue: jnp.ndarray  # Fatigue levels for each muscle/group
    movement_phase: float  # Phase of current movement (0-1)


class MotorCortex:
    """
    Motor Cortex module for motor command generation and execution.

    The motor cortex:
    1. Translates high-level intentions into specific motor commands
    2. Executes movements with proper timing and coordination
    3. Integrates corrections from the cerebellum for accuracy
    4. Composes complex movements from motor primitives
    5. Manages motor fatigue and recovery

    This module works with other brain regions by:
    - Receiving high-level action intentions from Prefrontal Cortex
    - Receiving action selections from Basal Ganglia
    - Receiving corrections from Cerebellum
    - Receiving inhibition signals from Amygdala
    - Sending motor commands to the robot's actuators

    Attributes:
        config: MotorCortexConfig with all parameters
        obs_dim: Dimension of observation input
        act_dim: Dimension of action space
    """

    def __init__(
        self,
        obs_dim: int = 88,
        act_dim: int = 24,
        config: Optional[MotorCortexConfig] = None,
    ):
        """
        Initialize motor cortex.

        Args:
            obs_dim: Dimension of observation input
            act_dim: Dimension of action space
            config: Optional MotorCortexConfig (defaults to standard values)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config if config is not None else MotorCortexConfig()

        # Motor primitives (learned basic movement patterns)
        self.primitives: List[MotorPrimitive] = []

        # Initialize with some default primitives
        self._initialize_default_primitives()

        # Intention to action mapping: intention (from PFC) -> action
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        # Map from intention/context to primitive activations
        self.w_intention_to_primitives = (
            jax.random.normal(
                keys[0], (self.config.primitive_dim, self.config.motor_primitives)
            )
            * 0.1
        )
        self.b_intention_to_primitives = jnp.zeros(self.config.motor_primitives)

        # Map from primitive activations to motor commands
        self.w_primitives_to_action = (
            jax.random.normal(keys[1], (self.config.motor_primitives, act_dim)) * 0.1
        )

        # Direct mapping from intention to action (bypass primitives)
        self.w_intention_to_action = (
            jax.random.normal(keys[2], (self.config.primitive_dim, act_dim)) * 0.1
        )
        self.b_intention_to_action = jnp.zeros(act_dim)

        # Initialize state
        self._reset_state()

    def _initialize_default_primitives(self) -> None:
        """Initialize a set of default motor primitives."""
        # Create primitives for basic movements
        primitive_names = [
            "reach_forward",
            "reach_backward",
            "reach_left",
            "reach_right",
            "grasp",
            "release",
            "rotate_cw",
            "rotate_ccw",
        ]

        key = jax.random.PRNGKey(100)
        for i, name in enumerate(primitive_names[: self.config.motor_primitives]):
            activation = jax.random.normal(key, (self.act_dim,)) * 0.3
            primitive = MotorPrimitive(
                name=name,
                activation_pattern=activation,
                duration=5,
                amplitude=0.5,
            )
            self.primitives.append(primitive)

    def _reset_state(self) -> None:
        """Reset motor cortex state."""
        self.state = MotorCortexState(
            current_action=jnp.zeros(self.act_dim),
            previous_action=jnp.zeros(self.act_dim),
            motor_activation=jnp.zeros(self.config.motor_primitives),
            fatigue=jnp.zeros(self.act_dim),
            movement_phase=0.0,
        )

    def reset(self) -> None:
        """Reset motor cortex to initial state."""
        self._reset_state()

    def add_primitive(
        self,
        name: str,
        activation_pattern: jnp.ndarray,
        duration: int = 5,
        amplitude: float = 0.5,
    ) -> None:
        """
        Add a new motor primitive.

        Args:
            name: Name of the primitive
            activation_pattern: (act_dim,) - pattern of muscle activations
            duration: Typical duration in timesteps
            amplitude: Typical amplitude
        """
        primitive = MotorPrimitive(
            name=name,
            activation_pattern=activation_pattern,
            duration=duration,
            amplitude=amplitude,
        )
        self.primitives.append(primitive)

    def activate_primitives(self, intention: jnp.ndarray) -> jnp.ndarray:
        """
        Activate motor primitives based on intention signal.

        Args:
            intention: (primitive_dim,) - intention signal from PFC/Basal Ganglia

        Returns:
            (motor_primitives,) - activation levels for each primitive
        """
        # Map intention to primitive activations
        activations = (
            jnp.dot(intention, self.w_intention_to_primitives)
            + self.b_intention_to_primitives
        )
        activations = jnp.tanh(activations)  # Normalize to [-1, 1]

        # Apply minimum activation threshold
        activations = jnp.clip(
            activations, self.config.min_activation, self.config.max_activation
        )

        # Update state
        self.state = replace(self.state, motor_activation=activations)

        return activations

    def compose_action_from_primitives(
        self, primitive_activations: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compose a motor action from activated primitives.

        Args:
            primitive_activations: (motor_primitives,) - activation levels

        Returns:
            (act_dim,) - composed motor action
        """
        # Weight each primitive's activation pattern by its activation level
        action = jnp.zeros(self.act_dim)
        for i, primitive in enumerate(self.primitives[: self.config.motor_primitives]):
            action += primitive_activations[i] * primitive.activation_pattern

        # Normalize and scale
        action = action / (jnp.abs(primitive_activations).sum() + 1e-8)
        action = action * self.config.output_gain

        return action

    def generate_action(
        self,
        intention: jnp.ndarray,
        use_primitives: bool = True,
    ) -> jnp.ndarray:
        """
        Generate a motor action from an intention signal.

        Args:
            intention: (primitive_dim,) - intention signal
            use_primitives: Whether to use motor primitives or direct mapping

        Returns:
            (act_dim,) - motor action
        """
        if use_primitives and self.config.motor_primitives > 0:
            # Use primitive-based generation
            activations = self.activate_primitives(intention)
            action = self.compose_action_from_primitives(activations)
        else:
            # Direct mapping from intention to action
            action = (
                jnp.dot(intention, self.w_intention_to_action)
                + self.b_intention_to_action
            )
            action = jnp.tanh(action) * self.config.output_gain

        # Apply fatigue effects
        action = self.apply_fatigue(action)

        # Smooth action based on previous action
        if self.state.previous_action is not None:
            action = (
                self.config.smoothing_factor * self.state.previous_action
                + (1 - self.config.smoothing_factor) * action
            )

        # Clamp to valid range
        action = jnp.clip(action, -1.0, 1.0)

        # Update state
        self.state = replace(
            self.state,
            previous_action=self.state.current_action,
            current_action=action,
        )

        return action

    def apply_fatigue(self, action: jnp.ndarray) -> jnp.ndarray:
        """
        Apply motor fatigue to action.

        Fatigued muscles produce less force.

        Args:
            action: (act_dim,) - proposed action

        Returns:
            (act_dim,) - action with fatigue applied
        """
        # Apply fatigue: reduce action magnitude for fatigued muscles
        fatigued_action = action * (1.0 - self.state.fatigue)

        # Update fatigue: muscles fatigue when used
        self.state = replace(
            self.state,
            fatigue=self.state.fatigue * self.config.fatigue_decay
            + jnp.abs(action) * (1 - self.config.fatigue_decay),
        )

        return fatigued_action

    def apply_correction(
        self, action: jnp.ndarray, correction: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply a correction to a motor action (from cerebellum).

        Args:
            action: (act_dim,) - original action
            correction: (act_dim,) - correction vector

        Returns:
            (act_dim,) - corrected action
        """
        corrected_action = action + correction
        corrected_action = jnp.clip(corrected_action, -1.0, 1.0)
        return corrected_action

    def apply_inhibition(self, action: jnp.ndarray, inhibition: float) -> jnp.ndarray:
        """
        Apply inhibition to a motor action (from amygdala/PFC).

        Args:
            action: (act_dim,) - original action
            inhibition: float - inhibition signal (0-1)

        Returns:
            (act_dim,) - inhibited action
        """
        # Inhibition reduces the magnitude of all actions
        inhibited_action = action * (1.0 - inhibition)
        return inhibited_action

    def step(
        self,
        intention: jnp.ndarray,
        correction: Optional[jnp.ndarray] = None,
        inhibition: Optional[float] = None,
        use_primitives: bool = True,
    ) -> jnp.ndarray:
        """
        Complete motor cortex processing step.

        This is the main method to call during training/inference:
        1. Generate action from intention
        2. Apply corrections (from cerebellum)
        3. Apply inhibition (from amygdala/PFC)
        4. Update internal state

        Args:
            intention: (primitive_dim,) - intention signal from PFC/Basal Ganglia
            correction: (act_dim,) - optional correction from cerebellum
            inhibition: float - optional inhibition signal
            use_primitives: Whether to use motor primitives

        Returns:
            (act_dim,) - final motor action
        """
        # Generate base action
        action = self.generate_action(intention, use_primitives)

        # Apply correction
        if correction is not None:
            action = self.apply_correction(action, correction)

        # Apply inhibition
        if inhibition is not None:
            action = self.apply_inhibition(action, inhibition)

        # Update phase
        self.state = replace(
            self.state, movement_phase=(self.state.movement_phase + 0.1) % 1.0
        )

        return action

    def get_current_action(self) -> jnp.ndarray:
        """Get the current motor action."""
        return self.state.current_action

    def get_primitive_activations(self) -> jnp.ndarray:
        """Get current primitive activation levels."""
        return self.state.motor_activation

    def get_fatigue(self) -> jnp.ndarray:
        """Get current fatigue levels."""
        return self.state.fatigue

    def get_primitives(self) -> List[MotorPrimitive]:
        """Get all motor primitives."""
        return self.primitives

    def reset_fatigue(self) -> None:
        """Reset motor fatigue."""
        self.state = replace(self.state, fatigue=jnp.zeros(self.act_dim))


# -- Test Function --
def test_motor_cortex():
    """Test the Motor Cortex module."""
    print("Testing Motor Cortex module...")

    import jax.random as jr

    obs_dim = 88
    act_dim = 24
    motor_cortex = MotorCortex(obs_dim=obs_dim, act_dim=act_dim)

    # Test 1: Initial state
    assert motor_cortex.state.current_action.shape == (act_dim,)
    assert motor_cortex.state.previous_action.shape == (act_dim,)
    assert motor_cortex.state.motor_activation.shape == (
        motor_cortex.config.motor_primitives,
    )
    assert motor_cortex.state.fatigue.shape == (act_dim,)
    assert len(motor_cortex.get_primitives()) > 0, "Should have default primitives"
    print(
        f"  ✓ Initial state: shapes correct, {len(motor_cortex.get_primitives())} primitives"
    )

    # Test 2: Action generation with primitives
    key = jr.PRNGKey(42)
    intention = jr.normal(key, (motor_cortex.config.primitive_dim,)) * 0.5
    action = motor_cortex.generate_action(intention, use_primitives=True)
    assert action.shape == (act_dim,)
    assert jnp.all(jnp.abs(action) <= 1.0), "Action should be in [-1, 1]"
    print(f"  ✓ Action generation with primitives: shape={action.shape}")

    # Test 3: Action generation without primitives (direct)
    motor_cortex.reset()
    action = motor_cortex.generate_action(intention, use_primitives=False)
    assert action.shape == (act_dim,)
    assert jnp.all(jnp.abs(action) <= 1.0), "Action should be in [-1, 1]"
    print(f"  ✓ Action generation without primitives: shape={action.shape}")

    # Test 4: Primitive activation
    activations = motor_cortex.activate_primitives(intention)
    assert activations.shape == (motor_cortex.config.motor_primitives,)
    print(f"  ✓ Primitive activation: shape={activations.shape}")

    # Test 5: Action composition from primitives
    action = motor_cortex.compose_action_from_primitives(activations)
    assert action.shape == (act_dim,)
    print(f"  ✓ Action composition: shape={action.shape}")

    # Test 6: Fatigue application
    motor_cortex.reset()
    motor_cortex.reset_fatigue()
    action_before = motor_cortex.generate_action(intention)
    # Apply fatigue by running multiple steps
    for _ in range(10):
        motor_cortex.generate_action(intention)
    action_after = motor_cortex.generate_action(intention)
    # Actions should be different due to fatigue
    # (fatigue reduces magnitude)
    assert not jnp.allclose(action_before, action_after), (
        "Fatigue should affect actions"
    )
    print(
        f"  ✓ Fatigue: before norm={jnp.linalg.norm(action_before):.4f}, after norm={jnp.linalg.norm(action_after):.4f}"
    )

    # Test 7: Correction application
    motor_cortex.reset()
    action = motor_cortex.generate_action(intention)
    correction = jr.normal(jr.PRNGKey(100), (act_dim,)) * 0.2
    corrected = motor_cortex.apply_correction(action, correction)
    assert corrected.shape == (act_dim,)
    assert jnp.all(jnp.abs(corrected) <= 1.0), "Corrected action should be in [-1, 1]"
    print(f"  ✓ Correction application: shape={corrected.shape}")

    # Test 8: Inhibition application
    inhibited = motor_cortex.apply_inhibition(action, inhibition=0.5)
    assert inhibited.shape == (act_dim,)
    # Inhibition should reduce magnitude
    assert jnp.linalg.norm(inhibited) < jnp.linalg.norm(action), (
        "Inhibition should reduce magnitude"
    )
    print(
        f"  ✓ Inhibition application: norm {jnp.linalg.norm(action):.4f} -> {jnp.linalg.norm(inhibited):.4f}"
    )

    # Test 9: Complete step
    motor_cortex.reset()
    action = motor_cortex.step(intention, correction=correction, inhibition=0.3)
    assert action.shape == (act_dim,)
    assert jnp.all(jnp.abs(action) <= 1.0)
    print(
        f"  ✓ Complete step: shape={action.shape}, norm={jnp.linalg.norm(action):.4f}"
    )

    # Test 10: Add primitive
    initial_count = len(motor_cortex.get_primitives())
    new_pattern = jr.normal(jr.PRNGKey(200), (act_dim,)) * 0.5
    motor_cortex.add_primitive(
        "test_primitive", new_pattern, duration=10, amplitude=0.8
    )
    assert len(motor_cortex.get_primitives()) == initial_count + 1
    print(f"  ✓ Add primitive: {initial_count} -> {len(motor_cortex.get_primitives())}")

    # Test 11: Getters
    assert isinstance(motor_cortex.get_current_action(), jnp.ndarray), (
        "Current action getter"
    )
    assert isinstance(motor_cortex.get_primitive_activations(), jnp.ndarray), (
        "Primitive activations getter"
    )
    assert isinstance(motor_cortex.get_fatigue(), jnp.ndarray), "Fatigue getter"
    assert isinstance(motor_cortex.get_primitives(), list), "Primitives getter"
    print("  ✓ Getter methods: all working")

    # Test 12: Reset fatigue
    motor_cortex.reset_fatigue()
    assert jnp.all(motor_cortex.get_fatigue() == 0), "Fatigue should be reset to 0"
    print("  ✓ Reset fatigue: all zeros")

    print("✅ All Motor Cortex tests passed!")


if __name__ == "__main__":
    test_motor_cortex()
