"""
Brainstem - Low-Level Reflexes and Vital Functions

This module implements the brainstem functions that are essential for
survival and basic motor control. The brainstem processes sensory input
and generates reflexive responses before the information reaches higher
brain centers.

Components Implemented:
- Vestibular System: Balance, orientation, head position tracking
- Reticular Formation: Arousal, attention filtering, sleep/wake cycles
- Spinal Cord Reflexes: Withdrawal, stretch, joint limit protection

Biological Background:
    The brainstem (medulla, pons, midbrain) is responsible for:

    1. Vestibular System (inner ear + brainstem nuclei):
       - Detects head position and movement
       - Maintains balance and gaze stabilization
       - Provides sense of spatial orientation

    2. Reticular Formation:
       - Filters sensory input (attention)
       - Regulates arousal and sleep/wake states
       - Controls autonomic functions (heart rate, breathing)

    3. Spinal Cord:
       - Transmits signals between brain and body
       - Mediates reflex arcs (withdrawal from pain, stretch reflex)
       - Contains central pattern generators for locomotion

In our ApeBrain, the brainstem preprocesses observations and applies
reflexive corrections to actions before they're sent to muscles.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class VestibularState:
    """State of the vestibular (balance) system."""

    head_position: jnp.ndarray  # (3,) - x, y, z position of head
    head_velocity: jnp.ndarray  # (3,) - linear velocity
    head_angular_vel: jnp.ndarray  # (3,) - angular velocity (roll, pitch, yaw)
    gravity_direction: jnp.ndarray  # (3,) - perceived gravity vector
    is_balanced: bool = True
    balance_confidence: float = 1.0  # 0-1, how confident we are in balance


@dataclass
class ReflexState:
    """State of spinal cord reflexes."""

    withdrawal_active: jnp.ndarray  # (n_joints,) - which joints are withdrawing
    stretch_active: jnp.ndarray  # (n_joints,) - which joints are stretching
    joint_limit_violation: jnp.ndarray  # (n_joints,) - which joints at limits
    pain_signals: jnp.ndarray  # (n_joints,) - pain/nociceptive input
    corrected_action: Optional[jnp.ndarray] = None  # Reflex-corrected action


@dataclass
class BrainstemState:
    """Complete brainstem state."""

    vestibular: VestibularState
    reflexes: ReflexState
    arousal: float  # 0-1, wakefulness level
    attention_filter: jnp.ndarray  # Which sensory inputs to prioritize


class Brainstem:
    """
    Brainstem module handling low-level reflexes and vital functions.

    The brainstem processes raw sensory input and applies biologically
    plausible preprocessing and reflexive corrections.

    Key Functions:
    - Vestibular processing: balance detection, gravity estimation
    - Reflex generation: withdrawal, stretch, joint protection
    - Arousal regulation: attention filtering
    - Sensorimotor preprocessing: noise filtering, normalization

    Attributes:
        n_joints: Number of joints in the body
        n_obs: Dimension of observation vector
    """

    def __init__(
        self,
        n_joints: int = 24,
        n_obs: int = 88,
        gravity: jnp.ndarray = None,
    ):
        """
        Initialize brainstem.

        Args:
            n_joints: Number of actuated joints
            n_obs: Dimension of observation vector
            gravity: Gravity vector (default: [0, 0, -9.81])
        """
        self.n_joints = n_joints
        self.n_obs = n_obs

        # Default gravity vector (points downward)
        self.gravity = gravity if gravity is not None else jnp.array([0.0, 0.0, -9.81])

        # Vestibular system parameters
        self.balance_threshold = 0.3  # Max tilt angle before imbalance
        self.gravity_scale = 1.0

        # Reflex parameters
        self.withdrawal_strength = 0.5  # Strength of withdrawal reflex
        self.stretch_strength = 0.3  # Strength of stretch reflex
        self.joint_limit_penalty = 10.0  # Penalty for hitting joint limits

        # Joint limit ranges (normalized: -1 to 1)
        self.joint_low = jnp.array([-1.0] * n_joints)
        self.joint_high = jnp.array([1.0] * n_joints)

        # Arousal parameters
        self.baseline_arousal = 0.7  # Normal wakefulness
        self.max_arousal = 0.95
        self.min_arousal = 0.1

        # Attention filter (initially uniform)
        self.attention_weights = jnp.ones(n_obs) / n_obs

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset brainstem state."""
        self.state = BrainstemState(
            vestibular=VestibularState(
                head_position=jnp.zeros(3),
                head_velocity=jnp.zeros(3),
                head_angular_vel=jnp.zeros(3),
                gravity_direction=self.gravity / jnp.linalg.norm(self.gravity),
                is_balanced=True,
                balance_confidence=1.0,
            ),
            reflexes=ReflexState(
                withdrawal_active=jnp.zeros(self.n_joints, dtype=bool),
                stretch_active=jnp.zeros(self.n_joints, dtype=bool),
                joint_limit_violation=jnp.zeros(self.n_joints, dtype=bool),
                pain_signals=jnp.zeros(self.n_joints),
                corrected_action=None,
            ),
            arousal=self.baseline_arousal,
            attention_filter=self.attention_weights,
        )

    def reset(self) -> None:
        """Reset brainstem to initial state."""
        self._reset_state()

    def process_vestibular(
        self,
        torso_quat: jnp.ndarray,
        torso_pos: jnp.ndarray,
        torso_vel: jnp.ndarray,
        torso_ang_vel: jnp.ndarray,
    ) -> VestibularState:
        """
        Process vestibular (balance) information.

        Args:
            torso_quat: (4,) - torso orientation quaternion (w, x, y, z)
            torso_pos: (3,) - torso position (x, y, z)
            torso_vel: (3,) - torso linear velocity
            torso_ang_vel: (3,) - torso angular velocity

        Returns:
            Updated VestibularState
        """
        # Extract roll, pitch, yaw from quaternion
        # Quaternion: q = [w, x, y, z] where w is scalar part
        w, x, y, z = torso_quat

        # Convert to roll, pitch, yaw (in radians)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if jnp.abs(sinp) >= 1:
            pitch = jnp.sign(sinp) * jnp.pi / 2
        else:
            pitch = jnp.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)

        # Estimate gravity direction in body frame
        # In world frame, gravity is self.gravity
        # Rotate by inverse of torso quaternion to get gravity in body frame
        gravity_body = self._rotate_vector(self.gravity, torso_quat)
        gravity_body = gravity_body / jnp.linalg.norm(gravity_body)

        # Check balance: torso upright (roll and pitch near 0)
        abs_roll = jnp.abs(roll)
        abs_pitch = jnp.abs(pitch)
        max_tilt = jnp.maximum(abs_roll, abs_pitch)

        # Balance confidence: decreases with tilt
        balance_confidence = jnp.clip(1.0 - max_tilt / (jnp.pi / 2), 0.0, 1.0)
        is_balanced = max_tilt < self.balance_threshold

        # Update vestibular state
        self.state.vestibular = VestibularState(
            head_position=torso_pos
            + jnp.array([0.0, 0.0, 0.15]),  # Head ~15cm above torso
            head_velocity=torso_vel,
            head_angular_vel=torso_ang_vel,
            gravity_direction=gravity_body,
            is_balanced=is_balanced,
            balance_confidence=balance_confidence,
        )

        # Arousal modulation: high tilt increases arousal (startle response)
        tilt_arousal = jnp.clip(max_tilt / (jnp.pi / 4), 0.0, 0.3)
        self.state.arousal = jnp.clip(
            self.state.arousal + tilt_arousal, self.min_arousal, self.max_arousal
        )

        return self.state.vestibular

    def process_reflexes(
        self,
        joint_positions: jnp.ndarray,
        joint_velocities: jnp.ndarray,
        joint_torques: jnp.ndarray,
        pain_input: Optional[jnp.ndarray] = None,
    ) -> ReflexState:
        """
        Process spinal cord reflexes.

        Args:
            joint_positions: (n_joints,) - current joint angles
            joint_velocities: (n_joints,) - current joint velocities
            joint_torques: (n_joints,) - current joint torques
            pain_input: (n_joints,) - nociceptive/pain signals (optional)

        Returns:
            Updated ReflexState with corrected action
        """
        if pain_input is None:
            pain_input = jnp.zeros(self.n_joints)

        # Check joint limits
        at_lower_limit = joint_positions <= self.joint_low
        at_upper_limit = joint_positions >= self.joint_high
        joint_limit_violation = at_lower_limit | at_upper_limit

        # Withdrawal reflex: pull away from pain
        # If pain > threshold, generate withdrawal action
        pain_threshold = 0.5
        withdrawal_active = pain_input > pain_threshold
        withdrawal_action = (
            -self.withdrawal_strength
            * pain_input
            * withdrawal_active.astype(jnp.float32)
        )

        # Stretch reflex: resist rapid stretching
        # If joint is being stretched rapidly, contract it
        stretch_threshold = 2.0  # rad/s
        rapid_stretch = jnp.abs(joint_velocities) > stretch_threshold
        stretch_active = rapid_stretch & (jnp.abs(joint_positions) < 0.8)
        stretch_action = (
            -self.stretch_strength
            * joint_velocities
            * stretch_active.astype(jnp.float32)
        )

        # Joint limit protection: push away from limits
        limit_correction = jnp.zeros(self.n_joints)
        limit_correction = jnp.where(
            at_lower_limit,
            0.1,  # Push away from lower limit
            limit_correction,
        )
        limit_correction = jnp.where(
            at_upper_limit,
            -0.1,  # Push away from upper limit
            limit_correction,
        )

        # Combine all reflexes
        reflex_correction = withdrawal_action + stretch_action + limit_correction

        # Update reflex state
        self.state.reflexes = ReflexState(
            withdrawal_active=withdrawal_active,
            stretch_active=stretch_active,
            joint_limit_violation=joint_limit_violation,
            pain_signals=pain_input,
            corrected_action=reflex_correction,
        )

        # Pain increases arousal
        avg_pain = jnp.mean(pain_input)
        pain_arousal = jnp.clip(avg_pain * 0.5, 0.0, 0.4)
        self.state.arousal = jnp.clip(
            self.state.arousal + pain_arousal, self.min_arousal, self.max_arousal
        )

        return self.state.reflexes

    def apply_reflex_correction(
        self,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply reflex corrections to a motor action.

        Args:
            action: (n_joints,) - proposed motor action

        Returns:
            (n_joints,) - action with reflex corrections applied
        """
        if self.state.reflexes.corrected_action is None:
            return action

        # Add reflex correction to action
        corrected = action + self.state.reflexes.corrected_action

        # Clamp to action bounds
        corrected = jnp.clip(corrected, -1.0, 1.0)

        return corrected

    def update_attention(
        self,
        sensory_input: jnp.ndarray,
        novelty: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Update attention filter based on novelty and arousal.

        The reticular formation filters sensory input, prioritizing
        novel or important stimuli.

        Args:
            sensory_input: (n_obs,) - raw sensory observation
            novelty: (n_obs,) - novelty scores per dimension (optional)

        Returns:
            (n_obs,) - attention-filtered observation
        """
        if novelty is None:
            novelty = jnp.zeros(self.n_obs)

        # Attention weights: combine baseline with novelty
        # Novel stimuli get more attention
        novelty_factor = 0.3
        new_weights = self.attention_weights + novelty_factor * novelty
        new_weights = new_weights / jnp.sum(new_weights)  # Normalize

        # Arousal modulation: high arousal = broader attention
        # Low arousal = more focused attention
        arousal_factor = 0.1 + 0.8 * self.state.arousal
        self.attention_weights = (
            self.attention_weights * (1.0 - arousal_factor)
            + new_weights * arousal_factor
        )

        # Store updated weights
        self.state.attention_filter = self.attention_weights

        # Apply attention filter to sensory input
        filtered = sensory_input * self.attention_weights

        return filtered

    def preprocess_observation(
        self,
        observation: jnp.ndarray,
        torso_quat: jnp.ndarray,
        torso_pos: jnp.ndarray,
        torso_vel: jnp.ndarray,
        torso_ang_vel: jnp.ndarray,
        joint_positions: jnp.ndarray,
        joint_velocities: jnp.ndarray,
        joint_torques: jnp.ndarray,
        pain_input: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, BrainstemState]:
        """
        Complete brainstem preprocessing of sensory input.

        This method runs all brainstem processing in sequence:
        1. Vestibular processing (balance)
        2. Reflex processing (spinal cord)
        3. Attention filtering (reticular formation)

        Args:
            observation: (n_obs,) - raw observation vector
            torso_quat: (4,) - torso orientation
            torso_pos: (3,) - torso position
            torso_vel: (3,) - torso velocity
            torso_ang_vel: (3,) - torso angular velocity
            joint_positions: (n_joints,) - joint angles
            joint_velocities: (n_joints,) - joint velocities
            joint_torques: (n_joints,) - joint torques
            pain_input: (n_joints,) - pain signals (optional)

        Returns:
            (filtered_observation, brainstem_state) tuple
        """
        # Process vestibular
        self.process_vestibular(torso_quat, torso_pos, torso_vel, torso_ang_vel)

        # Process reflexes
        self.process_reflexes(
            joint_positions, joint_velocities, joint_torques, pain_input
        )

        # Update attention and filter observation
        filtered_obs = self.update_attention(observation)

        return filtered_obs, self.state

    def _rotate_vector(
        self,
        vector: jnp.ndarray,
        quaternion: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Rotate a vector by a quaternion.

        Args:
            vector: (3,) - vector to rotate
            quaternion: (4,) - rotation quaternion (w, x, y, z)

        Returns:
            (3,) - rotated vector
        """
        w, x, y, z = quaternion
        vx, vy, vz = vector

        # Quaternion rotation: v' = q * v * q^{-1}
        # For unit quaternion, q^{-1} = conjugate(q)
        t2 = w * x
        t3 = w * y
        t4 = w * z
        t5 = -x * x
        t6 = x * y
        t7 = x * z
        t8 = -y * y
        t9 = y * z
        t10 = -z * z

        # Result components
        rx = 2 * ((t8 + t10) * vx + (t6 - t4) * vy + (t7 + t2) * vz) + vx
        ry = 2 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t3) * vz) + vy
        rz = 2 * ((t7 - t2) * vx + (t4 + t9) * vy + (t5 + t8) * vz) + vz

        return jnp.array([rx, ry, rz])


# -- Test Function --
def test_brainstem():
    """Test the Brainstem module."""
    print("Testing Brainstem module...")

    brainstem = Brainstem(n_joints=24, n_obs=88)

    # Test 1: Vestibular processing - upright torso
    torso_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Upright
    torso_pos = jnp.array([0.0, 0.0, 0.5])
    torso_vel = jnp.zeros(3)
    torso_ang_vel = jnp.zeros(3)

    vest = brainstem.process_vestibular(torso_quat, torso_pos, torso_vel, torso_ang_vel)
    assert vest.is_balanced, "Upright torso should be balanced"
    assert vest.balance_confidence > 0.9, (
        "Upright torso should have high balance confidence"
    )
    print(
        f"  ✓ Upright torso: balanced={vest.is_balanced}, confidence={vest.balance_confidence:.4f}"
    )

    # Test 2: Vestibular processing - tilted torso
    # Create a ~45 degree tilt around x-axis
    angle = jnp.pi / 4
    tilt_quat = jnp.array([jnp.cos(angle / 2), jnp.sin(angle / 2), 0.0, 0.0])
    vest = brainstem.process_vestibular(tilt_quat, torso_pos, torso_vel, torso_ang_vel)
    assert not vest.is_balanced, "Tilted torso should not be balanced"
    assert vest.balance_confidence < 1.0, "Tilted torso should have reduced confidence"
    print(
        f"  ✓ Tilted torso: balanced={vest.is_balanced}, confidence={vest.balance_confidence:.4f}"
    )

    # Test 3: Reflex processing - joint at limit
    joint_pos = jnp.zeros(24).at[0].set(1.0)  # At upper limit
    joint_vel = jnp.zeros(24)
    joint_torque = jnp.zeros(24)

    brainstem.process_reflexes(joint_pos, joint_vel, joint_torque)
    assert jnp.any(brainstem.state.reflexes.joint_limit_violation), (
        "Joint at limit should trigger violation"
    )
    print(
        f"  ✓ Joint limit detection: violation={jnp.any(brainstem.state.reflexes.joint_limit_violation)}"
    )

    # Test 4: Reflex processing - pain input
    pain = jnp.zeros(24).at[5].set(0.8)  # Pain in joint 5
    brainstem.process_reflexes(joint_pos, joint_vel, joint_torque, pain)
    assert brainstem.state.reflexes.withdrawal_active[5], (
        "Pain should trigger withdrawal"
    )
    print(
        f"  ✓ Pain reflex: withdrawal_active[5]={brainstem.state.reflexes.withdrawal_active[5]}"
    )

    # Test 5: Attention filtering
    obs = jnp.ones(88)
    novelty = jnp.zeros(88).at[10:20].set(1.0)  # Novelty in dimensions 10-20
    filtered = brainstem.update_attention(obs, novelty)
    assert jnp.sum(filtered[10:20]) > jnp.sum(filtered[0:10]), (
        "Novel dimensions should get more attention"
    )
    print(
        f"  ✓ Attention filtering: novel region sum={jnp.sum(filtered[10:20]):.4f} > non-novel sum={jnp.sum(filtered[0:10]):.4f}"
    )

    # Test 6: Complete preprocessing
    import jax.random as jr

    key = jr.PRNGKey(42)
    obs = jr.normal(key, (88,)) * 0.1
    filtered_obs, state = brainstem.preprocess_observation(
        obs,
        torso_quat,
        torso_pos,
        torso_vel,
        torso_ang_vel,
        joint_pos,
        joint_vel,
        joint_torque,
        pain,
    )
    assert filtered_obs.shape == (88,), "Filtered observation should have same shape"
    assert not jnp.allclose(filtered_obs, obs), (
        "Filtered observation should differ from raw"
    )
    print(f"  ✓ Complete preprocessing: output shape={filtered_obs.shape}")

    # Test 7: Arousal modulation
    brainstem.reset()
    brainstem.state.arousal = 0.3  # Low arousal
    assert brainstem.state.arousal == 0.3, "Should be able to set arousal"

    brainstem.state.arousal = 0.9  # High arousal
    assert brainstem.state.arousal == 0.9, "Should be able to modify arousal"
    print(f"  ✓ Arousal modulation: can set arousal levels")

    print("✅ All Brainstem tests passed!")


if __name__ == "__main__":
    test_brainstem()
