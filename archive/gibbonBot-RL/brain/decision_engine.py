"""
Decision Engine — Multi-layer decision making with safety, risk, priorities, resources.

This module implements a hierarchical decision-making system that integrates
all brain modules to produce safe, optimal actions. It operates at multiple
timescales:

1. Reflex Layer (ms): Immediate safety responses (brainstem)
2. Reactive Layer (10-100ms): Emotional/avoidance responses (amygdala)
3. Deliberative Layer (100ms-1s): Goal-directed action selection (PFC + BG)
4. Strategic Layer (s+): Long-term planning and resource management

Safety Philosophy:
- Safety first: reflexes can override all other layers
- Risk-aware: amygdala modulates exploration based on threat level
- Priority-driven: PFC selects goals based on urgency and value
- Resource-conscious: allocate computational and physical resources wisely

Usage:
    from brain.decision_engine import DecisionEngine
    from brain.ape_brain import ApeBrain

    brain = ApeBrain()
    engine = DecisionEngine(brain)

    # During training/evaluation
    action = engine.decide(observation, reward, context)

    # Get diagnostics
    diagnostics = engine.get_diagnostics()
    print(f"Safety score: {diagnostics['safety_score']:.2f}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


class SafetyLevel(Enum):
    """Safety levels for action filtering."""

    CRITICAL = 0  # Immediate danger, override everything
    HIGH = 1  # High risk, use conservative actions
    MEDIUM = 2  # Moderate risk, proceed with caution
    LOW = 3  # Low risk, normal operation
    NONE = 4  # No detected risk


class DecisionLayer(Enum):
    """Decision layers in the hierarchy."""

    REFLEX = "reflex"  # Brainstem: immediate safety
    REACTIVE = "reactive"  # Amygdala: emotional/avoidance
    DELIBERATIVE = "deliberative"  # PFC + BG: goal-directed
    STRATEGIC = "strategic"  # Long-term planning


@dataclass
class ActionCandidate:
    """A candidate action with metadata for decision making."""

    action: jnp.ndarray  # (act_dim,) - the action vector
    source: DecisionLayer  # Which layer proposed this action
    confidence: float  # 0-1, how confident the layer is
    value: float  # Expected reward/value
    risk: float  # 0-1, estimated risk of this action
    cost: float  # Resource cost (energy, time, etc.)
    safety_score: float  # 0-1, safety assessment
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionDiagnostics:
    """Diagnostics and metrics about the decision process."""

    selected_action: int
    selected_source: DecisionLayer
    safety_level: SafetyLevel
    candidates_evaluated: int
    layers_active: List[DecisionLayer]
    reflex_triggers: List[str]
    risk_factors: List[str]
    resource_usage: Dict[str, float]
    timing_ms: Dict[str, float]

    # Safety metrics
    balance_confidence: float = 1.0
    joint_limit_violations: int = 0
    threat_detected: bool = False
    fear_level: float = 0.0

    # Risk metrics
    action_risk: float = 0.0
    uncertainty: float = 0.0

    # Resource metrics
    energy_available: float = 100.0
    energy_used: float = 0.0
    compute_time: float = 0.0


class DecisionEngine:
    """
    Hierarchical decision engine integrating all brain modules.

    The engine operates as follows:

    1. SAFETY CHECK: Brainstem reflexes are checked first. If any critical
       reflex is triggered (e.g., joint limit violation, severe imbalance),
       the reflex action is returned immediately.

    2. RISK ASSESSMENT: Amygdala evaluates the current state for threats.
       High fear levels can suppress exploration and bias toward safe actions.

    3. CANDIDATE GENERATION: Multiple brain modules propose actions:
       - Basal Ganglia: Value-based action selection
       - Prefrontal Cortex: Goal-directed actions
       - Cerebellum: Error-corrected actions
       - Motor Cortex: Smooth motor primitives

    4. CANDIDATE EVALUATION: Each candidate is scored on:
       - Value (expected reward)
       - Risk (probability of bad outcomes)
       - Cost (resource requirements)
       - Safety (immediate danger)

    5. ACTION SELECTION: The best candidate is selected based on:
       - Safety constraints (must meet minimum safety score)
       - Priority (task-specific goals)
       - Risk tolerance (adjustable based on mode)
       - Resource availability

    6. POST-PROCESSING: Final action is processed through:
       - Motor cortex smoothing
       - Cerebellum error correction
       - Brainstem reflex application
    """

    def __init__(
        self,
        brain: Any,
        risk_tolerance: float = 0.5,
        safety_threshold: float = 0.8,
        energy_capacity: float = 100.0,
        max_candidates: int = 10,
    ):
        """
        Initialize the decision engine.

        Args:
            brain: ApeBrain instance
            risk_tolerance: 0-1, higher = more risk-accepting
            safety_threshold: 0-1, minimum safety score for action
            energy_capacity: Total energy budget
            max_candidates: Maximum number of action candidates to evaluate
        """
        self.brain = brain
        self.risk_tolerance = risk_tolerance
        self.safety_threshold = safety_threshold
        self.energy_capacity = energy_capacity
        self.max_candidates = max_candidates

        # Resource tracking
        self.energy_available = energy_capacity
        self.resource_usage: Dict[str, float] = {}

        # Initialize modules
        self._init_modules()

        # Diagnostics
        self.last_diagnostics: Optional[DecisionDiagnostics] = None

    def _init_modules(self):
        """Cache module references for faster access."""
        self.brainstem = self.brain.modules.get("brainstem")
        self.amygdala = self.brain.modules.get("amygdala")
        self.basal_ganglia = self.brain.modules.get("basal_ganglia")
        self.prefrontal_cortex = self.brain.modules.get("prefrontal_cortex")
        self.cerebellum = self.brain.modules.get("cerebellum")
        self.motor_cortex = self.brain.modules.get("motor_cortex")
        self.hippocampus = self.brain.modules.get("hippocampus")
        self.neurotransmitters = self.brain.modules.get("neurotransmitters")

    def decide(
        self,
        observation: jnp.ndarray,
        reward: Optional[float] = None,
        context: Optional[jnp.ndarray] = None,
        task: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, DecisionDiagnostics]:
        """
        Make a decision using the multi-layer hierarchy.

        Args:
            observation: Current observation from environment
            reward: Current reward (optional)
            context: Additional context (optional)
            task: Current task name (optional)

        Returns:
            (action, diagnostics) - selected action and decision diagnostics
        """
        import time

        start_time = time.time()
        diagnostics = DecisionDiagnostics(
            selected_action=-1,
            selected_source=DecisionLayer.STRATEGIC,
            safety_level=SafetyLevel.NONE,
            candidates_evaluated=0,
            layers_active=[],
            reflex_triggers=[],
            risk_factors=[],
            resource_usage={},
            timing_ms={},
        )

        # =========================================================================
        # LAYER 1: REFLEX (Immediate Safety)
        # =========================================================================
        layer_start = time.time()

        reflex_action = None
        reflex_triggers = []

        if self.brainstem:
            # Extract components for brainstem processing
            torso_z = observation[0:1]
            torso_quat = observation[1:5]
            torso_vel = observation[5:8]
            torso_ang_vel = observation[8:11]
            joint_positions = observation[11:35]
            joint_velocities = observation[35:59]
            joint_torques = observation[59:83]

            # Process through brainstem
            _, bs_state = self.brainstem.preprocess_observation(
                observation,
                torso_quat=torso_quat,
                torso_pos=jnp.array([0.0, 0.0, float(torso_z[0])]),
                torso_vel=torso_vel,
                torso_ang_vel=torso_ang_vel,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_torques=joint_torques,
            )

            diagnostics.reflex_triggers = []
            diagnostics.balance_confidence = bs_state.vestibular.balance_confidence
            diagnostics.joint_limit_violations = int(
                jnp.sum(bs_state.reflexes.joint_limit_violation)
            )

            # Check for critical reflexes
            if bs_state.reflexes.joint_limit_violation.any():
                reflex_triggers.append("joint_limit")
                diagnostics.reflex_triggers.append("joint_limit")

            if not bs_state.vestibular.is_balanced:
                reflex_triggers.append("imbalance")
                diagnostics.reflex_triggers.append("imbalance")

            if bs_state.reflexes.withdrawal_active.any():
                reflex_triggers.append("withdrawal")
                diagnostics.reflex_triggers.append("withdrawal")

            # If critical reflex triggered, use reflex action immediately
            if reflex_triggers:
                diagnostics.safety_level = SafetyLevel.CRITICAL
                diagnostics.layers_active.append(DecisionLayer.REFLEX)

                # Get reflex-corrected action
                if bs_state.reflexes.corrected_action is not None:
                    reflex_action = bs_state.reflexes.corrected_action

                    # Ensure it's the right shape
                    if reflex_action.shape[0] == self.brain.config.act_dim:
                        # Apply motor cortex smoothing
                        if self.motor_cortex:
                            reflex_action = self.motor_cortex.step(reflex_action)

                        diagnostics.timing_ms["reflex"] = (
                            time.time() - layer_start
                        ) * 1000
                        diagnostics.selected_action = -1  # Reflex action
                        diagnostics.selected_source = DecisionLayer.REFLEX

                        return reflex_action, diagnostics

        diagnostics.timing_ms["reflex"] = (time.time() - layer_start) * 1000

        # =========================================================================
        # LAYER 2: REACTIVE (Emotional/Avoidance)
        # =========================================================================
        layer_start = time.time()
        diagnostics.layers_active.append(DecisionLayer.REACTIVE)

        fear_level = 0.0
        if self.amygdala:
            # Get fear level from amygdala
            fear_level = self.amygdala.get_fear_level()
            diagnostics.fear_level = fear_level
            diagnostics.threat_detected = self.amygdala.is_threat_detected()

            if fear_level > 0.8:
                diagnostics.safety_level = SafetyLevel.HIGH
                diagnostics.risk_factors.append("high_fear")
            elif fear_level > 0.5:
                diagnostics.safety_level = SafetyLevel.MEDIUM
                diagnostics.risk_factors.append("elevated_fear")

        diagnostics.timing_ms["reactive"] = (time.time() - layer_start) * 1000

        # =========================================================================
        # LAYER 3: DELIBERATIVE (Goal-Directed Action Selection)
        # =========================================================================
        layer_start = time.time()
        diagnostics.layers_active.append(DecisionLayer.DELIBERATIVE)

        # Generate action candidates from multiple sources
        candidates: List[ActionCandidate] = []

        # Candidate 1: Basal Ganglia (value-based)
        if self.basal_ganglia:
            bg_state = self.basal_ganglia.process_cortical_input(
                observation, dopamine=0.5
            )
            bg_action = jnp.zeros(self.brain.config.act_dim)
            bg_action = bg_action.at[bg_state.selected_action].set(1.0)
            bg_action = bg_action * 2.0 - 1.0

            bg_value = float(bg_state.action_values[bg_state.selected_action])
            candidates.append(
                ActionCandidate(
                    action=bg_action,
                    source=DecisionLayer.DELIBERATIVE,
                    confidence=0.9,
                    value=bg_value,
                    risk=0.1,
                    cost=1.0,
                    safety_score=0.9,
                    metadata={"module": "basal_ganglia"},
                )
            )

        # Candidate 2: Prefrontal Cortex (goal-directed)
        if self.prefrontal_cortex:
            pfc_action, pfc_confidence, _ = self.prefrontal_cortex.step(
                observation, generate_new_plan=False
            )
            candidates.append(
                ActionCandidate(
                    action=pfc_action,
                    source=DecisionLayer.DELIBERATIVE,
                    confidence=float(pfc_confidence),
                    value=0.7,
                    risk=0.2,
                    cost=1.5,
                    safety_score=0.8,
                    metadata={"module": "prefrontal_cortex"},
                )
            )

        # Candidate 3: Current action (momentum)
        # Use previous action if available
        if hasattr(self, "_last_action"):
            candidates.append(
                ActionCandidate(
                    action=self._last_action,
                    source=DecisionLayer.DELIBERATIVE,
                    confidence=0.7,
                    value=0.5,
                    risk=0.1,
                    cost=0.5,
                    safety_score=0.8,
                    metadata={"module": "momentum"},
                )
            )

        diagnostics.candidates_evaluated = len(candidates)

        # Evaluate candidates
        scored_candidates = self._score_candidates(candidates, diagnostics)

        # Select best candidate
        best_candidate = self._select_best_candidate(scored_candidates, diagnostics)

        # Store last action for momentum
        self._last_action = best_candidate.action

        diagnostics.selected_action = int(jnp.argmax(best_candidate.action))
        diagnostics.selected_source = best_candidate.source

        diagnostics.timing_ms["deliberative"] = (time.time() - layer_start) * 1000

        # =========================================================================
        # LAYER 4: STRATEGIC (Resource Management)
        # =========================================================================
        layer_start = time.time()
        diagnostics.layers_active.append(DecisionLayer.STRATEGIC)

        # Apply resource constraints
        final_action = self._apply_resource_constraints(
            best_candidate.action, diagnostics
        )

        # Apply post-processing
        final_action = self._post_process_action(final_action, observation, diagnostics)

        diagnostics.timing_ms["strategic"] = (time.time() - layer_start) * 1000

        # Update resource usage
        diagnostics.resource_usage = self.resource_usage
        diagnostics.compute_time = (time.time() - start_time) * 1000

        # Store diagnostics
        self.last_diagnostics = diagnostics

        return final_action, diagnostics

    def _score_candidates(
        self,
        candidates: List[ActionCandidate],
        diagnostics: DecisionDiagnostics,
    ) -> List[ActionCandidate]:
        """
        Score each candidate action on multiple dimensions.

        Args:
            candidates: List of action candidates
            diagnostics: Current diagnostics (for context)

        Returns:
            List of candidates with updated scores
        """
        scored = []

        for candidate in candidates:
            # Start with base score
            score = candidate.confidence * 0.4 + candidate.value * 0.6

            # Adjust for risk (based on risk tolerance)
            risk_penalty = candidate.risk * (1.0 - self.risk_tolerance) * 2.0
            score = score - risk_penalty

            # Adjust for cost (resource constraints)
            cost_penalty = candidate.cost * (
                1.0 - self.energy_available / self.energy_capacity
            )
            score = score - cost_penalty * 0.5

            # Adjust for safety
            safety_penalty = (
                (1.0 - candidate.safety_score) * (1.0 - self.safety_threshold) * 5.0
            )
            score = score - safety_penalty

            # Adjust for fear level
            if diagnostics.fear_level > 0:
                fear_penalty = diagnostics.fear_level * candidate.risk * 2.0
                score = score - fear_penalty

            # Store scored candidate
            scored.append(candidate)

            # Update diagnostics
            if candidate.risk > 0.7:
                diagnostics.risk_factors.append(
                    f"high_risk_action_{candidate.source.value}"
                )

        return scored

    def _select_best_candidate(
        self,
        candidates: List[ActionCandidate],
        diagnostics: DecisionDiagnostics,
    ) -> ActionCandidate:
        """
        Select the best candidate based on scores and constraints.

        Args:
            candidates: List of scored candidates
            diagnostics: Current diagnostics

        Returns:
            Best action candidate
        """
        if not candidates:
            # Return neutral action
            return ActionCandidate(
                action=jnp.zeros(self.brain.config.act_dim),
                source=DecisionLayer.REFLEX,
                confidence=0.0,
                value=0.0,
                risk=0.0,
                cost=0.0,
                safety_score=1.0,
            )

        # Filter by minimum safety
        safe_candidates = [
            c for c in candidates if c.safety_score >= self.safety_threshold
        ]

        if not safe_candidates:
            # If no candidates meet safety threshold, use the safest one
            safe_candidates = sorted(
                candidates, key=lambda c: c.safety_score, reverse=True
            )[:1]

        # Select the highest scoring candidate
        best = max(
            safe_candidates,
            key=lambda c: (
                c.confidence * 0.4
                + c.value * 0.6
                - c.risk * (1.0 - self.risk_tolerance) * 2.0
                - c.cost * 0.5
            ),
        )

        # Update action risk in diagnostics
        diagnostics.action_risk = best.risk

        return best

    def _apply_resource_constraints(
        self,
        action: jnp.ndarray,
        diagnostics: DecisionDiagnostics,
    ) -> jnp.ndarray:
        """
        Apply resource constraints to the action.

        Args:
            action: Proposed action
            diagnostics: Current diagnostics

        Returns:
            Action with resource constraints applied
        """
        # Energy cost: proportional to action magnitude
        energy_cost = jnp.linalg.norm(action) * 0.1

        # Check if we have enough energy
        if self.energy_available < energy_cost:
            # Scale down action proportionally
            scale = self.energy_available / (energy_cost + 1e-8)
            action = action * jnp.clip(scale, 0.0, 1.0)

            # Record resource usage
            self.resource_usage["energy_conservation"] = (
                self.resource_usage.get("energy_conservation", 0) + 1
            )

        # Update energy
        self.energy_available = jnp.clip(
            self.energy_available - energy_cost, 0.0, self.energy_capacity
        )

        diagnostics.energy_used = energy_cost
        diagnostics.energy_available = self.energy_available

        return action

    def _post_process_action(
        self,
        action: jnp.ndarray,
        observation: jnp.ndarray,
        diagnostics: DecisionDiagnostics,
    ) -> jnp.ndarray:
        """
        Apply post-processing to the final action.

        Args:
            action: Action to post-process
            observation: Current observation
            diagnostics: Current diagnostics

        Returns:
            Post-processed action
        """
        # Apply cerebellum error correction
        if self.cerebellum:
            action = self.cerebellum.step(observation, action)

        # Apply motor cortex smoothing
        if self.motor_cortex:
            intention = jnp.zeros(self.motor_cortex.config.primitive_dim)
            mc_action = self.motor_cortex.step(intention, use_primitives=False)
            action = action * 0.7 + mc_action * 0.3

        # Apply brainstem reflex corrections
        if self.brainstem:
            action = self.brainstem.apply_reflex_correction(action)

        # Clamp to valid range
        action = jnp.clip(action, -1.0, 1.0)

        return action

    def replenish_energy(self, amount: float) -> None:
        """Add energy to the resource budget."""
        self.energy_available = jnp.clip(
            self.energy_available + amount, 0.0, self.energy_capacity
        )

    def set_risk_tolerance(self, tolerance: float) -> None:
        """Set risk tolerance (0=conservative, 1=aggressive)."""
        self.risk_tolerance = jnp.clip(tolerance, 0.0, 1.0)

    def set_safety_threshold(self, threshold: float) -> None:
        """Set minimum safety score for actions."""
        self.safety_threshold = jnp.clip(threshold, 0.0, 1.0)

    def get_diagnostics(self) -> Optional[DecisionDiagnostics]:
        """Get the last decision diagnostics."""
        return self.last_diagnostics

    def reset(self) -> None:
        """Reset the decision engine state."""
        self.energy_available = self.energy_capacity
        self.resource_usage = {}
        self.last_diagnostics = None
        if hasattr(self, "_last_action"):
            del self._last_action


# =============================================================================
# DECISION STRATEGIES
# =============================================================================


class DecisionStrategy:
    """Base class for decision strategies."""

    def __init__(self, engine: DecisionEngine):
        self.engine = engine

    def decide(self, observation: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, Any]:
        """Make a decision using this strategy."""
        raise NotImplementedError


class SafeStrategy(DecisionStrategy):
    """Conservative strategy that prioritizes safety."""

    def __init__(self, engine: DecisionEngine):
        super().__init__(engine)
        engine.set_safety_threshold(0.95)
        engine.set_risk_tolerance(0.1)

    def decide(self, observation: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, Any]:
        return self.engine.decide(observation, **kwargs)


class BalancedStrategy(DecisionStrategy):
    """Balanced strategy with moderate risk tolerance."""

    def __init__(self, engine: DecisionEngine):
        super().__init__(engine)
        engine.set_safety_threshold(0.8)
        engine.set_risk_tolerance(0.5)

    def decide(self, observation: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, Any]:
        return self.engine.decide(observation, **kwargs)


class AggressiveStrategy(DecisionStrategy):
    """Aggressive strategy that takes more risks for higher rewards."""

    def __init__(self, engine: DecisionEngine):
        super().__init__(engine)
        engine.set_safety_threshold(0.5)
        engine.set_risk_tolerance(0.9)

    def decide(self, observation: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, Any]:
        return self.engine.decide(observation, **kwargs)


class EnergyConservingStrategy(DecisionStrategy):
    """Strategy that conserves energy."""

    def __init__(self, engine: DecisionEngine):
        super().__init__(engine)
        engine.set_risk_tolerance(0.5)
        engine.set_safety_threshold(0.7)

    def decide(self, observation: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, Any]:
        action, diagnostics = self.engine.decide(observation, **kwargs)
        # Scale down actions to conserve energy
        scale = 0.5 + 0.5 * (self.engine.energy_available / self.engine.energy_capacity)
        return action * scale, diagnostics


# =============================================================================
# TESTS
# =============================================================================


def test_decision_engine() -> None:
    """Test the DecisionEngine."""
    print("Testing DecisionEngine...")

    from brain.ape_brain import ApeBrain, ApeBrainConfig

    # Create brain and engine
    brain = ApeBrain(
        ApeBrainConfig(obs_dim=88, act_dim=24, training=False, persist_dir=None)
    )
    engine = DecisionEngine(brain)

    # Create test observation
    import jax.random as jr

    key = jr.PRNGKey(42)
    obs = jr.normal(key, (88,)) * 0.1

    # Test decide
    action, diagnostics = engine.decide(obs, reward=1.0, task="test")

    # Verify action
    assert action.shape == (24,), f"Action shape: {action.shape}"
    assert jnp.all(jnp.abs(action) <= 1.0), "Action out of range"

    # Verify diagnostics
    assert diagnostics is not None
    assert isinstance(diagnostics, DecisionDiagnostics)
    assert diagnostics.candidates_evaluated > 0
    assert len(diagnostics.layers_active) > 0

    # Test strategies
    safe_strategy = SafeStrategy(engine)
    balanced_strategy = BalancedStrategy(engine)
    aggressive_strategy = AggressiveStrategy(engine)

    for strategy in [safe_strategy, balanced_strategy, aggressive_strategy]:
        action, _ = strategy.decide(obs)
        assert action.shape == (24,)

    # Test energy management
    engine.replenish_energy(50.0)
    assert engine.energy_available <= engine.energy_capacity

    # Test reset
    engine.reset()
    assert engine.energy_available == engine.energy_capacity

    print("✅ DecisionEngine tests passed")


def test_decision_with_all_modules() -> None:
    """Test DecisionEngine with all brain modules active."""
    print("Testing DecisionEngine with all modules...")

    import tempfile

    from brain.ape_brain import ApeBrain, ApeBrainConfig

    tmp = tempfile.mkdtemp()
    brain = ApeBrain(
        ApeBrainConfig(obs_dim=88, act_dim=24, training=False, persist_dir=tmp)
    )
    engine = DecisionEngine(brain)

    # Create observation with realistic values
    import jax.random as jr

    key = jr.PRNGKey(42)
    obs = jr.normal(key, (88,)) * 0.1

    # Test with reward
    action, diagnostics = engine.decide(obs, reward=1.0)
    assert action.shape == (24,)

    # Test with negative reward (should trigger some risk assessment)
    action, diagnostics = engine.decide(obs, reward=-1.0)
    assert action.shape == (24,)

    print("✅ DecisionEngine with all modules tests passed")


if __name__ == "__main__":
    test_decision_engine()
    test_decision_with_all_modules()
