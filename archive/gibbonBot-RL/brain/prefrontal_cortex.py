"""
Prefrontal Cortex - Executive Function and Planning

This module implements a biologically-inspired prefrontal cortex (PFC) for
executive functions including planning, decision-making, working memory,
and goal-directed behavior.

Biological Background:
    The prefrontal cortex is the most recently evolved part of the brain and
    is responsible for:

    1. **Dorsolateral PFC**: Working memory, cognitive flexibility, planning
    2. **Ventromedial PFC**: Emotional regulation, value-based decision making
    3. **Orbitofrontal Cortex**: Reward/punishment evaluation, outcome prediction
    4. **Anterior Cingulate Cortex**: Conflict monitoring, error detection, motivation

    The PFC:
    - Maintains goals and plans over time
    - Integrates information from sensory, memory, and emotional systems
    - Makes decisions based on long-term consequences
    - Inhibits automatic/impulsive responses when inappropriate
    - Coordinates complex, sequential behaviors

Simplification:
    This implementation focuses on the core executive functions:
    1. Goal maintenance and representation
    2. Action planning (sequence generation)
    3. Working memory for short-term information
    4. Decision making integrating reward/punishment signals
    5. Cognitive control (overriding automatic responses)
"""

from dataclasses import dataclass, replace
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class PrefrontalCortexConfig:
    """Configuration for the prefrontal cortex module."""

    plan_horizon: int = 5  # How many steps to look ahead in planning
    working_memory_dim: int = 32  # Dimension of working memory
    goal_strength: float = 1.0  # Strength of goal-directed behavior
    inhibition_strength: float = 0.5  # Strength of inhibitory control
    learning_rate: float = 0.05  # Learning rate for value predictions
    temperature: float = 1.0  # Temperature for softmax decision making

    def __post_init__(self):
        """Validate config."""
        pass


@dataclass
class Goal:
    """Represents a goal or subgoal."""

    description: str  # Human-readable description
    target_observation: Optional[
        jnp.ndarray
    ]  # Desired observation (e.g., standing pose)
    target_features: Optional[jnp.ndarray]  # Desired features to achieve
    reward_bonus: float  # Bonus reward for achieving this goal
    priority: float  # Goal priority (higher = more important)
    achieved: bool  # Whether the goal has been achieved


@dataclass
class Plan:
    """Represents a sequence of planned actions."""

    actions: List[jnp.ndarray]  # Sequence of actions
    predicted_rewards: List[float]  # Predicted rewards for each step
    predicted_values: List[float]  # Predicted long-term values
    confidence: float  # Confidence in this plan (0-1)


@dataclass
class PrefrontalCortexState:
    """State of the prefrontal cortex at one timestep."""

    current_goal: Optional[Goal]  # Current active goal
    working_memory: jnp.ndarray  # Current working memory contents
    plan: Optional[Plan]  # Current plan being executed
    plan_step: int  # Current step in the plan
    inhibition_signal: float  # Current inhibitory control signal
    conflict_detected: bool  # Whether conflict between systems is detected


class PrefrontalCortex:
    """
    Prefrontal Cortex module for executive function and planning.

    The PFC:
    1. Maintains and prioritizes goals
    2. Generates and evaluates action plans
    3. Maintains working memory of relevant information
    4. Integrates inputs from other brain regions for decision making
    5. Provides cognitive control to override automatic responses

    This module works with other brain regions by:
    - Receiving contextual information from the Hippocampus
    - Receiving reward/punishment signals from Basal Ganglia and Amygdala
    - Receiving motor error information from Cerebellum
    - Sending action biases to Motor Cortex
    - Coordinate high-level behavior

    Attributes:
        config: PrefrontalCortexConfig with all parameters
        obs_dim: Dimension of observation input
        act_dim: Dimension of action space
    """

    def __init__(
        self,
        obs_dim: int = 88,
        act_dim: int = 24,
        config: Optional[PrefrontalCortexConfig] = None,
    ):
        """
        Initialize prefrontal cortex.

        Args:
            obs_dim: Dimension of observation input
            act_dim: Dimension of action space
            config: Optional PrefrontalCortexConfig (defaults to standard values)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config if config is not None else PrefrontalCortexConfig()

        # Goal stack (prioritized list of goals)
        self.goals: List[Goal] = []

        # Working memory network: observation + context -> working_memory
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        self.w_working_memory = (
            jax.random.normal(keys[0], (obs_dim, self.config.working_memory_dim)) * 0.1
        )
        self.b_working_memory = jnp.zeros(self.config.working_memory_dim)

        # Value prediction: observation -> predicted value
        self.w_value = jax.random.normal(keys[1], (obs_dim, 1)) * 0.1
        self.b_value = jnp.zeros(1)

        # Action value prediction: observation + action -> predicted value
        self.w_action_value = jax.random.normal(keys[2], (obs_dim + act_dim, 1)) * 0.1
        self.b_action_value = jnp.zeros(1)

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset prefrontal cortex state."""
        self.state = PrefrontalCortexState(
            current_goal=None,
            working_memory=jnp.zeros(self.config.working_memory_dim),
            plan=None,
            plan_step=0,
            inhibition_signal=0.0,
            conflict_detected=False,
        )

    def reset(self) -> None:
        """Reset prefrontal cortex to initial state."""
        self._reset_state()

    def set_goal(
        self,
        description: str = "",
        target_observation: Optional[jnp.ndarray] = None,
        target_features: Optional[jnp.ndarray] = None,
        reward_bonus: float = 1.0,
        priority: float = 1.0,
    ) -> None:
        """
        Set a new goal.

        Args:
            description: Human-readable description of the goal
            target_observation: Desired observation to achieve
            target_features: Desired features (alternative to full observation)
            reward_bonus: Bonus reward for achieving this goal
            priority: Goal priority (higher = more important)
        """
        goal = Goal(
            description=description,
            target_observation=target_observation,
            target_features=target_features,
            reward_bonus=reward_bonus,
            priority=priority,
            achieved=False,
        )

        # Add to goal stack, maintaining priority order
        self.goals.append(goal)
        # Sort by priority (highest first)
        self.goals.sort(key=lambda g: g.priority, reverse=True)

        # Set as current goal if it's the highest priority
        if (
            self.state.current_goal is None
            or goal.priority > self.state.current_goal.priority
        ):
            self.state = replace(self.state, current_goal=goal)

    def clear_goals(self) -> None:
        """Clear all goals."""
        self.goals = []
        self.state = replace(self.state, current_goal=None)

    def update_working_memory(
        self,
        observation: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Update working memory with current information.

        Working memory holds information relevant to the current task,
        including recent observations, goals, and contextual information.

        Args:
            observation: (obs_dim,) - current observation
            context: (spatial_dim,) - optional contextual information from hippocampus

        Returns:
            (working_memory_dim,) - updated working memory
        """
        # Encode observation into working memory
        wm_obs = jnp.dot(observation, self.w_working_memory) + self.b_working_memory
        wm_obs = jnp.tanh(wm_obs)

        # If context is provided, integrate it
        if context is not None:
            # Simple integration: add context to working memory
            # In a more sophisticated implementation, we'd have learned weights
            wm_context = (
                context[: self.config.working_memory_dim]
                if context.shape[0] >= self.config.working_memory_dim
                else jnp.pad(
                    context, (0, self.config.working_memory_dim - context.shape[0])
                )
            )
            working_memory = 0.7 * wm_obs + 0.3 * wm_context
        else:
            working_memory = wm_obs

        # Update state
        self.state = replace(self.state, working_memory=working_memory)

        return working_memory

    def predict_value(self, observation: jnp.ndarray) -> float:
        """
        Predict the long-term value of a state.

        Args:
            observation: (obs_dim,) - observation to evaluate

        Returns:
            float - predicted value
        """
        value = jnp.dot(observation, self.w_value) + self.b_value
        return value.item()

    def predict_action_value(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> float:
        """
        Predict the value of a specific action in a specific state.

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - action to evaluate

        Returns:
            float - predicted action value
        """
        x = jnp.concatenate([observation, action])
        value = jnp.dot(x, self.w_action_value) + self.b_action_value
        return value.item()

    def generate_plan(
        self,
        observation: jnp.ndarray,
        horizon: Optional[int] = None,
    ) -> Optional[Plan]:
        """
        Generate a plan (sequence of actions) to achieve the current goal.

        This uses a simplified planning approach:
        1. For each possible action, predict the next state and value
        2. Select the action with highest predicted value
        3. Repeat for the planning horizon

        Args:
            observation: (obs_dim,) - current observation
            horizon: Number of steps to plan (defaults to config.plan_horizon)

        Returns:
            Plan - the generated plan, or None if no goal is set
        """
        if horizon is None:
            horizon = self.config.plan_horizon

        if self.state.current_goal is None:
            return None

        # Simplified planning: just pick the best immediate action
        # In a more sophisticated implementation, we'd use a model of the environment
        # to simulate future states and do proper tree search

        current_obs = observation
        actions = []
        predicted_rewards = []
        predicted_values = []

        for _ in range(horizon):
            # For each possible action dimension, try positive and negative
            # This is a very simplified approach
            best_action = None
            best_value = -jnp.inf

            # Sample a few random actions (in a real implementation, we'd use the policy)
            key = jax.random.PRNGKey(42)
            for i in range(10):
                action = jax.random.normal(key, (self.act_dim,)) * 0.5
                action = jnp.clip(action, -1.0, 1.0)

                # Predict value of this action
                value = self.predict_action_value(current_obs, action)

                # Also consider goal proximity
                if self.state.current_goal.target_observation is not None:
                    # How close would this get us to the goal?
                    # (Simplified: just use observation similarity)
                    similarity = self._compute_observation_similarity(
                        current_obs, self.state.current_goal.target_observation
                    )
                    value += (
                        self.config.goal_strength
                        * similarity
                        * self.state.current_goal.reward_bonus
                    )

                if value > best_value:
                    best_value = value
                    best_action = action

            if best_action is not None:
                actions.append(best_action)
                predicted_rewards.append(0.0)  # Placeholder
                predicted_values.append(best_value)

                # Update current observation (simplified: no actual dynamics model)
                current_obs = (
                    current_obs  # In real implementation, we'd predict next state
                )
            else:
                break

        if len(actions) == 0:
            return None

        # Create plan
        plan = Plan(
            actions=actions,
            predicted_rewards=predicted_rewards,
            predicted_values=predicted_values,
            confidence=0.5,  # Placeholder
        )

        return plan

    def _compute_observation_similarity(
        self, obs1: jnp.ndarray, obs2: jnp.ndarray
    ) -> float:
        """Compute similarity between two observations using cosine similarity."""
        obs1_norm = obs1 / (jnp.linalg.norm(obs1) + 1e-8)
        obs2_norm = obs2 / (jnp.linalg.norm(obs2) + 1e-8)
        cosine_sim = jnp.dot(obs1_norm, obs2_norm).item()
        return (cosine_sim + 1.0) / 2.0

    def check_goal_achievement(self, observation: jnp.ndarray) -> bool:
        """
        Check if the current goal has been achieved.

        Args:
            observation: (obs_dim,) - current observation

        Returns:
            bool - True if goal is achieved
        """
        if self.state.current_goal is None:
            return False

        goal = self.state.current_goal

        if goal.target_observation is not None:
            similarity = self._compute_observation_similarity(
                observation, goal.target_observation
            )
            # Goal is achieved if similarity is high
            if similarity > 0.9:
                goal.achieved = True
                return True

        if goal.target_features is not None:
            # Check if target features match
            # Simplified: just check if the dot product is high
            similarity = self._compute_observation_similarity(
                observation[: goal.target_features.shape[0]], goal.target_features
            )
            if similarity > 0.9:
                goal.achieved = True
                return True

        return False

    def decide_action(
        self,
        observation: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        action_values: Optional[jnp.ndarray] = None,
        use_plan: bool = True,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Decide which action to take based on current information.

        This integrates:
        1. Current goal
        2. Working memory
        3. Predicted action values
        4. Any existing plan

        Args:
            observation: (obs_dim,) - current observation
            context: (spatial_dim,) - optional contextual information
            action_values: (act_dim,) - optional pre-computed action values
            use_plan: Whether to use the current plan if available

        Returns:
            (action, confidence) - selected action and confidence score
        """
        # Update working memory
        self.update_working_memory(observation, context)

        # Check if we have a plan and should use it
        if use_plan and self.state.plan is not None:
            if self.state.plan_step < len(self.state.plan.actions):
                action = self.state.plan.actions[self.state.plan_step]
                confidence = self.state.plan.confidence
                # Move to next step in plan
                self.state = replace(self.state, plan_step=self.state.plan_step + 1)
                return action, confidence

        # Otherwise, select action based on values
        if action_values is not None:
            # Use provided action values
            # Apply softmax to get probabilities
            if self.config.temperature > 0:
                probabilities = jax.nn.softmax(action_values / self.config.temperature)
            else:
                # Greedy selection
                probabilities = jnp.zeros(self.act_dim)
                probabilities = probabilities.at[jnp.argmax(action_values)].set(1.0)

            # Sample from probabilities
            key = jax.random.PRNGKey(42)
            action_idx = jax.random.categorical(
                key, jnp.log(probabilities + 1e-8), shape=(1,)
            )[0]

            # Create one-hot-like action (simplified)
            action = jnp.zeros(self.act_dim)
            action = action.at[action_idx].set(1.0)
            # Scale to [-1, 1] range
            action = action * 2.0 - 1.0

            confidence = probabilities[action_idx]
            return action, confidence

        # If no action values provided, return zeros
        action = jnp.zeros(self.act_dim)
        return action, 0.0

    def compute_inhibition(
        self,
        automatic_action: jnp.ndarray,
        deliberate_action: jnp.ndarray,
        conflict_threshold: float = 0.5,
    ) -> float:
        """
                Compute inhibition signal when there's conflict between automatic and
        deliberate action selection systems.

                Args:
                    automatic_action: Action from automatic/habit system
                    deliberate_action: Action from deliberate/planning system
                    conflict_threshold: Threshold for detecting conflict

                Returns:
                    float - inhibition signal (0-1, higher = more inhibition)
        """
        # Compute difference between actions
        action_diff = jnp.linalg.norm(automatic_action - deliberate_action)

        # Normalize difference to [0, 1] based on maximum possible difference
        max_diff = jnp.linalg.norm(
            jnp.ones(self.act_dim) - jnp.ones(self.act_dim) * -1.0
        )
        normalized_diff = action_diff / (max_diff + 1e-8)

        # Conflict detected if difference is significant
        conflict = normalized_diff > conflict_threshold

        # Inhibition strength increases with conflict
        inhibition = jnp.clip(
            normalized_diff * self.config.inhibition_strength, 0.0, 1.0
        ).item()

        # Update state
        self.state = replace(
            self.state,
            inhibition_signal=inhibition,
            conflict_detected=conflict,
        )

        return inhibition

    def integrate_recall(self, recalled_episodes) -> Optional[jnp.ndarray]:
        """Fold k recalled episodes into a single context vector.

        Recency-weighted mean of the episodes' embeddings, scaled by valence
        magnitude so emotionally salient episodes dominate the recall context.
        Returns None if no episodes were recalled.

        Args:
            recalled_episodes: list of brain.episodic_store.Episode

        Returns:
            (spatial_dim,) jnp array or None
        """
        if not recalled_episodes:
            return None
        embs = jnp.stack(
            [jnp.asarray(ep.embedding) for ep in recalled_episodes], axis=0
        )
        weights = jnp.asarray(
            [abs(ep.valence) + 1e-3 for ep in recalled_episodes],
            dtype=embs.dtype,
        )
        weights = weights / weights.sum()
        return (embs * weights[:, None]).sum(axis=0)

    def step(
        self,
        observation: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        action_values: Optional[jnp.ndarray] = None,
        generate_new_plan: bool = True,
        recalled_episodes: Optional[list] = None,
    ) -> Tuple[jnp.ndarray, float, float]:
        """
        Complete prefrontal cortex processing step.

        This is the main method to call during training/inference:
        1. Update working memory
        2. Check goal achievement
        3. Generate new plan if needed
        4. Decide on action

        Args:
            observation: (obs_dim,) - current observation
            context: (spatial_dim,) - optional contextual information
            action_values: (act_dim,) - optional pre-computed action values
            generate_new_plan: Whether to generate a new plan this step

        Returns:
            (action, confidence, inhibition) - selected action, confidence, inhibition signal
        """
        # Fold episodic recall (long-horizon hippocampal context) into the
        # context passed to working memory — RAG-for-robots.
        recall_ctx = self.integrate_recall(recalled_episodes) if recalled_episodes else None
        if recall_ctx is not None:
            context = recall_ctx if context is None else 0.5 * context + 0.5 * recall_ctx

        # Update working memory
        self.update_working_memory(observation, context)

        # Check goal achievement
        goal_achieved = self.check_goal_achievement(observation)
        if goal_achieved and self.state.current_goal is not None:
            # Move to next goal
            if len(self.goals) > 1:
                self.goals.pop(0)  # Remove achieved goal
                self.state = replace(self.state, current_goal=self.goals[0])
            else:
                self.goals = []
                self.state = replace(self.state, current_goal=None)

        # Generate new plan if needed
        if generate_new_plan:
            plan = self.generate_plan(observation)
            if plan is not None:
                self.state = replace(
                    self.state,
                    plan=plan,
                    plan_step=0,
                )

        # Decide on action
        action, confidence = self.decide_action(observation, context, action_values)

        # Return action, confidence, and current inhibition signal
        return action, confidence, self.state.inhibition_signal

    def get_current_goal(self) -> Optional[Goal]:
        """Get the current active goal."""
        return self.state.current_goal

    def get_working_memory(self) -> jnp.ndarray:
        """Get the current working memory contents."""
        return self.state.working_memory

    def get_inhibition_signal(self) -> float:
        """Get the current inhibition signal."""
        return self.state.inhibition_signal

    def get_goals(self) -> List[Goal]:
        """Get all goals."""
        return self.goals


# -- Test Function --
def test_prefrontal_cortex():
    """Test the Prefrontal Cortex module."""
    print("Testing Prefrontal Cortex module...")

    import jax.random as jr

    obs_dim = 88
    act_dim = 24
    pfc = PrefrontalCortex(obs_dim=obs_dim, act_dim=act_dim)

    # Test 1: Initial state
    assert pfc.state.current_goal is None
    assert pfc.state.working_memory.shape == (pfc.config.working_memory_dim,)
    assert len(pfc.get_goals()) == 0
    print("  ✓ Initial state: no goals, empty working memory")

    # Test 2: Set goal
    key = jr.PRNGKey(42)
    target_obs = jr.normal(key, (obs_dim,)) * 0.5
    pfc.set_goal(
        description="Stand up",
        target_observation=target_obs,
        reward_bonus=10.0,
        priority=1.0,
    )
    assert pfc.state.current_goal is not None
    assert pfc.state.current_goal.description == "Stand up"
    assert len(pfc.get_goals()) == 1
    print("  ✓ Set goal: current goal set, in goal list")

    # Test 3: Working memory update
    obs = jr.normal(jr.PRNGKey(100), (obs_dim,)) * 0.1
    wm = pfc.update_working_memory(obs)
    assert wm.shape == (pfc.config.working_memory_dim,)
    assert jnp.all(jnp.abs(wm) <= 1.0), "Working memory should be in [-1, 1]"
    print(
        f"  ✓ Working memory update: shape={wm.shape}, range=[{wm.min():.2f}, {wm.max():.2f}]"
    )

    # Test 4: Value prediction
    value = pfc.predict_value(obs)
    assert isinstance(value, float), "Value should be float"
    print(f"  ✓ Value prediction: {value:.4f}")

    # Test 5: Action value prediction
    action = jr.normal(jr.PRNGKey(200), (act_dim,)) * 0.5
    action_value = pfc.predict_action_value(obs, action)
    assert isinstance(action_value, float), "Action value should be float"
    print(f"  ✓ Action value prediction: {action_value:.4f}")

    # Test 6: Plan generation
    plan = pfc.generate_plan(obs, horizon=3)
    assert plan is not None, "Should generate plan"
    assert len(plan.actions) == 3, (
        f"Plan should have 3 actions, got {len(plan.actions)}"
    )
    print(
        f"  ✓ Plan generation: {len(plan.actions)} actions, confidence={plan.confidence:.4f}"
    )

    # Test 7: Goal achievement check
    # Should not be achieved with random observation
    achieved = pfc.check_goal_achievement(obs)
    assert not achieved, "Random obs should not achieve goal"

    # Should be achieved with target observation
    achieved = pfc.check_goal_achievement(target_obs)
    assert achieved, "Target obs should achieve goal"
    print("  ✓ Goal achievement: correctly detects achievement")

    # Test 8: Action decision
    action_values = jr.normal(jr.PRNGKey(300), (act_dim,))
    action, confidence = pfc.decide_action(
        obs, context=None, action_values=action_values
    )
    assert action.shape == (act_dim,), "Action shape"
    assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    print(f"  ✓ Action decision: shape={action.shape}, confidence={confidence:.4f}")

    # Test 9: Inhibition computation
    automatic = jr.normal(jr.PRNGKey(400), (act_dim,)) * 0.5
    deliberate = jr.normal(jr.PRNGKey(500), (act_dim,)) * 0.5
    inhibition = pfc.compute_inhibition(automatic, deliberate)
    assert 0 <= inhibition <= 1, "Inhibition should be in [0, 1]"
    print(f"  ✓ Inhibition computation: {inhibition:.4f}")

    # Test 10: Complete step
    pfc.reset()
    pfc.set_goal(description="Test", target_observation=target_obs, reward_bonus=1.0)
    action, confidence, inhibition = pfc.step(obs, generate_new_plan=True)
    assert action.shape == (act_dim,)
    assert isinstance(confidence, float)
    assert isinstance(inhibition, float)
    print(
        f"  ✓ Complete step: action shape={action.shape}, confidence={confidence:.4f}, inhibition={inhibition:.4f}"
    )

    # Test 11: Multiple goals
    pfc.clear_goals()
    pfc.set_goal(description="Goal 1", priority=1.0)
    pfc.set_goal(description="Goal 2", priority=2.0)
    pfc.set_goal(description="Goal 3", priority=0.5)
    goals = pfc.get_goals()
    assert len(goals) == 3
    assert goals[0].priority == 2.0, "Highest priority should be first"
    assert pfc.state.current_goal.description == "Goal 2"
    print(
        f"  ✓ Multiple goals: {len(goals)} goals, highest priority={goals[0].priority}"
    )

    # Test 12: Getters
    assert isinstance(pfc.get_current_goal(), Goal) or pfc.get_current_goal() is None
    assert isinstance(pfc.get_working_memory(), jnp.ndarray)
    assert isinstance(pfc.get_inhibition_signal(), float)
    assert isinstance(pfc.get_goals(), list)
    print("  ✓ Getter methods: all working")

    # Test 13: Clear goals
    pfc.clear_goals()
    assert len(pfc.get_goals()) == 0
    assert pfc.state.current_goal is None
    print("  ✓ Clear goals: all cleared")

    print("✅ All Prefrontal Cortex tests passed!")


if __name__ == "__main__":
    test_prefrontal_cortex()
