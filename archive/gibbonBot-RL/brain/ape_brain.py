"""
ApeBrain - Integrated Bioinspired Brain Architecture

This is a simplified integrator for the bioinspired brain modules.
Each module maintains its own API and can be used independently.

For full integration documentation, see docs/APEBRAIN.md
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from brain.bus import EVENT_NOVELTY, EVENT_REPLAY_TICK, EVENT_THREAT, BrainBus


@dataclass
class ApeBrainConfig:
    """Configuration for ApeBrain."""

    obs_dim: int = 88
    act_dim: int = 24
    training: bool = True
    persist_dir: Optional[str] = None  # if set, hippocampus + semantic map persist here


class ApeBrain:
    """
    Simplified ApeBrain integrator.

    Provides access to all brain modules with a unified interface.
    Each module can be used independently for advanced usage.

    Usage:
        brain = ApeBrain()

        # Access individual modules
        bg = brain.modules["basal_ganglia"]
        action_idx, _ = bg.select_action_deterministic(observation, dopamine=0.0)

        # Or use the simplified step interface
        action = brain.step(observation, reward=reward)
    """

    def __init__(self, config: Optional[ApeBrainConfig] = None):
        """Initialize ApeBrain with all modules."""
        self.config = config if config is not None else ApeBrainConfig()
        self.modules = {}
        self._step_count = 0
        self.bus = BrainBus()
        self.semantic_map = None
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

        self.modules["basal_ganglia"] = BasalGanglia(
            n_actions=self.config.act_dim,
            n_inputs=self.config.obs_dim,
        )
        self.modules["cerebellum"] = Cerebellum(
            obs_dim=self.config.obs_dim,
            act_dim=self.config.act_dim,
        )
        self.modules["amygdala"] = Amygdala(
            obs_dim=self.config.obs_dim,
            act_dim=self.config.act_dim,
        )
        hippo_persist = None
        if self.config.persist_dir is not None:
            from pathlib import Path

            persist_dir = Path(self.config.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            hippo_persist = str(persist_dir / "episodic.sqlite")
            from brain.semantic_map import SemanticMap

            self.semantic_map = SemanticMap(persist_dir / "semantic.sqlite")
        self.modules["hippocampus"] = Hippocampus(
            obs_dim=self.config.obs_dim,
            act_dim=self.config.act_dim,
            persist_path=hippo_persist,
        )
        self.modules["prefrontal_cortex"] = PrefrontalCortex(
            obs_dim=self.config.obs_dim,
            act_dim=self.config.act_dim,
        )
        self.modules["motor_cortex"] = MotorCortex(
            obs_dim=self.config.obs_dim,
            act_dim=self.config.act_dim,
        )
        self.modules["brainstem"] = Brainstem(
            n_joints=self.config.act_dim,
            n_obs=self.config.obs_dim,
        )
        self.modules["neurotransmitters"] = Neurotransmitters()

    def reset(self):
        """Reset all modules."""
        for module in self.modules.values():
            if hasattr(module, "reset"):
                module.reset()
        self._step_count = 0

    def step(
        self,
        observation: jnp.ndarray,
        reward: Optional[float] = None,
        next_observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Perform one brain step using Basal Ganglia as the primary controller.

        This is a simplified integration. For full control, use individual modules.

        Args:
            observation: Current observation from environment
            reward: Optional reward signal
            next_observation: Optional next observation for learning

        Returns:
            Action vector in [-1, 1] range
        """
        import jax.random as jr

        self._step_count += 1

        # Initialize modules
        bg = self.modules["basal_ganglia"]
        amygdala = self.modules["amygdala"]
        dopamine = 0.0
        action_penalty = jnp.zeros(self.config.act_dim)

        if reward is not None:
            dopamine = min(max(reward, 0) * 2, 1.0)  # Convert reward to dopamine

            # Amygdala: assess threat and apply fear modulation
            # Predict BG action first for amygdala to evaluate
            pred_state = bg.process_cortical_input(observation, dopamine)
            pred_action_idx = int(jnp.argmax(pred_state.thalamus_output))
            pred_action = jnp.zeros(self.config.act_dim)
            if pred_action_idx < self.config.act_dim:
                pred_action = pred_action.at[pred_action_idx].set(1.0)
            pred_action = pred_action * 2.0 - 1.0

            # Process through amygdala
            action_penalty, _ = amygdala.step(observation, pred_action, reward)
            fear_level = amygdala.get_fear_level()

            # Fear reduces dopamine (biological: fear suppresses dopamine release)
            dopamine = dopamine * (1.0 - fear_level * 0.7)

        # Get action from BG with fear-modulated dopamine
        state = bg.process_cortical_input(observation, dopamine)
        action_idx, _ = bg.select_action_deterministic(observation, dopamine)

        # Convert action index to action vector
        action = jnp.zeros(self.config.act_dim)
        if action_idx < self.config.act_dim:
            action = action.at[action_idx].set(1.0)
        action = action * 2.0 - 1.0  # Scale to [-1, 1]

        # Apply amygdala's action penalty (avoidance behavior)
        action = action - action_penalty * 0.3

        # Prefrontal Cortex: goal-directed modulation
        pfc = self.modules["prefrontal_cortex"]
        hippo = self.modules["hippocampus"]
        context = hippo.get_context() if hasattr(hippo, "get_context") else None
        pfc_action, pfc_confidence, inhibition = pfc.step(
            observation,
            context=context,
            action_values=bg.action_values,
            generate_new_plan=(reward is not None),
        )

        # Apply PFC inhibition (cognitive control over automatic responses)
        if inhibition > 0:
            action = action * (1.0 - inhibition * 0.5)

        # Blend with PFC action when confident (goal-directed override)
        if pfc_confidence > 0.5:
            blend = pfc_confidence * 0.4  # Max 40% PFC influence
            action = action * (1.0 - blend) + pfc_action * blend

        # Learn if training
        if reward is not None and next_observation is not None and self.config.training:
            bg.learn(reward, next_observation, action_idx)

        # Record the transition in episodic memory (and persistent store if configured).
        if reward is not None and next_observation is not None:
            hippo = self.modules["hippocampus"]
            punishment = max(-float(reward), 0.0)
            hippo.store_memory(
                observation,
                action,
                next_observation,
                reward=float(reward),
                punishment=punishment,
            )

        # Apply cerebellum correction
        cerebellum = self.modules["cerebellum"]
        if next_observation is not None:
            action = cerebellum.step(
                observation,
                action,
                next_observation,
                learning_rate=0.01 if self.config.training else None,
            )
        else:
            action = cerebellum.step(observation, action)

        # Apply motor cortex smoothing
        motor_cortex = self.modules["motor_cortex"]
        intention = jnp.zeros(motor_cortex.config.primitive_dim)
        mc_action = motor_cortex.step(intention, use_primitives=False)
        action = action * 0.7 + mc_action * 0.3  # Blend

        # Clamp and return
        action = jnp.clip(action, -1.0, 1.0)
        return action

    def get_module(self, name: str) -> Optional[Any]:
        """Get a specific module by name."""
        return self.modules.get(name)

    def sleep(self, n_replays: int = 64, lr_scale: float = 2.0) -> int:
        """Consolidation replay — the "sleep" phase.

        Pulls priority-weighted episodes from the persistent hippocampal store
        and replays them through the basal ganglia at elevated learning rate.
        Mirrors how mammals consolidate the day's experience during sleep,
        compressing episodic memory into procedural skill.

        Returns the number of episodes replayed (0 if no persistent store).
        """
        hippo = self.modules["hippocampus"]
        if hippo.store is None:
            return 0
        episodes = hippo.consolidate(n=n_replays)
        if not episodes:
            return 0

        self.bus.publish(EVENT_REPLAY_TICK, {"n": len(episodes)})

        bg = self.modules["basal_ganglia"]
        original_lr = bg.learning_rate
        bg.learning_rate = original_lr * lr_scale
        try:
            for ep in episodes:
                # Map continuous action vector back to discrete action index.
                act_idx = int(np.argmax(ep.action))
                if act_idx >= self.config.act_dim:
                    continue
                bg.learn(
                    reward=float(ep.reward) - float(ep.punishment),
                    next_cortical_input=jnp.asarray(ep.next_obs),
                    selected_action=act_idx,
                )
        finally:
            bg.learning_rate = original_lr
        return len(episodes)

    def emit_novelty(
        self, obs: jnp.ndarray, similarity: float, threshold: float = 0.3
    ) -> None:
        """If the current obs is far from anything in memory, publish a novelty event."""
        if similarity < threshold:
            self.bus.publish(
                EVENT_NOVELTY,
                {"similarity": float(similarity), "step": self._step_count},
            )

    def emit_threat(self, valence: float, threshold: float = -1.0) -> None:
        """If valence is sharply negative, publish a threat event."""
        if valence <= threshold:
            self.bus.publish(EVENT_THREAT, {"valence": float(valence)})

    def get_info(self) -> Dict[str, Any]:
        """Get information about the brain state."""
        return {
            "step_count": self._step_count,
            "modules": list(self.modules.keys()),
        }


def test_ape_brain():
    """Test ApeBrain."""
    print("Testing ApeBrain...")
    import jax.random as jr

    brain = ApeBrain()

    # Test initialization
    assert len(brain.modules) == 8
    print("  ✓ All 8 modules initialized")

    # Test reset
    brain.reset()
    print("  ✓ Reset")

    # Test step
    key = jr.PRNGKey(42)
    obs = jr.normal(key, (88,)) * 0.1
    action = brain.step(obs)
    assert action.shape == (24,)
    assert jnp.all(jnp.abs(action) <= 1.0)
    print("  ✓ Step")

    # Test step with reward
    next_obs = obs + jr.normal(jr.PRNGKey(100), (88,)) * 0.01
    action = brain.step(obs, reward=1.0, next_observation=next_obs)
    assert action.shape == (24,)
    print("  ✓ Step with reward")

    # Test get_module
    bg = brain.get_module("basal_ganglia")
    assert bg is not None
    print("  ✓ Get module")

    # Test get_info
    info = brain.get_info()
    assert "step_count" in info
    print("  ✓ Get info")

    print("✅ All ApeBrain tests passed!")


if __name__ == "__main__":
    test_ape_brain()
