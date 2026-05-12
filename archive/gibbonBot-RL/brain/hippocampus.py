"""
Hippocampus - Episodic Memory and Spatial Navigation

This module implements a biologically-inspired hippocampus for episodic memory,
spatial navigation, and context-based learning. The hippocampus is critical
for forming, storing, and retrieving memories of events and their context.

Biological Background:
    The hippocampus consists of several regions:

    1. **Dentate Gyrus**: Acts as a gatekeeper, sparsifying inputs
    2. **CA3**: Autoassociative network for pattern completion
    3. **CA1**: Output layer, integrates information
    4. **Entorhinal Cortex**: Main input/output interface

    The hippocampus:
    - Encodes experiences into episodic memories
    - Enables spatial navigation (place cells, grid cells)
    - Provides context for decision making
    - Supports pattern completion (retrieving full memory from partial cues)
    - Enables rapid one-shot learning

Simplification:
    This implementation focuses on the core functions:
    1. Memory encoding: store state-action-reward tuples
    2. Memory retrieval: find similar past experiences
    3. Context generation: provide contextual information for decision making
    4. Spatial coding: simplified place cell representation
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from brain.episodic_store import EpisodicStore


@dataclass
class HippocampusConfig:
    """Configuration for the hippocampus module."""

    memory_capacity: int = 1000  # Maximum number of memories to store
    similarity_threshold: float = 0.9  # Threshold for memory match
    learning_rate: float = 0.1  # Learning rate for memory encoding
    retrieval_strength: float = 0.5  # Strength of retrieved memory influence
    spatial_dim: int = 16  # Dimension of spatial/latent representation

    def __post_init__(self):
        """Validate config."""
        pass


@dataclass
class Memory:
    """A single episodic memory."""

    observation: jnp.ndarray  # State at the time of the memory
    action: jnp.ndarray  # Action taken
    next_observation: jnp.ndarray  # Resulting state
    reward: float  # Reward received
    punishment: float  # Punishment signal
    timestamp: int  # When the memory was formed
    spatial_code: jnp.ndarray  # Compressed spatial representation


@dataclass
class HippocampusState:
    """State of the hippocampus at one timestep."""

    current_context: jnp.ndarray  # Current contextual representation
    retrieved_memory: Optional[Memory]  # Currently retrieved memory
    similarity_score: float  # Similarity of retrieved memory
    spatial_code: jnp.ndarray  # Current spatial code


class Hippocampus:
    """
    Hippocampus module for episodic memory and context generation.

    The hippocampus:
    1. Encodes experiences as episodic memories
    2. Retrieves similar past experiences to inform current decisions
    3. Generates contextual representations for the current state
    4. Provides spatial coding for navigation

    This module works with other brain regions by:
    - Providing contextual information to the Prefrontal Cortex for planning
    - Supplying episodic memories to the Basal Ganglia for action selection
    - Interacting with the Amygdala for emotionally significant memories

    Attributes:
        config: HippocampusConfig with all parameters
        obs_dim: Dimension of observation input
        act_dim: Dimension of action space
    """

    def __init__(
        self,
        obs_dim: int = 88,
        act_dim: int = 24,
        config: Optional[HippocampusConfig] = None,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize hippocampus.

        Args:
            obs_dim: Dimension of observation input
            act_dim: Dimension of action space
            config: Optional HippocampusConfig (defaults to standard values)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config if config is not None else HippocampusConfig()

        # Memory storage
        self.memories: List[Memory] = []

        # Spatial coding: observation -> spatial_dim
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.w_spatial = (
            jax.random.normal(keys[0], (obs_dim, self.config.spatial_dim)) * 0.1
        )
        self.b_spatial = jnp.zeros(self.config.spatial_dim)

        # Context generation: observation + action -> context
        self.w_context = (
            jax.random.normal(keys[1], (obs_dim + act_dim, self.config.spatial_dim))
            * 0.1
        )
        self.b_context = jnp.zeros(self.config.spatial_dim)

        # Optional persistent episodic store (long-term consolidation substrate).
        self.store: Optional[EpisodicStore] = None
        if persist_path is not None:
            self.store = EpisodicStore(
                persist_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
                embed_dim=self.config.spatial_dim,
                capacity=self.config.memory_capacity * 10,
            )

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset hippocampus state."""
        self.state = HippocampusState(
            current_context=jnp.zeros(self.config.spatial_dim),
            retrieved_memory=None,
            similarity_score=0.0,
            spatial_code=jnp.zeros(self.config.spatial_dim),
        )
        self._current_timestamp = 0

    def reset(self) -> None:
        """Reset hippocampus to initial state."""
        self._reset_state()

    def encode_spatial(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Encode observation into spatial representation (place cell-like).

        Args:
            observation: (obs_dim,) - current observation

        Returns:
            (spatial_dim,) - spatial code
        """
        spatial_code = jnp.dot(observation, self.w_spatial) + self.b_spatial
        spatial_code = jnp.tanh(spatial_code)  # Normalize to [-1, 1]
        return spatial_code

    def encode_context(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate contextual representation from observation and action.

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - current action

        Returns:
            (spatial_dim,) - context code
        """
        x = jnp.concatenate([observation, action])
        context = jnp.dot(x, self.w_context) + self.b_context
        context = jnp.tanh(context)
        return context

    def encode_memory(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
        reward: float,
        punishment: float = 0.0,
    ) -> Memory:
        """
        Create a memory from an experience.

        Args:
            observation: (obs_dim,) - state before action
            action: (act_dim,) - action taken
            next_observation: (obs_dim,) - state after action
            reward: Reward received
            punishment: Punishment signal

        Returns:
            Memory - the encoded memory
        """
        spatial_code = self.encode_spatial(observation)

        memory = Memory(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            punishment=punishment,
            timestamp=self._current_timestamp,
            spatial_code=spatial_code,
        )

        return memory

    def store_memory(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
        reward: float,
        punishment: float = 0.0,
    ) -> None:
        """
        Store an experience as a memory.

        If the memory buffer is full, remove the oldest memory.
        Memories with higher emotional significance (reward/punishment) are
        more likely to be retained.

        Args:
            observation: (obs_dim,) - state before action
            action: (act_dim,) - action taken
            next_observation: (obs_dim,) - state after action
            reward: Reward received
            punishment: Punishment signal
        """
        memory = self.encode_memory(
            observation, action, next_observation, reward, punishment
        )

        # Check capacity
        if len(self.memories) >= self.config.memory_capacity:
            # Remove oldest memory
            self.memories.pop(0)

        self.memories.append(memory)

        if self.store is not None:
            valence = float(reward) - float(punishment)
            self.store.add(
                timestamp=self._current_timestamp,
                obs=np.asarray(observation, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_observation, dtype=np.float32),
                embedding=np.asarray(memory.spatial_code, dtype=np.float32),
                reward=float(reward),
                punishment=float(punishment),
                valence=valence,
            )

        self._current_timestamp += 1

    def compute_similarity(self, obs1: jnp.ndarray, obs2: jnp.ndarray) -> float:
        """
        Compute similarity between two observations.

        Uses cosine similarity.

        Args:
            obs1: (obs_dim,) - first observation
            obs2: (obs_dim,) - second observation

        Returns:
            float - similarity score in [0, 1]
        """
        # Normalize observations
        obs1_norm = obs1 / (jnp.linalg.norm(obs1) + 1e-8)
        obs2_norm = obs2 / (jnp.linalg.norm(obs2) + 1e-8)

        # Cosine similarity
        cosine_sim = jnp.dot(obs1_norm, obs2_norm).item()

        # Convert from [-1, 1] to [0, 1]
        similarity = (cosine_sim + 1.0) / 2.0

        return similarity

    def retrieve_memory(
        self, observation: jnp.ndarray, action: Optional[jnp.ndarray] = None
    ) -> Tuple[Optional[Memory], float]:
        """
        Retrieve the most similar memory to the current state/action.

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - optional current action for context

        Returns:
            (memory, similarity) - most similar memory and its similarity score
        """
        if len(self.memories) == 0:
            return None, 0.0

        best_memory = None
        best_similarity = 0.0

        for memory in self.memories:
            # Compute similarity between current observation and memory observation
            similarity = self.compute_similarity(observation, memory.observation)

            # If action is provided, add action similarity
            if action is not None:
                action_sim = self.compute_similarity(action, memory.action)
                # Weight: 70% observation, 30% action
                similarity = 0.7 * similarity + 0.3 * action_sim

            if similarity > best_similarity:
                best_similarity = similarity
                best_memory = memory

        return best_memory, best_similarity

    def generate_context(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate a contextual representation for the current state-action pair.

        This context incorporates:
        1. Current spatial encoding
        2. Retrieved memory information
        3. Learned contextual associations

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - current action

        Returns:
            (spatial_dim,) - contextual representation
        """
        # Encode spatial context
        spatial_code = self.encode_spatial(observation)

        # Encode action context
        context_code = self.encode_context(observation, action)

        # Retrieve relevant memory
        memory, similarity = self.retrieve_memory(observation, action)

        # If a similar memory exists, incorporate it
        if memory is not None and similarity > self.config.similarity_threshold:
            # Blend current context with remembered context
            remembered_context = memory.spatial_code
            blended_context = (
                1 - self.config.retrieval_strength
            ) * context_code + self.config.retrieval_strength * remembered_context
            context = blended_context
        else:
            context = context_code

        # Update state
        self.state = replace(
            self.state,
            current_context=context,
            retrieved_memory=memory,
            similarity_score=similarity,
            spatial_code=spatial_code,
        )

        return context

    def get_next_state_prediction(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Optional[jnp.ndarray]:
        """
        Predict the next state based on retrieved memories.

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - current action

        Returns:
            (obs_dim,) - predicted next observation, or None if no similar memory
        """
        memory, similarity = self.retrieve_memory(observation, action)

        if memory is not None and similarity > self.config.similarity_threshold:
            return memory.next_observation
        else:
            return None

    def get_expected_reward(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Get the expected reward based on retrieved memories.

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - current action

        Returns:
            (expected_reward, similarity) - expected reward and similarity score
        """
        memory, similarity = self.retrieve_memory(observation, action)

        if memory is not None:
            return memory.reward, similarity
        else:
            return None, 0.0

    def step(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: Optional[jnp.ndarray] = None,
        reward: Optional[float] = None,
        punishment: Optional[float] = None,
        store: bool = True,
    ) -> Tuple[jnp.ndarray, Optional[Memory]]:
        """
        Complete hippocampus processing step.

        This is the main method to call during training/inference:
        1. Generate context for current state-action
        2. Retrieve relevant memory
        3. (Optional) Store current experience as memory

        Args:
            observation: (obs_dim,) - current observation
            action: (act_dim,) - current action
            next_observation: (obs_dim,) - next state (for storing memory)
            reward: Reward received (for storing memory)
            punishment: Punishment signal (for storing memory)
            store: Whether to store this experience as memory

        Returns:
            (context, retrieved_memory) - contextual representation and retrieved memory
        """
        # Generate context
        context = self.generate_context(observation, action)

        # Store memory if requested and all information is available
        if store and next_observation is not None and reward is not None:
            self.store_memory(
                observation, action, next_observation, reward, punishment or 0.0
            )

        # Increment timestamp
        self._current_timestamp += 1

        return context, self.state.retrieved_memory

    def get_context(self) -> jnp.ndarray:
        """Get the current contextual representation."""
        return self.state.current_context

    def get_spatial_code(self) -> jnp.ndarray:
        """Get the current spatial code."""
        return self.state.spatial_code

    def get_memory_count(self) -> int:
        """Get the number of stored memories."""
        return len(self.memories)

    def recall_persistent(self, observation: jnp.ndarray, k: int = 8):
        """Retrieve k most-similar persisted episodes by spatial-code cosine.

        Returns an empty list if no persistent store was configured. This is
        the long-horizon recall path used by the PFC for context augmentation;
        the in-RAM `retrieve_memory` is the working-memory fast path.
        """
        if self.store is None:
            return []
        embedding = np.asarray(self.encode_spatial(observation), dtype=np.float32)
        return self.store.recall(embedding, k=k)

    def consolidate(self, n: int = 32):
        """Hippocampal replay: sample priority-weighted persisted episodes.

        Intended to be called between rollouts (the "sleep" phase). Returns
        episodes the caller can replay into BG/cerebellum at elevated LR.
        """
        if self.store is None:
            return []
        return self.store.sample_by_priority(n)

    def clear_memories(self) -> None:
        """Clear all stored memories."""
        self.memories = []
        self._current_timestamp = 0


# -- Test Function --
def test_hippocampus():
    """Test the Hippocampus module."""
    print("Testing Hippocampus module...")

    import jax.random as jr

    obs_dim = 88
    act_dim = 24
    hippocampus = Hippocampus(obs_dim=obs_dim, act_dim=act_dim)

    # Test 1: Initial state
    assert hippocampus.state.current_context.shape == (hippocampus.config.spatial_dim,)
    assert hippocampus.get_memory_count() == 0
    print("  ✓ Initial state: shapes correct, no memories")

    # Test 2: Spatial encoding
    key = jr.PRNGKey(42)
    obs = jr.normal(key, (obs_dim,)) * 0.1
    spatial = hippocampus.encode_spatial(obs)
    assert spatial.shape == (hippocampus.config.spatial_dim,)
    assert jnp.all(jnp.abs(spatial) <= 1.0), "Spatial code should be in [-1, 1]"
    print(
        f"  ✓ Spatial encoding: shape={spatial.shape}, range=[{spatial.min():.2f}, {spatial.max():.2f}]"
    )

    # Test 3: Context encoding
    action = jr.normal(jr.PRNGKey(100), (act_dim,)) * 0.1
    context = hippocampus.encode_context(obs, action)
    assert context.shape == (hippocampus.config.spatial_dim,)
    print(f"  ✓ Context encoding: shape={context.shape}")

    # Test 4: Store and retrieve memory
    next_obs = obs + jr.normal(jr.PRNGKey(200), (obs_dim,)) * 0.01
    hippocampus.store_memory(obs, action, next_obs, reward=1.0, punishment=0.0)
    assert hippocampus.get_memory_count() == 1

    retrieved, similarity = hippocampus.retrieve_memory(obs, action)
    assert retrieved is not None, "Should retrieve stored memory"
    assert similarity > 0.9, f"Similarity should be high: {similarity}"
    print(
        f"  ✓ Memory storage and retrieval: count={hippocampus.get_memory_count()}, similarity={similarity:.4f}"
    )

    # Test 5: Similarity computation
    similar_obs = obs + jr.normal(jr.PRNGKey(300), (obs_dim,)) * 0.001
    dissimilar_obs = jr.normal(jr.PRNGKey(400), (obs_dim,)) * 10.0

    sim_similar = hippocampus.compute_similarity(obs, similar_obs)
    sim_dissimilar = hippocampus.compute_similarity(obs, dissimilar_obs)
    assert sim_similar > sim_dissimilar, "Similar obs should have higher similarity"
    print(f"  ✓ Similarity: similar={sim_similar:.4f}, dissimilar={sim_dissimilar:.4f}")

    # Test 6: Context generation
    context = hippocampus.generate_context(obs, action)
    assert context.shape == (hippocampus.config.spatial_dim,)
    print(f"  ✓ Context generation: shape={context.shape}")

    # Test 7: Next state prediction
    next_obs_pred = hippocampus.get_next_state_prediction(obs, action)
    assert next_obs_pred is not None, "Should predict next state"
    assert next_obs_pred.shape == (obs_dim,)
    print(f"  ✓ Next state prediction: shape={next_obs_pred.shape}")

    # Test 8: Expected reward
    expected_reward, similarity = hippocampus.get_expected_reward(obs, action)
    assert expected_reward is not None, "Should get expected reward"
    assert expected_reward == 1.0, (
        f"Expected reward should be 1.0, got {expected_reward}"
    )
    print(f"  ✓ Expected reward: {expected_reward:.4f}, similarity={similarity:.4f}")

    # Test 9: Complete step with storing
    hippocampus.reset()
    hippocampus.clear_memories()
    context, retrieved = hippocampus.step(obs, action, next_obs, reward=1.0, store=True)
    assert context.shape == (hippocampus.config.spatial_dim,)
    assert hippocampus.get_memory_count() == 1
    print(
        f"  ✓ Complete step with storing: context shape={context.shape}, memories={hippocampus.get_memory_count()}"
    )

    # Test 10: Complete step without storing
    context, retrieved = hippocampus.step(obs, action, store=False)
    assert context.shape == (hippocampus.config.spatial_dim,)
    assert hippocampus.get_memory_count() == 1, "Memory count should not increase"
    print(
        f"  ✓ Complete step without storing: memories={hippocampus.get_memory_count()}"
    )

    # Test 11: Memory capacity
    for i in range(10):
        obs_i = jr.normal(jr.PRNGKey(500 + i), (obs_dim,)) * 0.1
        action_i = jr.normal(jr.PRNGKey(600 + i), (act_dim,)) * 0.1
        next_obs_i = obs_i + jr.normal(jr.PRNGKey(700 + i), (obs_dim,)) * 0.01
        hippocampus.store_memory(obs_i, action_i, next_obs_i, reward=float(i) * 0.1)
    assert hippocampus.get_memory_count() == 11, (
        f"Should have 11 memories, got {hippocampus.get_memory_count()}"
    )
    print(f"  ✓ Memory capacity: stored {hippocampus.get_memory_count()} memories")

    # Test 12: Getters
    assert isinstance(hippocampus.get_context(), jnp.ndarray), "Context getter"
    assert isinstance(hippocampus.get_spatial_code(), jnp.ndarray), (
        "Spatial code getter"
    )
    assert isinstance(hippocampus.get_memory_count(), int), "Memory count getter"
    print("  ✓ Getter methods: all working")

    # Test 13: Clear memories
    hippocampus.clear_memories()
    assert hippocampus.get_memory_count() == 0
    print("  ✓ Clear memories: count=0")

    print("✅ All Hippocampus tests passed!")


if __name__ == "__main__":
    test_hippocampus()
