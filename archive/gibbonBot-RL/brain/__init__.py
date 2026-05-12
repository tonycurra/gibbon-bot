"""
ApeBrain - Bioinspired Neural Architecture for GibbonBot RL

This module implements a biologically-plausible brain architecture for
reinforcement learning, inspired by ape and human neuroanatomy.

Core Components:
- Neurotransmitters: Dopamine, Serotonin, GABA, Noradrenaline, Acetylcholine
- Brainstem: Vestibular system, reticular formation, spinal cord reflexes
- Basal Ganglia: Action selection via Direct/Indirect pathways
- Cerebellum: Motor error correction and fine-tuning
- Hippocampus: Episodic memory storage and retrieval
- Amygdala: Threat detection and fear conditioning
- Prefrontal Cortex: High-level planning and working memory

The ApeBrain class integrates all these components into a cohesive
bioinspired learning system that is also quantum-ready for future
quantum computing integration.

Architecture Design:
    ┌─────────────────────────────────────────────────────────────┐
    │                      NEOCORTEX (Planning)                      │
    │   Prefrontal Cortex → Motor Cortex → Action Output            │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                    SUBORTICAL STRUCTURES                       │
    │   Basal Ganglia ←→ Cerebellum ←→ Amygdala ←→ Hippocampus      │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    BRAINSTEM & SPINAL CORD                     │
    │   Vestibular System, Reticular Formation, Spinal Reflexes     │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from brain import ApeBrain

    brain = ApeBrain(obs_dim=88, act_dim=24, n_poses=4)

    # Step through environment
    action = brain.step(observation)

    # Learn from experience
    brain.learn(observation, action, reward, next_obs, done)

Quantum-Ready:
    The architecture is designed with modular interfaces that can be
    replaced with quantum equivalents (QNN layers, quantum memory, etc.)
"""

# Lazy imports to avoid circular dependencies and missing module errors
_modules = {}


def __getattr__(name):
    """Lazy import of brain submodules."""
    _module_map = {
        "Neurotransmitters": ("brain.neurotransmitters", "Neurotransmitters"),
        "Brainstem": ("brain.brainstem", "Brainstem"),
        "BasalGanglia": ("brain.basal_ganglia", "BasalGanglia"),
        "Cerebellum": ("brain.cerebellum", "Cerebellum"),
        "Hippocampus": ("brain.hippocampus", "Hippocampus"),
        "Amygdala": ("brain.amygdala", "Amygdala"),
        "PrefrontalCortex": ("brain.prefrontal_cortex", "PrefrontalCortex"),
        "MotorCortex": ("brain.motor_cortex", "MotorCortex"),
        "ApeBrain": ("brain.ape_brain", "ApeBrain"),
        "EpisodicStore": ("brain.episodic_store", "EpisodicStore"),
        "SemanticMap": ("brain.semantic_map", "SemanticMap"),
        "BrainBus": ("brain.bus", "BrainBus"),
        "NeuromodState": ("brain.bus", "NeuromodState"),
    }

    if name in _module_map:
        module_path, class_name = _module_map[name]
        if module_path not in _modules:
            try:
                _modules[module_path] = __import__(module_path, fromlist=[class_name])
            except ModuleNotFoundError:
                return None
        module = _modules.get(module_path)
        if module is not None:
            return getattr(module, class_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Neurotransmitters",
    "Brainstem",
    "BasalGanglia",
    "Cerebellum",
    "Hippocampus",
    "Amygdala",
    "PrefrontalCortex",
    "MotorCortex",
    "ApeBrain",
    "EpisodicStore",
    "SemanticMap",
    "BrainBus",
    "NeuromodState",
]

__version__ = "0.1.0"
