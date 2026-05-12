"""
BrainBus — inter-region communication for ApeBrain.

Biology has three signaling channels between regions:

    1. Fast axonal projections    -> direct tensor passing (function calls)
    2. Slow neuromodulators       -> global scalars read by every region
    3. Oscillatory binding/events -> sparse async notifications

(1) is already how modules call each other. This file provides (2) and (3):

    NeuromodState   — shared dopamine/serotonin/noradrenaline/acetylcholine/GABA
    BrainBus        — pub/sub for discrete events (novelty, threat, replay tick)

Both are deliberately tiny: a dataclass plus a dict-of-callbacks. The point
is the *contract* — every region reads `NeuromodState` instead of taking a
literal `learning_rate` constant, and any region can emit/subscribe to
events without knowing who else is listening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class NeuromodState:
    """Global neuromodulator levels — read by every brain region.

    Each value is a unitless scalar in roughly [0, 1] (except dopamine which
    can briefly spike higher on big positive RPE). Defaults represent a
    calm, awake animal.
    """
    dopamine: float = 0.5         # reward / RPE; scales BG learning rate
    serotonin: float = 0.5        # patience / discount horizon (γ)
    noradrenaline: float = 0.3    # arousal / exploration temperature
    acetylcholine: float = 0.4    # signal-to-noise gating / attention
    gaba: float = 0.5             # global inhibition / motor calm

    def update(self, **kwargs: float) -> "NeuromodState":
        """Return a new state with the given fields overridden."""
        cur = {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "noradrenaline": self.noradrenaline,
            "acetylcholine": self.acetylcholine,
            "gaba": self.gaba,
        }
        cur.update(kwargs)
        return NeuromodState(**cur)


# Standard event names — string constants so typos surface as misses.
EVENT_NOVELTY = "novelty"          # hippocampus saw an embedding far from anything stored
EVENT_THREAT = "threat"            # amygdala flagged high punishment / instability
EVENT_GOAL_ACHIEVED = "goal_done"  # PFC marked a goal complete
EVENT_REPLAY_TICK = "replay_tick"  # consolidation pass starting
EVENT_REWARD_SPIKE = "reward_spike"  # large positive RPE


@dataclass
class BrainBus:
    """Minimal pub/sub. Synchronous, single-threaded — fine for one robot.

    Usage:
        bus = BrainBus()
        bus.subscribe("threat", amygdala.on_threat)
        bus.publish("threat", payload={"valence": -2.0, "obs": obs})

    Handlers receive the event name and an arbitrary payload dict.
    Exceptions in one handler do not block the others.
    """

    neuromod: NeuromodState = field(default_factory=NeuromodState)
    _subs: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = field(
        default_factory=dict
    )
    _last_payloads: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def subscribe(self, event: str, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        self._subs.setdefault(event, []).append(handler)

    def publish(self, event: str, payload: Dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self._last_payloads[event] = payload
        for h in self._subs.get(event, ()):
            try:
                h(event, payload)
            except Exception as e:  # noqa: BLE001 — handlers must not crash bus
                print(f"[BrainBus] handler for {event!r} raised: {e!r}")

    def last(self, event: str) -> Dict[str, Any] | None:
        return self._last_payloads.get(event)

    def set_neuromod(self, **kwargs: float) -> None:
        self.neuromod = self.neuromod.update(**kwargs)


def test_brain_bus() -> None:
    bus = BrainBus()
    assert bus.neuromod.dopamine == 0.5

    seen: list = []
    bus.subscribe(EVENT_THREAT, lambda e, p: seen.append((e, p["valence"])))
    bus.subscribe(EVENT_THREAT, lambda e, p: seen.append(("second", p["valence"])))
    bus.publish(EVENT_THREAT, {"valence": -1.5})
    assert seen == [(EVENT_THREAT, -1.5), ("second", -1.5)]

    # Crashing handler does not break siblings.
    def bad(e, p): raise RuntimeError("boom")
    bus.subscribe(EVENT_NOVELTY, bad)
    bus.subscribe(EVENT_NOVELTY, lambda e, p: seen.append("survived"))
    bus.publish(EVENT_NOVELTY, {"x": 1})
    assert "survived" in seen

    bus.set_neuromod(dopamine=0.9, noradrenaline=0.7)
    assert bus.neuromod.dopamine == 0.9
    assert bus.neuromod.serotonin == 0.5  # unchanged

    assert bus.last(EVENT_THREAT) == {"valence": -1.5}
    print("✅ BrainBus tests passed")


if __name__ == "__main__":
    test_brain_bus()
