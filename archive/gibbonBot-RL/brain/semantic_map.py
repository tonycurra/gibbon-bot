"""
SemanticMap — cortical-style consolidated world model.

Where `EpisodicStore` holds individual time-stamped experiences (hippocampus),
this holds *aggregated facts* about persistent entities in the world (cortex).
For the olive grove: which trees exist, where they are, when each was last
visited, and the current best-known fruit state at each.

This is the slow, deduplicated semantic layer. It is updated by aggregating
episodic rows during consolidation, not by every step. Reads are O(log N)
indexed lookups so the PFC can cheaply ask "what's at this location" or
"which trees haven't been visited recently."

Schema
------
trees(id, x, y, z, last_visit_step, last_visit_time, fruit_state, visit_count, notes)

`fruit_state` is a small float in [0, 1]: 0=barren, 1=ripe heavy load.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trees (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    z               REAL NOT NULL DEFAULT 0.0,
    last_visit_step INTEGER NOT NULL DEFAULT 0,
    last_visit_time REAL    NOT NULL DEFAULT 0.0,
    fruit_state     REAL    NOT NULL DEFAULT 0.0,
    visit_count     INTEGER NOT NULL DEFAULT 0,
    notes           TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trees_xy ON trees(x, y);
CREATE INDEX IF NOT EXISTS idx_trees_last_visit ON trees(last_visit_step);
CREATE INDEX IF NOT EXISTS idx_trees_fruit ON trees(fruit_state);
"""


@dataclass
class Tree:
    id: int
    x: float
    y: float
    z: float
    last_visit_step: int
    last_visit_time: float
    fruit_state: float
    visit_count: int
    notes: str


class SemanticMap:
    """Persistent world map: trees and their state."""

    # Trees within this distance (m) are treated as the same tree.
    MERGE_RADIUS = 0.4

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0])

    def observe_tree(
        self,
        x: float,
        y: float,
        step: int,
        fruit_state: float = 0.0,
        z: float = 0.0,
        notes: str = "",
    ) -> int:
        """Record an observation of a tree at (x, y).

        If a tree already exists within MERGE_RADIUS, update it; otherwise
        insert a new row. Returns the row id.
        """
        existing = self._nearest(x, y, max_dist=self.MERGE_RADIUS)
        if existing is not None:
            self._conn.execute(
                "UPDATE trees SET last_visit_step=?, last_visit_time=?, "
                "fruit_state=?, visit_count=visit_count+1, "
                "x = (x*visit_count + ?) / (visit_count+1), "
                "y = (y*visit_count + ?) / (visit_count+1) "
                "WHERE id=?",
                (step, time.time(), fruit_state, x, y, existing.id),
            )
            self._conn.commit()
            return existing.id

        cur = self._conn.execute(
            "INSERT INTO trees (x, y, z, last_visit_step, last_visit_time, "
            "fruit_state, visit_count, notes) VALUES (?,?,?,?,?,?,1,?)",
            (x, y, z, step, time.time(), fruit_state, notes),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def all_trees(self) -> List[Tree]:
        rows = self._conn.execute(
            "SELECT id, x, y, z, last_visit_step, last_visit_time, fruit_state, "
            "visit_count, notes FROM trees ORDER BY id ASC"
        ).fetchall()
        return [Tree(*r) for r in rows]

    def _nearest(self, x: float, y: float, max_dist: float) -> Optional[Tree]:
        # Small N (forest, not galaxy) — full scan in Python is fine.
        rows = self._conn.execute(
            "SELECT id, x, y, z, last_visit_step, last_visit_time, fruit_state, "
            "visit_count, notes FROM trees"
        ).fetchall()
        best: Optional[Tree] = None
        best_d = max_dist * max_dist
        for r in rows:
            dx, dy = r[1] - x, r[2] - y
            d = dx * dx + dy * dy
            if d <= best_d:
                best_d = d
                best = Tree(*r)
        return best

    def nearest(self, x: float, y: float, max_dist: float = 1e9) -> Optional[Tree]:
        """Return the closest tree to (x, y) within max_dist, or None."""
        return self._nearest(x, y, max_dist)

    def stale_trees(self, current_step: int, min_age: int = 10_000) -> List[Tree]:
        """Trees not visited for at least `min_age` steps. Targets for exploration."""
        rows = self._conn.execute(
            "SELECT id, x, y, z, last_visit_step, last_visit_time, fruit_state, "
            "visit_count, notes FROM trees WHERE ? - last_visit_step >= ? "
            "ORDER BY last_visit_step ASC",
            (current_step, min_age),
        ).fetchall()
        return [Tree(*r) for r in rows]

    def ripe_trees(self, min_fruit: float = 0.5) -> List[Tree]:
        """Trees believed to have fruit. Targets for harvesting."""
        rows = self._conn.execute(
            "SELECT id, x, y, z, last_visit_step, last_visit_time, fruit_state, "
            "visit_count, notes FROM trees WHERE fruit_state >= ? "
            "ORDER BY fruit_state DESC",
            (min_fruit,),
        ).fetchall()
        return [Tree(*r) for r in rows]

    def decay_fruit(self, factor: float = 0.99) -> None:
        """Exponentially decay fruit-state belief over time.

        Stale beliefs should not be trusted forever — call this each
        consolidation tick so unvisited trees drift toward 0.
        """
        self._conn.execute("UPDATE trees SET fruit_state = fruit_state * ?", (factor,))
        self._conn.commit()


def test_semantic_map() -> None:
    import tempfile

    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "semantic_test.sqlite"
    if path.exists():
        path.unlink()

    m = SemanticMap(path)
    assert len(m) == 0

    a = m.observe_tree(1.0, 2.0, step=100, fruit_state=0.8)
    b = m.observe_tree(5.0, 5.0, step=110, fruit_state=0.0)
    assert len(m) == 2
    # Re-observation within merge radius updates the same tree.
    a2 = m.observe_tree(1.1, 2.05, step=200, fruit_state=0.6)
    assert a == a2
    assert len(m) == 2

    near = m.nearest(1.0, 2.0)
    assert near is not None and near.id == a
    assert near.visit_count == 2
    # Position averaged toward second observation.
    assert 1.0 < near.x < 1.1

    # Ripe lookup respects threshold.
    ripe = m.ripe_trees(min_fruit=0.5)
    assert len(ripe) == 1 and ripe[0].id == a

    # Staleness.
    stale = m.stale_trees(current_step=10_300, min_age=10_150)
    assert len(stale) == 1 and stale[0].id == b, f"stale={[(t.id, t.last_visit_step) for t in stale]}"

    m.decay_fruit(0.5)
    near = m.nearest(1.0, 2.0)
    assert near is not None and abs(near.fruit_state - 0.3) < 1e-6

    # Persistence.
    m.close()
    m2 = SemanticMap(path)
    assert len(m2) == 2
    m2.close()
    print("✅ SemanticMap tests passed")


if __name__ == "__main__":
    test_semantic_map()
