"""
Episodic memory persistence — SQLite-backed store for the Hippocampus.

Biological analogue: long-term consolidation of hippocampal episodes. While
`Hippocampus` holds a fast in-RAM working set (CA3-like), this store is the
slow, durable substrate that survives across runs and supports k-NN recall
by state embedding.

Storage: one row per episode with columns:
    id INTEGER PRIMARY KEY
    timestamp INTEGER       -- step counter when stored
    wall_time REAL          -- unix time when stored
    obs BLOB                -- float32 (obs_dim,)
    action BLOB             -- float32 (act_dim,)
    next_obs BLOB           -- float32 (obs_dim,)
    reward REAL
    punishment REAL
    valence REAL            -- amygdala tag, signed salience
    embedding BLOB          -- float32 (embed_dim,) normalized for cosine

Retrieval loads the embedding matrix into RAM once (cached) and does a
vectorized cosine scan. Sufficient up to ~100k rows; swap for FAISS/LanceDB
later if needed without changing the public API.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,
    wall_time   REAL    NOT NULL,
    obs         BLOB    NOT NULL,
    action      BLOB    NOT NULL,
    next_obs    BLOB    NOT NULL,
    reward      REAL    NOT NULL,
    punishment  REAL    NOT NULL DEFAULT 0.0,
    valence     REAL    NOT NULL DEFAULT 0.0,
    embedding   BLOB    NOT NULL,
    task        TEXT    NOT NULL DEFAULT '',
    scene_hash  TEXT    NOT NULL DEFAULT '',
    concept_id  INTEGER,
    run_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_valence   ON episodes(valence);
-- Indexes for migration-added columns are created in _migrate after ensuring
-- those columns exist on legacy DBs (would fail here for pre-migration tables).
"""

# Columns added after the original schema. Each entry is run as
# `ALTER TABLE episodes ADD COLUMN <ddl>` once, only if missing.
_EPISODES_MIGRATIONS = [
    ("task",       "TEXT NOT NULL DEFAULT ''"),
    ("scene_hash", "TEXT NOT NULL DEFAULT ''"),
    ("concept_id", "INTEGER"),
    ("run_id",     "TEXT"),
]


@dataclass
class Episode:
    id: int
    timestamp: int
    wall_time: float
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    reward: float
    punishment: float
    valence: float
    embedding: np.ndarray


def _to_blob(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(shape)


class EpisodicStore:
    """Persistent episodic memory with cosine-similarity recall."""

    def __init__(
        self,
        path: str | Path,
        obs_dim: int,
        act_dim: int,
        embed_dim: int,
        capacity: Optional[int] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.capacity = capacity

        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

        # Cached embedding matrix for fast scan. Rebuilt on write.
        self._cache_ids: Optional[np.ndarray] = None
        self._cache_emb: Optional[np.ndarray] = None
        self._cache_dirty = True

    def _migrate(self) -> None:
        """Idempotently add columns introduced after the original schema."""
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(episodes)").fetchall()
        }
        for name, ddl in _EPISODES_MIGRATIONS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE episodes ADD COLUMN {name} {ddl}")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_task    ON episodes(task)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_concept ON episodes(concept_id)")

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM episodes")
        return int(cur.fetchone()[0])

    def add(
        self,
        timestamp: int,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        embedding: np.ndarray,
        reward: float,
        punishment: float = 0.0,
        valence: float = 0.0,
        task: str = "",
        scene_hash: str = "",
        run_id: Optional[str] = None,
        concept_id: Optional[int] = None,
    ) -> int:
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        cur = self._conn.execute(
            "INSERT INTO episodes (timestamp, wall_time, obs, action, next_obs, "
            "reward, punishment, valence, embedding, task, scene_hash, run_id, concept_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                int(timestamp),
                time.time(),
                _to_blob(obs),
                _to_blob(action),
                _to_blob(next_obs),
                float(reward),
                float(punishment),
                float(valence),
                _to_blob(emb),
                str(task or ""),
                str(scene_hash or ""),
                run_id,
                concept_id,
            ),
        )
        row_id = int(cur.lastrowid)

        if self.capacity is not None:
            # Evict lowest-|valence| oldest rows when over capacity.
            n = len(self)
            if n > self.capacity:
                excess = n - self.capacity
                self._conn.execute(
                    "DELETE FROM episodes WHERE id IN ("
                    "SELECT id FROM episodes ORDER BY ABS(valence) ASC, timestamp ASC LIMIT ?)",
                    (excess,),
                )

        self._conn.commit()
        self._cache_dirty = True
        return row_id

    def _rebuild_cache(self) -> None:
        rows = self._conn.execute(
            "SELECT id, embedding FROM episodes ORDER BY id ASC"
        ).fetchall()
        if not rows:
            self._cache_ids = np.zeros((0,), dtype=np.int64)
            self._cache_emb = np.zeros((0, self.embed_dim), dtype=np.float32)
        else:
            self._cache_ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._cache_emb = np.stack(
                [_from_blob(r[1], (self.embed_dim,)) for r in rows], axis=0
            )
        self._cache_dirty = False

    def recall(self, query_embedding: np.ndarray, k: int = 8) -> List[Episode]:
        """Return up to k episodes most similar to query (cosine)."""
        if self._cache_dirty or self._cache_emb is None:
            self._rebuild_cache()
        if self._cache_emb is None or self._cache_emb.shape[0] == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        sims = self._cache_emb @ q  # (N,)
        k = min(k, sims.shape[0])
        # Argpartition for top-k, then sort that slice.
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        wanted_ids = self._cache_ids[idx].tolist()
        return self._fetch_by_ids(wanted_ids)

    def _fetch_by_ids(self, ids: List[int]) -> List[Episode]:
        if not ids:
            return []
        q = "SELECT id, timestamp, wall_time, obs, action, next_obs, reward, " \
            "punishment, valence, embedding FROM episodes WHERE id IN ({})".format(
                ",".join("?" * len(ids))
            )
        rows = self._conn.execute(q, ids).fetchall()
        by_id = {r[0]: r for r in rows}
        out: List[Episode] = []
        for i in ids:
            r = by_id.get(i)
            if r is None:
                continue
            out.append(Episode(
                id=r[0],
                timestamp=r[1],
                wall_time=r[2],
                obs=_from_blob(r[3], (self.obs_dim,)),
                action=_from_blob(r[4], (self.act_dim,)),
                next_obs=_from_blob(r[5], (self.obs_dim,)),
                reward=r[6],
                punishment=r[7],
                valence=r[8],
                embedding=_from_blob(r[9], (self.embed_dim,)),
            ))
        return out

    def sample_by_priority(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        current_task: Optional[str] = None,
        cross_task_boost: float = 1.5,
    ) -> List[Episode]:
        """Sample n episodes with probability proportional to |valence| + small floor.

        Mirrors hippocampal replay: emotionally tagged episodes are revisited
        more often during consolidation. When `current_task` is provided,
        episodes from *other* tasks are upweighted by `cross_task_boost`,
        biasing replay toward foreign experience for cross-task generalization.
        """
        rng = rng or np.random.default_rng()
        rows = self._conn.execute("SELECT id, valence, task FROM episodes").fetchall()
        if not rows:
            return []
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        w = np.abs(np.array([r[1] for r in rows], dtype=np.float64)) + 1e-3
        if current_task is not None and cross_task_boost != 1.0:
            tasks = np.array([(r[2] or "") for r in rows])
            w = np.where(tasks != current_task, w * float(cross_task_boost), w)
        p = w / w.sum()
        n = min(n, ids.shape[0])
        picked = rng.choice(ids, size=n, replace=False, p=p).tolist()
        return self._fetch_by_ids(picked)


def test_episodic_store(tmp_dir: Optional[str] = None) -> None:
    import tempfile

    tmp = tmp_dir or tempfile.mkdtemp()
    path = Path(tmp) / "episodic_test.sqlite"
    if path.exists():
        path.unlink()

    obs_dim, act_dim, embed_dim = 8, 4, 6
    store = EpisodicStore(path, obs_dim, act_dim, embed_dim, capacity=100)
    assert len(store) == 0

    rng = np.random.default_rng(0)
    for t in range(20):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        act = rng.normal(size=act_dim).astype(np.float32)
        nxt = obs + 0.01 * rng.normal(size=obs_dim).astype(np.float32)
        emb = rng.normal(size=embed_dim).astype(np.float32)
        store.add(t, obs, act, nxt, emb, reward=float(t % 5), valence=float(t - 10))
    assert len(store) == 20

    # Recall: query close to episode 5's embedding should rank it high.
    target = store._fetch_by_ids([6])[0]  # id is 1-indexed; row 6 is t=5
    hits = store.recall(target.embedding, k=3)
    assert len(hits) == 3
    assert hits[0].id == target.id, f"top hit should be self, got {hits[0].id}"

    # Persistence: reopen and confirm count.
    store.close()
    store2 = EpisodicStore(path, obs_dim, act_dim, embed_dim, capacity=100)
    assert len(store2) == 20

    # Priority sampling returns valid rows.
    picks = store2.sample_by_priority(5)
    assert len(picks) == 5

    # Capacity eviction.
    small = EpisodicStore(Path(tmp) / "small.sqlite", obs_dim, act_dim, embed_dim, capacity=10)
    for t in range(25):
        small.add(
            t,
            rng.normal(size=obs_dim).astype(np.float32),
            rng.normal(size=act_dim).astype(np.float32),
            rng.normal(size=obs_dim).astype(np.float32),
            rng.normal(size=embed_dim).astype(np.float32),
            reward=0.0,
            valence=float(t),
        )
    assert len(small) == 10, f"capacity eviction failed: {len(small)}"
    small.close()
    store2.close()
    print("✅ EpisodicStore tests passed")


if __name__ == "__main__":
    test_episodic_store()
