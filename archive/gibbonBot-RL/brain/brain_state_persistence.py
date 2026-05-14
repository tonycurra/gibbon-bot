"""
Brain State Persistence — SQLite-backed storage for complete brain module states.

This module provides functions to save and restore the complete state of all
ApeBrain modules, enabling:
- Checkpointing brain state across training sessions
- Transferring learned knowledge between robots
- Analyzing brain development over time

Storage Schema:
    Each module's state is stored in its own table with a version field for
    backward compatibility. States are serialized as JSON or pickled arrays.

Usage:
    from brain.brain_state_persistence import BrainStatePersistence

    persistence = BrainStatePersistence("data/brain_states/brain_state.sqlite")

    # Save current brain state
    persistence.save(brain, step_count=1000, metadata={"task": "walk"})

    # Load brain state
    persistence.load(brain, step_count=1000)

    # Query available states
    states = persistence.list_states(task="walk")
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

# =============================================================================
# Schema
# =============================================================================

_SCHEMA = """
-- Main table: metadata about each saved state
CREATE TABLE IF NOT EXISTS brain_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_count INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    task TEXT,
    scene_hash TEXT,
    metadata TEXT DEFAULT '{}',
    UNIQUE(step_count, task, scene_hash)
);
CREATE INDEX IF NOT EXISTS idx_brain_states_step ON brain_states(step_count DESC);
CREATE INDEX IF NOT EXISTS idx_brain_states_task ON brain_states(task);
CREATE INDEX IF NOT EXISTS idx_brain_states_time ON brain_states(wall_time DESC);

-- Module-specific state tables
-- Each stores: (state_id, module_name, state_blob, version, timestamp)
CREATE TABLE IF NOT EXISTS module_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id INTEGER NOT NULL,
    module_name TEXT NOT NULL,
    state_blob BLOB NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    timestamp REAL NOT NULL,
    FOREIGN KEY (state_id) REFERENCES brain_states(id) ON DELETE CASCADE,
    UNIQUE(state_id, module_name)
);
CREATE INDEX IF NOT EXISTS idx_module_states_id ON module_states(state_id);
CREATE INDEX IF NOT EXISTS idx_module_states_module ON module_states(module_name);

-- Neuromodulator history for analysis
CREATE TABLE IF NOT EXISTS neuromodulator_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id INTEGER NOT NULL,
    step_count INTEGER NOT NULL,
    dopamine REAL NOT NULL,
    serotonin REAL NOT NULL,
    gaba REAL NOT NULL,
    noradrenaline REAL NOT NULL,
    acetylcholine REAL NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (state_id) REFERENCES brain_states(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_neuromod_log_step ON neuromodulator_log(step_count DESC);
"""


@dataclass
class BrainStateRecord:
    """Metadata about a saved brain state."""

    id: int
    step_count: int
    wall_time: float
    task: Optional[str]
    scene_hash: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ModuleStateRecord:
    """State data for a single module."""

    module_name: str
    state_data: Dict[str, Any]
    version: int = 1


class BrainStatePersistence:
    """
    Persistent storage for complete ApeBrain states.

    Saves and loads:
    - All module parameters (weights, biases, etc.)
    - Neuromodulator levels
    - Working memory contents
    - Goal stacks
    - Fear memories
    - Action values
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize the persistence layer.

        Args:
            db_path: Path to SQLite database file
        """
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # =========================================================================
    # Save Operations
    # =========================================================================

    def save(
        self,
        brain: "ApeBrain",
        step_count: int,
        task: Optional[str] = None,
        scene_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save complete brain state to database.

        Args:
            brain: ApeBrain instance to save
            step_count: Training step count
            task: Task name (e.g., "walk", "stand")
            scene_hash: Hash of the scene XML for compatibility tracking
            metadata: Additional metadata to store

        Returns:
            ID of the saved state
        """
        import sqlite3
        import time

        metadata = metadata or {}
        wall_time = time.time()

        # Start transaction
        cursor = self._conn.cursor()

        try:
            # Save brain_states metadata
            cursor.execute(
                "INSERT INTO brain_states (step_count, wall_time, task, scene_hash, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (step_count, wall_time, task, scene_hash, json.dumps(metadata)),
            )
            state_id = cursor.lastrowid

            # Save each module's state
            for module_name, module in brain.modules.items():
                module_state = self._extract_module_state(module)
                if module_state:
                    state_blob = pickle.dumps(module_state)
                    cursor.execute(
                        "INSERT INTO module_states (state_id, module_name, state_blob, version, timestamp) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (state_id, module_name, state_blob, 1, wall_time),
                    )

            # Save neuromodulator state
            neuro = brain.modules.get("neurotransmitters")
            if neuro:
                neuro_state = neuro.state
                cursor.execute(
                    "INSERT INTO neuromodulator_log "
                    "(state_id, step_count, dopamine, serotonin, gaba, noradrenaline, acetylcholine, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        state_id,
                        step_count,
                        neuro_state.dopamine,
                        neuro_state.serotonin,
                        neuro_state.gaba,
                        neuro_state.noradrenaline,
                        neuro_state.acetylcholine,
                        wall_time,
                    ),
                )

            self._conn.commit()
            return state_id

        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"Failed to save brain state: {e}")

    def _extract_module_state(self, module: Any) -> Optional[Dict[str, Any]]:
        """
        Extract serializable state from a brain module.

        Handles JAX arrays, numpy arrays, dataclasses, and primitive types.
        """
        if module is None:
            return None

        state = {}

        # Handle common module attributes
        for attr in dir(module):
            # Skip private and magic attributes
            if attr.startswith("_"):
                continue

            # Skip methods
            val = getattr(module, attr)
            if callable(val):
                continue

            # Try to serialize
            try:
                if isinstance(val, jnp.ndarray):
                    state[attr] = {"__type__": "jax_array", "value": np.array(val)}
                elif isinstance(val, np.ndarray):
                    state[attr] = {"__type__": "numpy_array", "value": val.tolist()}
                elif isinstance(val, (int, float, str, bool, list, dict, tuple)):
                    state[attr] = val
                elif hasattr(val, "__dict__"):
                    # Nested dataclass - recursively extract
                    state[attr] = self._extract_module_state(val)
                else:
                    # Try to convert to string
                    state[attr] = str(val)
            except Exception:
                # Skip unserializable attributes
                pass

        return state if state else None

    # =========================================================================
    # Load Operations
    # =========================================================================

    def load(
        self,
        brain: "ApeBrain",
        state_id: Optional[int] = None,
        step_count: Optional[int] = None,
        task: Optional[str] = None,
    ) -> bool:
        """
        Load brain state from database.

        Args:
            brain: ApeBrain instance to load into
            state_id: ID of the state to load ( mutually exclusive with step_count)
            step_count: Load state from this step count (mutually exclusive with state_id)
            task: Task name to filter by

        Returns:
            True if state was loaded, False if not found
        """
        cursor = self._conn.cursor()

        # Find the state_id
        if state_id is None:
            query = "SELECT id FROM brain_states"
            params: List[Any] = []
            if step_count is not None:
                query += " WHERE step_count = ?"
                params.append(step_count)
                if task:
                    query += " AND task = ?"
                    params.append(task)
            elif task:
                query += " WHERE task = ? ORDER BY step_count DESC LIMIT 1"
                params.append(task)
            else:
                query += " ORDER BY step_count DESC LIMIT 1"

            cursor.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return False
            state_id = row[0]

        # Load module states
        cursor.execute(
            "SELECT module_name, state_blob FROM module_states WHERE state_id = ?",
            (state_id,),
        )

        for row in cursor.fetchall():
            module_name = row[0]
            state_blob = row[1]
            module_state = pickle.loads(state_blob)
            self._restore_module_state(brain.modules.get(module_name), module_state)

        # Load neuromodulator state
        cursor.execute(
            "SELECT dopamine, serotonin, gaba, noradrenaline, acetylcholine "
            "FROM neuromodulator_log WHERE state_id = ? ORDER BY step_count DESC LIMIT 1",
            (state_id,),
        )
        row = cursor.fetchone()
        if row:
            neuro = brain.modules.get("neurotransmitters")
            if neuro:
                neuro.state.dopamine = float(row[0])
                neuro.state.serotonin = float(row[1])
                neuro.state.gaba = float(row[2])
                neuro.state.noradrenaline = float(row[3])
                neuro.state.acetylcholine = float(row[4])

        return True

    def _restore_module_state(self, module: Any, state: Dict[str, Any]) -> None:
        """
        Restore module state from serialized data.
        """
        if module is None or not state:
            return

        for attr, val in state.items():
            try:
                if isinstance(val, dict) and "__type__" in val:
                    if val["__type__"] == "jax_array":
                        setattr(module, attr, jnp.array(val["value"]))
                    elif val["__type__"] == "numpy_array":
                        setattr(module, attr, np.array(val["value"]))
                elif isinstance(val, dict):
                    # Nested state - recursively restore
                    submodule = getattr(module, attr, None)
                    self._restore_module_state(submodule, val)
                else:
                    setattr(module, attr, val)
            except Exception:
                # Skip attributes that can't be restored
                pass

    # =========================================================================
    # Query Operations
    # =========================================================================

    def list_states(
        self,
        task: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BrainStateRecord]:
        """
        List available saved states.

        Args:
            task: Filter by task name
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of BrainStateRecord objects
        """
        cursor = self._conn.cursor()
        query = "SELECT id, step_count, wall_time, task, scene_hash, metadata FROM brain_states"
        params: List[Any] = []

        if task:
            query += " WHERE task = ?"
            params.append(task)

        query += " ORDER BY step_count DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)

        states = []
        for row in cursor.fetchall():
            states.append(
                BrainStateRecord(
                    id=row[0],
                    step_count=row[1],
                    wall_time=row[2],
                    task=row[3],
                    scene_hash=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                )
            )

        return states

    def get_latest(self, task: Optional[str] = None) -> Optional[BrainStateRecord]:
        """Get the latest saved state, optionally filtered by task."""
        states = self.list_states(task=task, limit=1)
        return states[0] if states else None

    def get_neuromodulator_history(
        self,
        state_id: Optional[int] = None,
        step_count: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get neuromodulator level history for analysis.

        Args:
            state_id: Filter by state ID
            step_count: Filter by step count
            limit: Maximum number of results

        Returns:
            List of neuromodulator state dicts
        """
        cursor = self._conn.cursor()
        query = (
            "SELECT step_count, dopamine, serotonin, gaba, noradrenaline, acetylcholine, timestamp "
            "FROM neuromodulator_log"
        )
        params: List[Any] = []

        if state_id is not None:
            query += " WHERE state_id = ?"
            params.append(state_id)
        elif step_count is not None:
            query += " WHERE step_count = ?"
            params.append(step_count)

        query += " ORDER BY step_count DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "step_count": row[0],
                    "dopamine": row[1],
                    "serotonin": row[2],
                    "gaba": row[3],
                    "noradrenaline": row[4],
                    "acetylcholine": row[5],
                    "timestamp": row[6],
                }
            )

        return history

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def delete_state(self, state_id: int) -> bool:
        """Delete a saved state and all associated data."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM brain_states WHERE id = ?", (state_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_old_states(self, max_count: int = 100) -> int:
        """Keep only the most recent N states, delete older ones."""
        cursor = self._conn.cursor()

        # Get IDs of states to delete
        cursor.execute(
            "SELECT id FROM brain_states ORDER BY step_count DESC LIMIT -1 OFFSET ?",
            (max_count,),
        )
        ids_to_delete = [row[0] for row in cursor.fetchall()]

        if not ids_to_delete:
            return 0

        # Delete in chunks to avoid SQLite limits
        for chunk in [
            ids_to_delete[i : i + 100] for i in range(0, len(ids_to_delete), 100)
        ]:
            cursor.execute(
                "DELETE FROM brain_states WHERE id IN ({})".format(
                    ",".join("?" * len(chunk))
                ),
                chunk,
            )

        self._conn.commit()
        return len(ids_to_delete)

    def vacuum(self) -> None:
        """Optimize database storage."""
        self._conn.execute("VACUUM")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

import sqlite3


def create_brain_state_db(db_path: str | Path) -> BrainStatePersistence:
    """Create and return a new BrainStatePersistence instance."""
    return BrainStatePersistence(db_path)


# =============================================================================
# Tests
# =============================================================================


def test_brain_state_persistence(tmp_path: Optional[str] = None) -> None:
    """Test the BrainStatePersistence module."""
    import tempfile

    tmp = tmp_path or tempfile.mkdtemp()
    db_path = Path(tmp) / "test_brain_states.sqlite"

    # Create persistence layer
    persistence = BrainStatePersistence(db_path)

    # Test save and load with mock brain
    from brain.ape_brain import ApeBrain, ApeBrainConfig

    brain = ApeBrain(ApeBrainConfig(obs_dim=88, act_dim=24, training=False))

    # Save state
    state_id = persistence.save(
        brain, step_count=100, task="test", metadata={"note": "test save"}
    )
    assert state_id > 0, "Failed to save state"

    # List states
    states = persistence.list_states()
    assert len(states) == 1, f"Expected 1 state, got {len(states)}"
    assert states[0].step_count == 100

    # Get latest
    latest = persistence.get_latest()
    assert latest is not None
    assert latest.step_count == 100

    # Load state (into a new brain)
    brain2 = ApeBrain(ApeBrainConfig(obs_dim=88, act_dim=24, training=False))
    assert persistence.load(brain2, state_id=state_id)

    # Clean up
    persistence.close()
    db_path.unlink()

    print("✅ BrainStatePersistence tests passed")


if __name__ == "__main__":
    test_brain_state_persistence()
