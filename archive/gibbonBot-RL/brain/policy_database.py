import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    scene_hash TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL UNIQUE,
    step_count INTEGER NOT NULL,
    mean_reward REAL,
    max_reward REAL,
    stability_score REAL DEFAULT 0.0,
    obs_dim INTEGER NOT NULL,
    act_dim INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_task ON policies(task);
CREATE INDEX IF NOT EXISTS idx_scene_hash ON policies(scene_hash);
CREATE INDEX IF NOT EXISTS idx_reward ON policies(mean_reward DESC);
CREATE INDEX IF NOT EXISTS idx_timestamp ON policies(timestamp DESC);
CREATE TABLE IF NOT EXISTS scene_metadata (
    scene_hash TEXT PRIMARY KEY,
    scene_path TEXT NOT NULL,
    description TEXT,
    timestamp INTEGER NOT NULL
);
"""

@dataclass
class PolicyRecord:
    id: int
    task: str
    scene_hash: str
    checkpoint_path: str
    step_count: int
    mean_reward: float
    max_reward: float
    stability_score: float
    obs_dim: int
    act_dim: int
    timestamp: int
    metadata: Dict[str, Any]

    @classmethod
    def from_row(cls, row):
        return cls(
            id=row["id"], task=row["task"], scene_hash=row["scene_hash"],
            checkpoint_path=row["checkpoint_path"], step_count=row["step_count"],
            mean_reward=row["mean_reward"], max_reward=row["max_reward"],
            stability_score=row["stability_score"], obs_dim=row["obs_dim"],
            act_dim=row["act_dim"], timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

class PolicyDatabase:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def compute_scene_hash(scene_path):
        content = Path(scene_path).read_text()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_policy(self, task, checkpoint_path, step_count, mean_reward=None,
                    max_reward=None, stability_score=None, scene_path=None,
                    obs_dim=88, act_dim=24, metadata=None):
        checkpoint_path = str(checkpoint_path)
        timestamp = int(Path(checkpoint_path).stat().st_mtime) if Path(checkpoint_path).exists() else 0
        scene_path = scene_path or "data/scene.xml"
        scene_hash = self.compute_scene_hash(scene_path)
        self._store_scene_metadata(scene_path, scene_hash)
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO policies (task, scene_hash, checkpoint_path,"
            "step_count, mean_reward, max_reward, stability_score, obs_dim, act_dim,"
            "timestamp, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (task, scene_hash, checkpoint_path, step_count, mean_reward, max_reward,
             stability_score, obs_dim, act_dim, timestamp, json.dumps(metadata or {})))
        self._conn.commit()
        cursor.execute("SELECT * FROM policies WHERE checkpoint_path = ?", (checkpoint_path,))
        row = cursor.fetchone()
        return PolicyRecord.from_row(row) if row else PolicyRecord(
            id=0, task=task, scene_hash=scene_hash, checkpoint_path=checkpoint_path,
            step_count=step_count, mean_reward=mean_reward or 0.0, max_reward=max_reward or 0.0,
            stability_score=stability_score or 0.0, obs_dim=obs_dim, act_dim=act_dim,
            timestamp=timestamp, metadata=metadata or {})

    def _store_scene_metadata(self, scene_path, scene_hash):
        scene_path = str(scene_path)
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO scene_metadata (scene_hash, scene_path, description, timestamp)"
            "VALUES (?,?,?,?)",
            (scene_hash, scene_path, "", int(Path(scene_path).stat().st_mtime)))
        self._conn.commit()

    def query_best(self, task=None, limit=1, min_reward=None):
        cursor = self._conn.cursor()
        query = "SELECT * FROM policies"
        params = []
        conditions = []
        if task: conditions.append("task = ?"); params.append(task)
        if min_reward: conditions.append("mean_reward >= ?"); params.append(min_reward)
        if conditions: query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY mean_reward DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        return [PolicyRecord.from_row(row) for row in cursor.fetchall()]

    def query_by_scene(self, scene_hash=None, scene_path=None, limit=10):
        if scene_path: scene_hash = self.compute_scene_hash(scene_path)
        if not scene_hash: return []
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM policies WHERE scene_hash = ? ORDER BY mean_reward DESC LIMIT ?",
                       (scene_hash, limit))
        return [PolicyRecord.from_row(row) for row in cursor.fetchall()]

    def query_by_task(self, task, limit=10):
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM policies WHERE task = ? ORDER BY mean_reward DESC LIMIT ?",
                       (task, limit))
        return [PolicyRecord.from_row(row) for row in cursor.fetchall()]

    def query_all(self, limit=100):
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM policies ORDER BY mean_reward DESC LIMIT ?", (limit,))
        return [PolicyRecord.from_row(row) for row in cursor.fetchall()]

    def count(self, task=None):
        cursor = self._conn.cursor()
        if task: cursor.execute("SELECT COUNT(*) FROM policies WHERE task = ?", (task,))
        else: cursor.execute("SELECT COUNT(*) FROM policies")
        return cursor.fetchone()[0]

    def close(self):
        if self._conn: self._conn.close(); self._conn = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

def import_checkpoints_from_runs(db_path, runs_dir="runs", task=None):
    db = PolicyDatabase(db_path)
    runs_dir = Path(runs_dir)
    imported = 0
    for task_dir in [d for d in runs_dir.iterdir() if d.is_dir() and "ape_" in d.name]:
        if task and f"ape_{task}_mjx" not in task_dir.name: continue
        task_name = task_dir.name.replace("ape_", "").replace("_mjx", "")
        for npz_file in task_dir.glob("mjx_*.npz"):
            try: step_count = int(npz_file.stem.split("_")[1])
            except (IndexError, ValueError): continue
            db.add_policy(task=task_name, checkpoint_path=npz_file, step_count=step_count,
                          scene_path="data/scene.xml",
                          metadata={"source": "import_from_runs", "filename": npz_file.name})
            imported += 1
    db.close()
    return imported
