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

-- ============================================================================
-- Training-artifacts index (Addendum: ape_brain general-knowledge plan)
-- ============================================================================
-- One queryable surface across runs, checkpoints, lineage, fingerprints, logs.
-- Populated by utils/training_index.py from the on-disk runs/ tree.

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    run_dir TEXT NOT NULL UNIQUE,
    mlflow_run_id TEXT,
    parent_run_id TEXT,
    scene_hash TEXT,
    fingerprint_hash TEXT,
    started_at REAL,
    ended_at REAL,
    final_steps INTEGER,
    stop_reason TEXT,
    inherited_from TEXT,
    label TEXT,
    autopilot_phase TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_fp ON runs(fingerprint_hash);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    task TEXT NOT NULL,
    scene_hash TEXT,
    path TEXT NOT NULL UNIQUE,
    safetensors_path TEXT,
    step_count INTEGER NOT NULL,
    is_keeper INTEGER NOT NULL DEFAULT 0,
    keeper_tag TEXT,
    mean_reward REAL,
    metrics_json TEXT,
    fingerprint_hash TEXT,
    wall_time REAL,
    archived INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_task_rew ON checkpoints(task, mean_reward DESC);
CREATE INDEX IF NOT EXISTS idx_ckpt_scene ON checkpoints(scene_hash);
CREATE INDEX IF NOT EXISTS idx_ckpt_step ON checkpoints(step_count);
CREATE INDEX IF NOT EXISTS idx_ckpt_keeper ON checkpoints(is_keeper);
CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id);

CREATE TABLE IF NOT EXISTS lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    task TEXT NOT NULL,
    seq INTEGER NOT NULL,
    timestamp TEXT,
    inherited_from TEXT,
    n_envs INTEGER,
    physics_json TEXT,
    env_json TEXT,
    UNIQUE(run_id, seq),
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_lineage_inh ON lineage(inherited_from);

CREATE TABLE IF NOT EXISTS fingerprints (
    hash TEXT PRIMARY KEY,
    parts_json TEXT,
    first_seen REAL,
    last_seen REAL,
    n_runs INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS log_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    wall_time REAL,
    level TEXT,
    source TEXT,
    update_count INTEGER,
    total_steps INTEGER,
    metrics_json TEXT,
    message TEXT,
    UNIQUE(run_id, source, wall_time, message),
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_log_run_time ON log_events(run_id, wall_time);
CREATE INDEX IF NOT EXISTS idx_log_level ON log_events(level);

CREATE TABLE IF NOT EXISTS autopilot_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    task TEXT,
    wall_time REAL,
    step INTEGER,
    from_phase TEXT,
    to_phase TEXT,
    fallback INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    UNIQUE(run_id, wall_time, to_phase),
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_apevt_task ON autopilot_events(task);
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

    # ------------------------------------------------------------------
    # Training-artifacts upserts (used by utils/training_index.py)
    # ------------------------------------------------------------------
    def upsert_run(self, run_dir: str, task: str, **fields) -> int:
        """Idempotent upsert of a run row. Returns runs.id."""
        run_dir = str(run_dir)
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM runs WHERE run_dir = ?", (run_dir,))
        row = cur.fetchone()
        cols = ("mlflow_run_id", "parent_run_id", "scene_hash", "fingerprint_hash",
                "started_at", "ended_at", "final_steps", "stop_reason",
                "inherited_from", "label", "autopilot_phase")
        if row is None:
            cur.execute(
                "INSERT INTO runs (task, run_dir, " + ", ".join(cols) + ") "
                "VALUES (?, ?, " + ",".join(["?"] * len(cols)) + ")",
                (task, run_dir, *(fields.get(c) for c in cols)),
            )
            run_id = int(cur.lastrowid)
        else:
            run_id = int(row["id"])
            sets = [f"{c} = COALESCE(?, {c})" for c in cols]
            cur.execute(
                f"UPDATE runs SET task = ?, {', '.join(sets)} WHERE id = ?",
                (task, *(fields.get(c) for c in cols), run_id),
            )
        self._conn.commit()
        return run_id

    def upsert_checkpoint(self, path: str, task: str, step_count: int, *,
                          run_id: Optional[int] = None,
                          scene_hash: Optional[str] = None,
                          safetensors_path: Optional[str] = None,
                          is_keeper: bool = False, keeper_tag: Optional[str] = None,
                          mean_reward: Optional[float] = None,
                          metrics: Optional[Dict[str, Any]] = None,
                          fingerprint_hash: Optional[str] = None,
                          wall_time: Optional[float] = None,
                          archived: bool = False) -> int:
        path = str(path)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO checkpoints "
            "(run_id, task, scene_hash, path, safetensors_path, step_count, is_keeper, "
            " keeper_tag, mean_reward, metrics_json, fingerprint_hash, wall_time, archived) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "  run_id           = COALESCE(excluded.run_id, checkpoints.run_id), "
            "  scene_hash       = COALESCE(excluded.scene_hash, checkpoints.scene_hash), "
            "  safetensors_path = COALESCE(excluded.safetensors_path, checkpoints.safetensors_path), "
            "  is_keeper        = excluded.is_keeper, "
            "  keeper_tag       = COALESCE(excluded.keeper_tag, checkpoints.keeper_tag), "
            "  mean_reward      = COALESCE(excluded.mean_reward, checkpoints.mean_reward), "
            "  metrics_json     = COALESCE(excluded.metrics_json, checkpoints.metrics_json), "
            "  fingerprint_hash = COALESCE(excluded.fingerprint_hash, checkpoints.fingerprint_hash), "
            "  wall_time        = COALESCE(excluded.wall_time, checkpoints.wall_time), "
            "  archived         = excluded.archived",
            (run_id, task, scene_hash, path, safetensors_path, int(step_count),
             1 if is_keeper else 0, keeper_tag, mean_reward,
             json.dumps(metrics) if metrics is not None else None,
             fingerprint_hash, wall_time, 1 if archived else 0),
        )
        self._conn.commit()
        cur.execute("SELECT id FROM checkpoints WHERE path = ?", (path,))
        return int(cur.fetchone()["id"])

    def upsert_lineage_entry(self, run_id: int, task: str, seq: int, entry: Dict[str, Any]) -> None:
        physics = entry.get("physics") or {}
        env = entry.get("env") or {}
        self._conn.execute(
            "INSERT OR REPLACE INTO lineage "
            "(run_id, task, seq, timestamp, inherited_from, n_envs, physics_json, env_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, task, int(seq), entry.get("timestamp"),
             entry.get("inherited_from"), entry.get("n_envs"),
             json.dumps(physics), json.dumps(env)),
        )
        self._conn.commit()

    def upsert_fingerprint(self, fp_hash: str, parts: Dict[str, str], wall_time: float) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT first_seen, n_runs FROM fingerprints WHERE hash = ?", (fp_hash,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO fingerprints (hash, parts_json, first_seen, last_seen, n_runs) "
                "VALUES (?,?,?,?,1)",
                (fp_hash, json.dumps(parts), wall_time, wall_time),
            )
        else:
            cur.execute(
                "UPDATE fingerprints SET parts_json = ?, last_seen = ?, n_runs = n_runs + 1 "
                "WHERE hash = ?",
                (json.dumps(parts), wall_time, fp_hash),
            )
        self._conn.commit()

    def insert_log_event(self, run_id: Optional[int], wall_time: float, level: str,
                         source: str, message: str = "", *,
                         update_count: Optional[int] = None,
                         total_steps: Optional[int] = None,
                         metrics: Optional[Dict[str, Any]] = None) -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO log_events "
                "(run_id, wall_time, level, source, update_count, total_steps, metrics_json, message) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (run_id, wall_time, level, source, update_count, total_steps,
                 json.dumps(metrics) if metrics is not None else None, message),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass

    def insert_autopilot_event(self, run_id: Optional[int], task: str, wall_time: float, *,
                               step: Optional[int] = None,
                               from_phase: Optional[str] = None,
                               to_phase: Optional[str] = None,
                               fallback: bool = False, reason: str = "") -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO autopilot_events "
                "(run_id, task, wall_time, step, from_phase, to_phase, fallback, reason) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (run_id, task, wall_time, step, from_phase, to_phase,
                 1 if fallback else 0, reason),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass

    def close(self):
        if self._conn: self._conn.close(); self._conn = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def connect_readonly(policy_db_path: str = "runs/ape_policy_db.sqlite",
                     attach: Optional[Dict[str, str]] = None) -> sqlite3.Connection:
    """Open the policy DB read-only and ATTACH the brain + mlflow stores.

    Default attached aliases (if files exist):
      m -> mlflow.db
      e -> data/brain/episodic.sqlite
      s -> data/brain/semantic.sqlite
      b -> data/brain/brain_state.sqlite

    Returns a sqlite3.Connection with row_factory=sqlite3.Row.
    """
    if attach is None:
        attach = {
            "m": "mlflow.db",
            "e": "data/brain/episodic.sqlite",
            "s": "data/brain/semantic.sqlite",
            "b": "data/brain/brain_state.sqlite",
        }
    uri = f"file:{policy_db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    for alias, path in attach.items():
        if Path(path).exists():
            try:
                conn.execute(f"ATTACH DATABASE 'file:{path}?mode=ro' AS {alias}")
            except sqlite3.OperationalError:
                pass
    return conn

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
