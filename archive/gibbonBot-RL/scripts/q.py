#!/usr/bin/env python3
"""q.py — quick queries against the unified training-artifacts index.

    q.py runs                                      list every run
    q.py runs --task walk --since 2026-05-01
    q.py keepers --task walk                       all keepers, newest first
    q.py keepers --task walk --metric mean_upright --gt 0.95
    q.py best --task walk -k 5                     top-5 by mean_reward
    q.py lineage <checkpoint_path|run_dir>         walk inherited_from chain
    q.py phases [--task walk]                      autopilot phase transitions
    q.py episodes --near <obs.npy> -k 10           cross-DB → episodic.sqlite
    q.py logs --run <id|run_dir> [--level KEEPER]  events for a run
    q.py stats                                     row counts across tables

The connection opens the policy DB read-only and ATTACHes mlflow.db,
data/brain/episodic.sqlite, data/brain/semantic.sqlite,
data/brain/brain_state.sqlite as `m`, `e`, `s`, `b` aliases when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.policy_database import connect_readonly  # noqa: E402


def _print_rows(rows, cols=None, max_width=120):
    rows = list(rows)
    if not rows:
        print("(no rows)")
        return
    if cols is None:
        cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c] if r[c] is not None else "")) for r in rows)) for c in cols}
    widths = {c: min(w, max_width // max(1, len(cols))) for c, w in widths.items()}
    line = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            s = "" if v is None else (json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else str(v))
            if len(s) > widths[c]:
                s = s[: widths[c] - 1] + "…"
            vals.append(f"{s:<{widths[c]}}")
        print("  ".join(vals))
    print(f"\n({len(rows)} rows)")


def _parse_since(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        sys.exit(f"--since: not ISO-8601: {s}")


def cmd_runs(conn, args):
    where, params = [], []
    if args.task:
        where.append("task = ?"); params.append(args.task)
    if args.since:
        where.append("started_at >= ?"); params.append(_parse_since(args.since))
    if args.phase:
        where.append("autopilot_phase = ?"); params.append(args.phase)
    sql = "SELECT id, task, run_dir, autopilot_phase, fingerprint_hash, started_at FROM runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_at DESC"
    _print_rows(conn.execute(sql, params).fetchall())


def cmd_keepers(conn, args):
    sql = ("SELECT path, task, step_count, mean_reward, metrics_json, wall_time "
           "FROM checkpoints WHERE is_keeper = 1")
    params = []
    if args.task:
        sql += " AND task = ?"; params.append(args.task)
    if args.metric and args.gt is not None:
        sql += f" AND CAST(json_extract(metrics_json, '$.{args.metric}') AS REAL) > ?"
        params.append(float(args.gt))
    if args.metric and args.lt is not None:
        sql += f" AND CAST(json_extract(metrics_json, '$.{args.metric}') AS REAL) < ?"
        params.append(float(args.lt))
    sql += " ORDER BY wall_time DESC LIMIT ?"
    params.append(args.k)
    _print_rows(conn.execute(sql, params).fetchall())


def cmd_best(conn, args):
    sql = ("SELECT path, task, step_count, mean_reward, is_keeper "
           "FROM checkpoints WHERE mean_reward IS NOT NULL")
    params = []
    if args.task:
        sql += " AND task = ?"; params.append(args.task)
    sql += " ORDER BY mean_reward DESC LIMIT ?"
    params.append(args.k)
    _print_rows(conn.execute(sql, params).fetchall())


def cmd_lineage(conn, args):
    """Walk the inherited_from chain."""
    target = args.target
    chain, seen = [], set()
    cur = target
    while cur and cur not in seen:
        seen.add(cur)
        # Try checkpoint match first, then run_dir
        row = conn.execute(
            "SELECT c.path AS path, c.step_count, r.run_dir, r.task, r.inherited_from, r.fingerprint_hash "
            "FROM checkpoints c LEFT JOIN runs r ON r.id = c.run_id "
            "WHERE c.path = ? LIMIT 1", (cur,)
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT NULL AS path, NULL AS step_count, run_dir, task, inherited_from, fingerprint_hash "
                "FROM runs WHERE run_dir = ? LIMIT 1", (cur,)
            ).fetchone()
        if row is None:
            chain.append({"target": cur, "note": "(not indexed)"})
            break
        chain.append(dict(row))
        nxt = row["inherited_from"]
        if not nxt or nxt == "scratch":
            break
        cur = nxt
    for i, step in enumerate(chain):
        prefix = "└─" if i == len(chain) - 1 else "├─"
        print(f"{prefix} {step}")


def cmd_phases(conn, args):
    sql = ("SELECT a.task, a.step, a.from_phase, a.to_phase, a.fallback, a.reason, "
           "       datetime(a.wall_time,'unixepoch') AS at_iso "
           "FROM autopilot_events a")
    params = []
    if args.task:
        sql += " WHERE a.task = ?"; params.append(args.task)
    sql += " ORDER BY a.wall_time DESC LIMIT ?"
    params.append(args.k)
    _print_rows(conn.execute(sql, params).fetchall())


def cmd_episodes(conn, args):
    """Cross-DB query: nearest episodic memories to a query observation."""
    if "e" not in [r[1] for r in conn.execute("PRAGMA database_list").fetchall()]:
        sys.exit("episodic.sqlite is not attached (data/brain/episodic.sqlite missing)")
    import numpy as np
    obs = np.load(args.near).astype("float32")
    # Pull all embeddings on-host (fine up to ~100k); cosine-rank.
    rows = conn.execute("SELECT id, task, reward, embedding FROM e.episodes").fetchall()
    if not rows:
        print("(episodic store is empty)"); return
    embs = np.stack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])
    qn = obs / (np.linalg.norm(obs) + 1e-8)
    if embs.shape[1] != qn.shape[0]:
        sys.exit(f"obs dim {qn.shape[0]} != embedding dim {embs.shape[1]}; "
                 f"pass an embedding (e.g. spatial_code), not a raw obs")
    sims = embs @ qn
    idx = np.argsort(-sims)[: args.k]
    out = [{"id": rows[i]["id"], "task": rows[i]["task"], "reward": rows[i]["reward"],
            "sim": float(sims[i])} for i in idx]
    _print_rows(out, cols=["id", "task", "reward", "sim"])


def cmd_logs(conn, args):
    sql = ("SELECT datetime(wall_time,'unixepoch') AS at_iso, level, "
           "       update_count, total_steps, message "
           "FROM log_events WHERE 1=1")
    params = []
    if args.run is not None:
        try:
            rid = int(args.run)
            sql += " AND run_id = ?"; params.append(rid)
        except ValueError:
            sql += " AND run_id = (SELECT id FROM runs WHERE run_dir = ?)"
            params.append(args.run)
    if args.level:
        sql += " AND level = ?"; params.append(args.level)
    sql += " ORDER BY wall_time DESC LIMIT ?"
    params.append(args.k)
    _print_rows(conn.execute(sql, params).fetchall())


def cmd_stats(conn, args):
    tables = ["runs", "checkpoints", "lineage", "fingerprints",
              "log_events", "autopilot_events", "policies", "scene_metadata"]
    out = []
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        out.append({"table": t, "rows": n})
    # Cross-DB counts when attached
    attached = {r[1]: r[2] for r in conn.execute("PRAGMA database_list").fetchall()}
    for alias, label in (("m", "mlflow.runs"), ("e", "episodic.episodes"),
                         ("s", "semantic.semantic_facts"), ("b", "brain_states")):
        if alias not in attached:
            continue
        try:
            if alias == "m":
                n = conn.execute("SELECT COUNT(*) FROM m.runs").fetchone()[0]
            elif alias == "e":
                n = conn.execute("SELECT COUNT(*) FROM e.episodes").fetchone()[0]
            elif alias == "s":
                # try both possible table names
                try:
                    n = conn.execute("SELECT COUNT(*) FROM s.semantic_facts").fetchone()[0]
                except Exception:
                    n = conn.execute("SELECT COUNT(*) FROM s.trees").fetchone()[0]
            elif alias == "b":
                n = conn.execute("SELECT COUNT(*) FROM b.brain_states").fetchone()[0]
            out.append({"table": label, "rows": n})
        except Exception as ex:
            out.append({"table": label, "rows": f"err: {ex}"})
    _print_rows(out, cols=["table", "rows"])


def main():
    p = argparse.ArgumentParser(description="Query the training-artifacts index.")
    p.add_argument("--db", default="runs/ape_policy_db.sqlite")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("runs"); s.add_argument("--task"); s.add_argument("--since"); s.add_argument("--phase")
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("keepers"); s.add_argument("--task"); s.add_argument("--metric"); s.add_argument("--gt", type=float); s.add_argument("--lt", type=float); s.add_argument("-k", type=int, default=20)
    s.set_defaults(fn=cmd_keepers)

    s = sub.add_parser("best"); s.add_argument("--task"); s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_best)

    s = sub.add_parser("lineage"); s.add_argument("target")
    s.set_defaults(fn=cmd_lineage)

    s = sub.add_parser("phases"); s.add_argument("--task"); s.add_argument("-k", type=int, default=20)
    s.set_defaults(fn=cmd_phases)

    s = sub.add_parser("episodes"); s.add_argument("--near", required=True, help="path to .npy embedding")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(fn=cmd_episodes)

    s = sub.add_parser("logs"); s.add_argument("--run"); s.add_argument("--level"); s.add_argument("-k", type=int, default=50)
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)

    args = p.parse_args()
    conn = connect_readonly(args.db)
    try:
        args.fn(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
