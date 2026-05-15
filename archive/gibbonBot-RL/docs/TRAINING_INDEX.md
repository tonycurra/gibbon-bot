# Training Data Index

A single SQLite file (`runs/ape_policy_db.sqlite`) that indexes every
training artifact on disk so you can find/query past runs in seconds
instead of `glob`+`cat`+regex.

## What's in it

Six tables alongside the original `policies` / `scene_metadata`:

| Table | Holds |
|---|---|
| `runs` | one row per `runs/ape_<task>_mjx/` (task, mlflow_run_id, scene/fingerprint hashes, started/ended, autopilot_phase) |
| `checkpoints` | every `mjx_*.npz` + `.safetensors` (live, keeper, archived) with sidecar metrics |
| `lineage` | one row per `lineage.json` entry (timestamps, inheritance, physics, env constants) |
| `fingerprints` | env/code identity hashes; `n_runs` per hash |
| `log_events` | structured rows from `events.jsonl` (preferred) or regex-parsed `train.log`. Levels: `METRIC`, `KEEPER`, `BRAIN`, `STOP`, `AUTOPILOT`, `INFO` |
| `autopilot_events` | phase transitions and fallbacks |

All upserts are idempotent — re-indexing the same disk state produces
no duplicate rows.

## Cross-DB joins

`brain.policy_database.connect_readonly()` opens `runs/ape_policy_db.sqlite`
read-only and `ATTACH`es:

| Alias | File |
|---|---|
| `m` | `mlflow.db` |
| `e` | `data/brain/episodic.sqlite` |
| `s` | `data/brain/semantic.sqlite` |
| `b` | `data/brain/brain_state.sqlite` |

Now `JOIN` across them in one query — no copying.

## How it's populated

- **Trainer hook**: `train_mjx.py` calls `_reindex_silently(RUNS)` after every
  periodic checkpoint, every keeper save, and at exit. Best-effort — never
  blocks training.
- **Structured events**: `train_mjx.py` writes one JSON line per
  `METRIC` / `KEEPER` / `BRAIN` / `STOP` event to `<run_dir>/events.jsonl`
  via `utils.structured_log.log_event(...)`. The indexer prefers this file
  and falls back to regex-parsing `train.log` for legacy runs.
- **Manual rebuild**: `python scripts/reindex.py` walks every
  `runs/ape_*_mjx/` and rebuilds. Use after restoring from Scaleway or
  changing schema.

## Querying — `scripts/q.py`

```sh
python scripts/q.py stats                       # row counts across all tables (incl. attached)
python scripts/q.py runs                        # every indexed run
python scripts/q.py runs --task walk --since 2026-05-01
python scripts/q.py best --task stand -k 5      # top-5 checkpoints by mean_reward
python scripts/q.py keepers --task walk --metric mean_upright --gt 0.95
python scripts/q.py lineage runs/ape_stand_mjx/mjx_000000098304.npz   # walk inherited_from chain
python scripts/q.py phases --task walk          # autopilot phase transitions + fallbacks
python scripts/q.py logs --run 5 --level KEEPER
python scripts/q.py episodes --near my_obs.npy -k 10   # cross-DB → episodic.sqlite (k-NN by spatial code)
```

Each subcommand is one parameterized SQL string — see `scripts/q.py` for
the bodies if you want to write your own.

## Direct SQL

```sh
sqlite3 runs/ape_policy_db.sqlite
> ATTACH DATABASE 'mlflow.db' AS m;
> ATTACH DATABASE 'data/brain/episodic.sqlite' AS e;
> SELECT c.path, c.task, c.step_count, c.mean_reward
>   FROM checkpoints c
>   JOIN runs r ON r.id = c.run_id
>  WHERE r.autopilot_phase = 'C' AND c.is_keeper = 1
>  ORDER BY c.mean_reward DESC LIMIT 10;
```

## Files

- `brain/policy_database.py` — schema + upsert helpers + `connect_readonly()`
- `utils/training_index.py` — `index_run()`, `index_all()`
- `utils/structured_log.py` — `log_event()` (one-line JSONL writer)
- `scripts/reindex.py` — full rebuild CLI
- `scripts/q.py` — query CLI

## Out of scope

Raw policy weights stay in `.npz` / `.safetensors`. Free-form `train.log`
text stays a file (use `grep`). MLflow time-series metrics stay in
`mlflow.db` and are joinable via `ATTACH`.
