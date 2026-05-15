#!/usr/bin/env python3
"""Rebuild the training-artifacts index from disk.

    python scripts/reindex.py                # walk all runs/ape_*_mjx/
    python scripts/reindex.py runs/ape_walk_mjx
"""
import sys
from pathlib import Path

# Allow running from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.training_index import index_all, index_run
from brain.policy_database import PolicyDatabase


def main():
    db_path = "runs/ape_policy_db.sqlite"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir() and (target / "lineage.json").exists():
            db = PolicyDatabase(db_path)
            rid = index_run(target, db)
            db.close()
            print(f"indexed {target} → run_id={rid}")
            return
        if target.is_dir():
            res = index_all(str(target), db_path=db_path)
            print(f"indexed {len(res)} runs from {target}:", list(res.keys()))
            return
        sys.exit(f"not a directory: {target}")
    res = index_all("runs", db_path=db_path)
    print(f"indexed {len(res)} runs:", list(res.keys()))


if __name__ == "__main__":
    main()
