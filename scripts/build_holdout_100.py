#!/usr/bin/env python3
"""
Build a hold-out validation set of 100 BIRD-Dev questions that does NOT
overlap with the development 99 samples (first 9 per DB).

Selection: questions 9..18 per database (next 9 after the dev set),
plus an extra question from any DB to reach 100. Total: 11 × 9 + 1 = 100.
"""
import json
from collections import defaultdict
from pathlib import Path


DEV_JSON = Path(__file__).resolve().parents[1] / "data/bird/dev/dev.json"
OUT = Path(__file__).resolve().parents[1] / "experiments/holdout_100.json"


def main():
    data = json.loads(DEV_JSON.read_text())
    by_db = defaultdict(list)
    for idx, item in enumerate(data):
        by_db[item.get("db_id")].append(idx)

    # Take questions 9..17 (next 9 per DB after the dev 0..8 used as max_samples_per_db=9)
    holdout_indices = []
    holdout_by_db = {}
    for db_id, idxs in sorted(by_db.items()):
        next_9 = idxs[9:18]
        holdout_indices.extend(next_9)
        holdout_by_db[db_id] = next_9
        print(f"  {db_id:30s}  count_in_dev={len(idxs):4d}  holdout={len(next_9)}")

    # If we have 99, pad with 1 more from the DB with the most questions
    if len(holdout_indices) < 100:
        biggest_db = max(by_db, key=lambda k: len(by_db[k]))
        extra = by_db[biggest_db][18]
        holdout_indices.append(extra)
        holdout_by_db[biggest_db].append(extra)
        print(f"  + 1 extra from {biggest_db}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "holdout_100",
        "description": "Held-out validation set: questions 9..18 of each BIRD-Dev DB, non-overlapping with max_samples_per_db=9 dev set.",
        "sample_indices": sorted(holdout_indices),
        "per_db": holdout_by_db,
        "total": len(holdout_indices),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n  Wrote {len(holdout_indices)} indices → {OUT}")


if __name__ == "__main__":
    main()
