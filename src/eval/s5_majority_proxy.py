#!/usr/bin/env python3
"""
Compute a MAJORITY-VOTE PROXY accuracy from S5 (SQL Generation) candidates
on the in-progress workspace_full run. Read-only — does not disturb the running pipeline.

For each completed item:
  - take its 12 sql_candidates from sql_generation artifact
  - execute each on the DB, group by result tuple
  - the largest group's result becomes the majority vote
  - compare against gold

This is a NOISY proxy for what S7 Selection would pick. Treat as directional only.
"""
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
ART = _ROOT / "external/DeepEye-SQL/workspace_full/sql_generation/bird/dev.artifacts/sql_generation.jsonl"
SCHEMA_SNAP = _ROOT / "external/DeepEye-SQL/workspace_full/schema_linking/bird/dev.snapshot.data/items.jsonl"


def _norm(rows):
    if rows is None:
        return None
    return tuple(sorted([tuple(r) for r in rows], key=lambda r: str(r)))


def execute_one(db_path, sql, timeout=10):
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = None
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return _norm(rows)
    except Exception:
        return None


def main():
    if not ART.exists() or not SCHEMA_SNAP.exists():
        print("Required files missing")
        sys.exit(1)

    # Build input map from schema_linking snapshot (has gold_sql + db_path)
    print("Loading schema_linking snapshot for gold SQLs...", flush=True)
    inputs = {}
    for line in open(SCHEMA_SNAP):
        item = json.loads(line)
        qid = item["input"]["question_id"]
        inputs[qid] = {
            "gold": item["input"]["gold_sql"],
            "db_path": item["input"]["database_schema"]["db_path"],
            "difficulty": item["input"].get("difficulty", "unknown"),
            "db_id": item["input"]["database_id"],
        }
    print(f"  Loaded {len(inputs)} gold references\n")

    # Load S5 artifact entries
    print(f"Loading S5 candidates from {ART}...", flush=True)
    candidates = {}
    for line in open(ART):
        if not line.strip(): continue
        entry = json.loads(line)
        qid = entry["item_id"]
        try: qid = int(qid)
        except (TypeError, ValueError): pass
        cands = entry.get("stage_artifact", {}).get("sql_candidates", [])
        if cands:
            candidates[qid] = cands  # latest wins
    n_done = len(candidates)
    print(f"  Found {n_done} items with S5 candidates done\n")

    # Compute majority proxy
    print("Computing majority-vote proxy (executing candidates)...", flush=True)
    t = time.time()
    matches = 0
    n_evaluable = 0
    by_diff = defaultdict(lambda: [0, 0])
    by_db = defaultdict(lambda: [0, 0])

    for qid in sorted(candidates):
        if qid not in inputs: continue
        info = inputs[qid]
        gold_rows = execute_one(info["db_path"], info["gold"])
        if gold_rows is None: continue  # gold itself fails; skip
        n_evaluable += 1

        results = Counter()
        for cand in candidates[qid]:
            r = execute_one(info["db_path"], cand)
            if r is not None:
                results[r] += 1
        if not results:
            # all candidates exec-errored
            by_diff[info["difficulty"]][1] += 1
            by_db[info["db_id"]][1] += 1
            continue

        # Majority winner
        winner, _ = results.most_common(1)[0]
        ok = (winner == gold_rows)
        if ok: matches += 1

        by_diff[info["difficulty"]][1] += 1
        if ok: by_diff[info["difficulty"]][0] += 1
        by_db[info["db_id"]][1] += 1
        if ok: by_db[info["db_id"]][0] += 1

        if n_evaluable % 200 == 0:
            print(f"  ...{n_evaluable} done, current proxy acc = "
                  f"{matches}/{n_evaluable} = {matches/n_evaluable:.2%}", flush=True)

    elapsed = time.time() - t
    print(f"  Completed eval of {n_evaluable} items in {elapsed:.0f}s\n")

    print("=" * 70)
    print(f"S5 Majority-Vote PROXY (read-only on in-progress run)")
    print("=" * 70)
    print(f"Items evaluated:     {n_evaluable} of {n_done} S5-completed")
    print(f"Majority correct:    {matches}/{n_evaluable} = {matches/n_evaluable:.2%}")
    print()
    print("By difficulty:")
    for d in ["simple","moderate","challenging"]:
        if d in by_diff:
            ok, tot = by_diff[d]
            print(f"  {d:<12} {ok}/{tot} = {ok/tot:.1%}" if tot else f"  {d:<12} 0/0")
    print()
    print("Top/bottom DBs:")
    db_acc = []
    for db, (ok, tot) in by_db.items():
        if tot >= 5:
            db_acc.append((ok/tot, ok, tot, db))
    db_acc.sort(reverse=True)
    print("  Best:")
    for r, ok, tot, db in db_acc[:3]:
        print(f"    {db:<28} {ok}/{tot} = {r:.1%}")
    print("  Worst:")
    for r, ok, tot, db in db_acc[-3:]:
        print(f"    {db:<28} {ok}/{tot} = {r:.1%}")
    print()
    print("⚠ This is a PROXY. Final S7 accuracy may differ ±5pp.")
    print("⚠ Read-only on artifact files. Did NOT disturb the running pipeline.")


if __name__ == "__main__":
    main()
