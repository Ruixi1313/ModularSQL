#!/usr/bin/env python3
"""
Quick experiment: for each failed sample on the 99-sample baseline,
test whether adding DISTINCT to the SELECT would yield the gold result.

No LLM calls. Just SQL rewrite + execution.
This sets the upper bound for what a "DISTINCT-aware Semantic Verifier" could achieve.
"""

import json
import re
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict


SNAPSHOT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/sql_selection/bird/dev.snapshot.data/items.jsonl"


SELECT_RE = re.compile(r"^(\s*SELECT)(\s+)(?!DISTINCT\b)", re.IGNORECASE)


def add_distinct(sql: str) -> str:
    """Insert DISTINCT after the first SELECT keyword (skip if already present)."""
    return SELECT_RE.sub(r"\1 DISTINCT ", sql, count=1)


def execute(conn, sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return sorted([tuple(r) for r in cur.fetchall()]), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def already_has_distinct(sql: str) -> bool:
    return bool(re.search(r"\bSELECT\s+DISTINCT\b", sql, re.IGNORECASE))


def has_aggregate_only(sql: str) -> bool:
    """If outermost SELECT is purely aggregate (COUNT/SUM/...), DISTINCT on row level doesn't matter."""
    # Heuristic: SELECT clause contains aggregate and no GROUP BY → single row
    sql_u = sql.upper()
    if "GROUP BY" in sql_u:
        return False
    return any(f"{agg}(" in sql_u for agg in ("COUNT", "SUM", "AVG", "MIN", "MAX"))


def main():
    items = [json.loads(l) for l in open(SNAPSHOT)]
    print(f"Loaded {len(items)} samples\n")

    stats = Counter()
    fixed_by_distinct = []
    broken_by_distinct = []
    no_change = []

    for it in items:
        db_id = it["input"]["database_id"]
        db_path = it["input"]["database_schema"]["db_path"]
        q_id = it["input"]["question_id"]
        gold = it["input"]["gold_sql"]
        pred = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]

        conn = sqlite3.connect(db_path)
        gold_rows, _ = execute(conn, gold)
        pred_rows, _ = execute(conn, pred)

        # Categorize current state
        if pred_rows == gold_rows:
            stats["already_matches"] += 1
            continue

        # Skip cases where DISTINCT would not apply
        if already_has_distinct(pred):
            stats["pred_already_has_distinct_but_fails"] += 1
            continue
        if has_aggregate_only(pred):
            stats["aggregate_no_distinct_needed"] += 1
            continue

        # Try adding DISTINCT
        new_sql = add_distinct(pred)
        new_rows, new_err = execute(conn, new_sql)

        if new_err:
            stats["distinct_caused_exec_error"] += 1
            continue

        if new_rows == gold_rows:
            stats["fixed_by_adding_distinct"] += 1
            fixed_by_distinct.append({
                "db": db_id, "q_id": q_id,
                "difficulty": it["input"]["difficulty"],
                "question": it["input"]["question"][:100],
                "pred": pred[:150],
                "fixed": new_sql[:150],
            })
        else:
            stats["distinct_does_not_fix"] += 1
            no_change.append((db_id, q_id, it["input"]["difficulty"]))

    print("=" * 70)
    print("DISTINCT-fix experiment results")
    print("=" * 70)
    for k, v in stats.most_common():
        print(f"  {k:42s} {v:3d}")

    print(f"\n  Total samples:                      {sum(stats.values())}")

    fixed_n = stats["fixed_by_adding_distinct"]
    total = len(items)
    current_correct = stats["already_matches"]
    potential = current_correct + fixed_n
    print(f"\n  Current accuracy:                   {current_correct}/{total} = {current_correct/total:.1%}")
    print(f"  After DISTINCT fix:                 {potential}/{total} = {potential/total:.1%}")
    print(f"  Improvement (upper bound):          +{fixed_n} samples = +{fixed_n/total*100:.1f}pp")

    if fixed_by_distinct:
        print("\n" + "=" * 70)
        print(f"Samples fixed by adding DISTINCT ({len(fixed_by_distinct)})")
        print("=" * 70)
        for f in fixed_by_distinct:
            print(f"\n  [{f['db']}/{f['q_id']}] {f['difficulty']}")
            print(f"    Q: {f['question']}")
            print(f"    BEFORE: {f['pred']}")
            print(f"    AFTER:  {f['fixed']}")


if __name__ == "__main__":
    main()
