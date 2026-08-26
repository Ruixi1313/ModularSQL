#!/usr/bin/env python3
"""Generate per-sample summary CSV for the full BIRD-Dev run with Pattern 1 applied."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.distinct_verifier import verify_and_fix, execute_sql

S = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"
OUT = Path(__file__).resolve().parents[2] / "results/ModularSQL_Baseline_dev1534_20260514/summary.csv"


def norm(rows):
    if rows is None:
        return None
    return sorted([tuple(r) for r in rows], key=lambda r: str(r))


def main():
    items = [json.loads(l) for l in open(S)]
    print(f"Loaded {len(items)} items", flush=True)

    rows = []
    for i, it in enumerate(items, 1):
        db = it["input"]["database_schema"]["db_path"]
        gold_rows, _ = execute_sql(db, it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        gold = norm(gold_rows)
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        pred_rows, _ = execute_sql(db, pred_sql)
        base_pass = pred_rows is not None and norm(pred_rows) == gold

        decision = verify_and_fix(pred_sql, db, pred_rows=pred_rows)
        if decision.needs_distinct:
            p1_sql = decision.new_sql
            p1_pass = norm(decision.new_rows) == gold
        else:
            p1_sql = pred_sql
            p1_pass = base_pass

        rows.append({
            "qid": it["input"]["question_id"],
            "db_id": it["input"]["database_id"],
            "difficulty": it["input"].get("difficulty"),
            "baseline_match": "PASS" if base_pass else "FAIL",
            "p1_match": "PASS" if p1_pass else "FAIL",
            "p1_fired": "YES" if decision.needs_distinct else "NO",
            "p1_reason": decision.reason,
            "baseline_sql": pred_sql[:500],
            "p1_sql": p1_sql[:500],
            "gold_sql": it["input"]["gold_sql"][:500],
        })
        if i % 200 == 0:
            print(f"  ...{i}/{len(items)} processed", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Wrote {OUT} with {len(rows)} rows")


if __name__ == "__main__":
    main()
