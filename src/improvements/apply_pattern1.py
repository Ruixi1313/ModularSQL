#!/usr/bin/env python3
"""
Apply the DISTINCT-aware Semantic Verifier (Pattern 1) to the 99-sample
baseline snapshot. Report fixed / broken / net delta vs baseline.

Pure post-processing: no LLM calls, no pipeline re-run. Reads the existing
sql_selection snapshot, rewrites SQLs whose verifier fires, and re-evaluates.
"""

import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.improvements.distinct_verifier import (
    verify_and_fix,
    execute_sql,
)


SNAPSHOT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/sql_selection/bird/dev.snapshot.data/items.jsonl"
OUT_DIR = Path(__file__).resolve().parents[2] / "results/pattern1_distinct_intermediate"


def main():
    items = [json.loads(l) for l in open(SNAPSHOT)]
    print(f"Loaded {len(items)} samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = []      # baseline FAIL, after-fix PASS
    broken = []     # baseline PASS, after-fix FAIL
    both_pass = []
    both_fail = []
    decisions = Counter()
    per_db_baseline = defaultdict(lambda: [0, 0])
    per_db_fixed = defaultdict(lambda: [0, 0])

    rows_out = []

    for it in items:
        q_id = it["input"]["question_id"]
        db_id = it["input"]["database_id"]
        diff = it["input"]["difficulty"]
        db_path = it["input"]["database_schema"]["db_path"]
        gold_sql = it["input"]["gold_sql"]
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        question = it["input"]["question"]

        gold_rows, gold_err = execute_sql(db_path, gold_sql)
        pred_rows, pred_err = execute_sql(db_path, pred_sql)

        def _norm(rows):
            return sorted(rows, key=lambda r: str(r))
        baseline_pass = (pred_err is None and gold_err is None and
                         _norm(pred_rows) == _norm(gold_rows))

        # Apply verifier
        decision = verify_and_fix(pred_sql, db_path, pred_rows=pred_rows)
        decisions[decision.reason] += 1

        # After-fix state
        if decision.needs_distinct:
            new_pass = (_norm(decision.new_rows) == _norm(gold_rows or []))
            new_sql = decision.new_sql
        else:
            new_pass = baseline_pass
            new_sql = pred_sql

        per_db_baseline[db_id][1] += 1
        if baseline_pass: per_db_baseline[db_id][0] += 1
        per_db_fixed[db_id][1] += 1
        if new_pass: per_db_fixed[db_id][0] += 1

        if baseline_pass and not new_pass:
            broken.append((q_id, db_id, diff, question[:80]))
        elif not baseline_pass and new_pass:
            fixed.append((q_id, db_id, diff, question[:80]))
        elif baseline_pass and new_pass:
            both_pass.append(q_id)
        else:
            both_fail.append(q_id)

        rows_out.append({
            "question_id": q_id,
            "db_id": db_id,
            "difficulty": diff,
            "baseline_match": "PASS" if baseline_pass else "FAIL",
            "pattern1_match": "PASS" if new_pass else "FAIL",
            "verifier_fired": "YES" if decision.needs_distinct else "NO",
            "verifier_reason": decision.reason,
            "baseline_sql": pred_sql,
            "pattern1_sql": new_sql,
            "gold_sql": gold_sql,
        })

    # Write summary CSV
    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Summary
    n_total = len(items)
    n_baseline = len(both_pass) + len(broken)
    n_fixed = len(both_pass) + len(fixed)

    print("\n" + "=" * 70)
    print("Pattern 1: DISTINCT-aware Semantic Verifier — Result")
    print("=" * 70)
    print(f"\nBaseline accuracy:  {n_baseline}/{n_total} = {n_baseline/n_total:.2%}")
    print(f"After Pattern 1:    {n_fixed}/{n_total} = {n_fixed/n_total:.2%}")
    print(f"\nNet delta:          {len(fixed) - len(broken):+d} samples "
          f"({(len(fixed)-len(broken))/n_total*100:+.2f}pp)")
    print(f"  Fixed   (FAIL→PASS): {len(fixed)}")
    print(f"  Broken  (PASS→FAIL): {len(broken)}")

    print("\nVerifier decisions (when fired vs skipped):")
    for r, c in decisions.most_common():
        print(f"  {r:50s} {c:3d}")

    if fixed:
        print("\nFixed cases:")
        for q_id, db, diff, q in fixed:
            print(f"  [{db}/{q_id}] {diff}: {q}")
    if broken:
        print("\nBroken cases (need investigation):")
        for q_id, db, diff, q in broken:
            print(f"  [{db}/{q_id}] {diff}: {q}")

    print("\nPer-DB accuracy (baseline → pattern1):")
    for db in sorted(per_db_baseline):
        b_ok, b_n = per_db_baseline[db]
        p_ok, p_n = per_db_fixed[db]
        delta = p_ok - b_ok
        marker = "  ⭐" if delta != 0 else ""
        print(f"  {db:28s} {b_ok}/{b_n} → {p_ok}/{p_n} (Δ={delta:+d}){marker}")

    print(f"\n📄 Per-sample CSV: {csv_path}")


if __name__ == "__main__":
    main()
