#!/usr/bin/env python3
"""
Apply the two-stage DISTINCT-aware Semantic Verifier (Pattern 1B):
  Stage 1 (rule):  JOIN + no-aggregate + duplicates  → candidate
  Stage 2 (LLM):   Qwen3-Coder judges whether DISTINCT is semantically correct

LLM is invoked ONLY for rule-fired candidates (~8 out of 99), saving tokens.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.improvements.distinct_verifier import (
    verify_and_fix,
    execute_sql,
    add_distinct,
)
from src.improvements.distinct_llm_judge import llm_should_add_distinct, _get_client


SNAPSHOT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/sql_selection/bird/dev.snapshot.data/items.jsonl"
OUT_DIR = Path(__file__).resolve().parents[2] / "results/pattern1b_distinct_intermediate"


def _norm(rows):
    return sorted(rows, key=lambda r: str(r))


def main():
    items = [json.loads(l) for l in open(SNAPSHOT)]
    print(f"Loaded {len(items)} samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    fixed, broken, both_pass, both_fail = [], [], [], []
    per_db_baseline = defaultdict(lambda: [0, 0])
    per_db_fixed = defaultdict(lambda: [0, 0])
    rows_out = []
    llm_calls = 0
    llm_decisions = Counter()

    for it in items:
        q_id = it["input"]["question_id"]
        db_id = it["input"]["database_id"]
        diff = it["input"]["difficulty"]
        db_path = it["input"]["database_schema"]["db_path"]
        question = it["input"]["question"]
        evidence = it["input"].get("evidence", "")
        gold_sql = it["input"]["gold_sql"]
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]

        gold_rows, gold_err = execute_sql(db_path, gold_sql)
        pred_rows, pred_err = execute_sql(db_path, pred_sql)

        baseline_pass = (pred_err is None and gold_err is None and
                         _norm(pred_rows) == _norm(gold_rows))

        # Stage 1: rule-based pre-filter
        rule_decision = verify_and_fix(pred_sql, db_path, pred_rows=pred_rows)

        new_sql = pred_sql
        verifier_action = "skipped"

        if rule_decision.needs_distinct:
            # Stage 2: LLM judge
            llm_calls += 1
            try:
                llm_yes, llm_raw = llm_should_add_distinct(
                    question=question,
                    evidence=evidence,
                    sql=pred_sql,
                    pred_rows=pred_rows,
                    client=client,
                )
            except Exception as e:
                print(f"  LLM error on q_id={q_id}: {e}")
                llm_yes, llm_raw = False, f"ERROR: {e}"

            llm_decisions[llm_raw[:20]] += 1
            if llm_yes:
                new_sql = add_distinct(pred_sql)
                verifier_action = "added_distinct"
            else:
                verifier_action = "llm_rejected"

        # Re-execute to evaluate
        if new_sql != pred_sql:
            new_rows, _ = execute_sql(db_path, new_sql)
            new_pass = new_rows is not None and _norm(new_rows) == _norm(gold_rows or [])
        else:
            new_pass = baseline_pass

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
            "question_id": q_id, "db_id": db_id, "difficulty": diff,
            "baseline_match": "PASS" if baseline_pass else "FAIL",
            "pattern1b_match": "PASS" if new_pass else "FAIL",
            "rule_fired": "YES" if rule_decision.needs_distinct else "NO",
            "rule_reason": rule_decision.reason,
            "verifier_action": verifier_action,
            "baseline_sql": pred_sql,
            "pattern1b_sql": new_sql,
            "gold_sql": gold_sql,
        })

    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    n_total = len(items)
    n_baseline = len(both_pass) + len(broken)
    n_fixed = len(both_pass) + len(fixed)

    print("\n" + "=" * 70)
    print("Pattern 1B: Two-Stage (Rule + LLM Judge) DISTINCT Verifier")
    print("=" * 70)
    print(f"\nBaseline accuracy:  {n_baseline}/{n_total} = {n_baseline/n_total:.2%}")
    print(f"After Pattern 1B:   {n_fixed}/{n_total} = {n_fixed/n_total:.2%}")
    print(f"\nNet delta:          {len(fixed) - len(broken):+d} samples "
          f"({(len(fixed)-len(broken))/n_total*100:+.2f}pp)")
    print(f"  Fixed   (FAIL→PASS): {len(fixed)}")
    print(f"  Broken  (PASS→FAIL): {len(broken)}")
    print(f"\nLLM judge invocations: {llm_calls}")
    print(f"LLM response distribution:")
    for r, c in llm_decisions.most_common():
        print(f"  '{r}': {c}")

    if fixed:
        print("\nFixed cases:")
        for q_id, db, d, q in fixed:
            print(f"  [{db}/{q_id}] {d}: {q}")
    if broken:
        print("\nBroken cases:")
        for q_id, db, d, q in broken:
            print(f"  [{db}/{q_id}] {d}: {q}")

    print("\nPer-DB (baseline → pattern1B):")
    for db in sorted(per_db_baseline):
        b_ok, b_n = per_db_baseline[db]
        p_ok, _ = per_db_fixed[db]
        delta = p_ok - b_ok
        marker = "  ⭐" if delta != 0 else ""
        print(f"  {db:28s} {b_ok}/{b_n} → {p_ok}/{b_n} (Δ={delta:+d}){marker}")

    print(f"\n📄 Per-sample CSV: {csv_path}")


if __name__ == "__main__":
    main()
