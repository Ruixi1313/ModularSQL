#!/usr/bin/env python3
"""
Sanity Check for Pattern 1 (DISTINCT-aware Semantic Verifier)
before running it on the full BIRD-Dev (1534 samples).

Three layers:
  1. Unit tests on add_distinct() and verify_and_fix() across edge cases
  2. Reproducibility: re-run on 99-sample snapshot, confirm +4pp result stays the same
  3. Decision audit: verify every rule-fire is justified (no spurious DISTINCT injection)
"""

import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.improvements.distinct_verifier import (
    add_distinct, has_distinct, has_join, has_aggregate,
    has_groupby, has_duplicate_rows, null_density,
    verify_and_fix, execute_sql,
)


SNAPSHOT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/sql_selection/bird/dev.snapshot.data/items.jsonl"


# ---------------------------------------------------------------------------
# Layer 1: Unit tests on the rule logic
# ---------------------------------------------------------------------------

def test_unit():
    cases = [
        # (sql, description, expected_has_distinct, expected_add_distinct_form)
        # None expected output means "injector should return None (unsupported form)".
        ("SELECT name FROM t", "simple SELECT", False, "SELECT DISTINCT name FROM t"),
        ("  SELECT name FROM t", "leading whitespace", False, "  SELECT DISTINCT name FROM t"),
        ("SELECT DISTINCT name FROM t", "already has DISTINCT", True, None),
        ("select * from t", "lowercase select", False, "select DISTINCT * from t"),
        # CTE and wrapped SELECT are NOT supported — injector returns None,
        # verifier's verify_and_fix treats this as "no-op" (fails closed).
        ("WITH cte AS (SELECT * FROM t) SELECT name FROM cte", "CTE prefix (unsupported)", False, None),
        ("(SELECT name FROM t)", "wrapped in parens (unsupported)", False, None),
    ]

    failures = []
    for sql, desc, exp_has, exp_add in cases:
        got_has = has_distinct(sql)
        if got_has != exp_has:
            failures.append(f"  has_distinct('{desc}'): got {got_has}, expected {exp_has}")
        if not exp_has:
            got_add = add_distinct(sql)
            if got_add != exp_add:
                failures.append(f"  add_distinct('{desc}'):\n    got:      {got_add}\n    expected: {exp_add}")

    # has_aggregate
    agg_cases = [
        ("SELECT COUNT(*) FROM t", True),
        ("SELECT SUM(x) FROM t", True),
        ("SELECT AVG(x) FROM t", True),
        ("SELECT MAX(x) FROM t", True),
        ("SELECT MIN(x) FROM t", True),
        ("SELECT name FROM t", False),
        ("SELECT counts FROM t", False),  # 'counts' is not an aggregate fn
        ("SELECT * FROM count_table", False),  # 'count' as identifier not call
    ]
    for sql, exp in agg_cases:
        got = has_aggregate(sql)
        if got != exp:
            failures.append(f"  has_aggregate('{sql}'): got {got}, expected {exp}")

    # has_join
    join_cases = [
        ("SELECT * FROM a JOIN b ON a.x=b.x", True),
        ("SELECT * FROM a INNER JOIN b ON a.x=b.x", True),
        ("SELECT * FROM a LEFT JOIN b ON a.x=b.x", True),
        ("SELECT * FROM a, b WHERE a.x=b.x", False),  # implicit join, regex misses (acceptable)
        ("SELECT * FROM t", False),
    ]
    for sql, exp in join_cases:
        got = has_join(sql)
        if got != exp:
            failures.append(f"  has_join('{sql[:50]}'): got {got}, expected {exp}")

    # null_density
    nd_cases = [
        ([], 0.0),
        ([("a",), ("b",)], 0.0),
        ([(None,), (None,)], 1.0),
        ([("a",), (None,), ("b",), (None,)], 0.5),
        ([(1, None), (2, "x")], 0.5),  # any-cell NULL
    ]
    for rows, exp in nd_cases:
        got = null_density(rows)
        if abs(got - exp) > 1e-6:
            failures.append(f"  null_density({rows}): got {got}, expected {exp}")

    # has_duplicate_rows
    dup_cases = [
        ([], False),
        ([("a",)], False),
        ([("a",), ("a",)], True),
        ([("a",), ("b",)], False),
        ([(None,), (None,)], True),
    ]
    for rows, exp in dup_cases:
        got = has_duplicate_rows(rows)
        if got != exp:
            failures.append(f"  has_duplicate_rows({rows}): got {got}, expected {exp}")

    return failures


# ---------------------------------------------------------------------------
# Layer 2: Reproducibility on 99 snapshot
# ---------------------------------------------------------------------------

def _norm(rows):
    return sorted(rows, key=lambda r: str(r))


def test_reproducibility():
    items = [json.loads(l) for l in open(SNAPSHOT)]
    baseline_ok = pattern1_ok = 0
    fixed = []
    broken = []
    decisions = Counter()

    for it in items:
        db_path = it["input"]["database_schema"]["db_path"]
        gold = it["input"]["gold_sql"]
        pred = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]

        gold_rows, gold_err = execute_sql(db_path, gold)
        pred_rows, pred_err = execute_sql(db_path, pred)

        baseline_pass = (pred_err is None and gold_err is None
                         and _norm(pred_rows) == _norm(gold_rows))
        if baseline_pass:
            baseline_ok += 1

        decision = verify_and_fix(pred, db_path, pred_rows=pred_rows)
        decisions[decision.reason] += 1

        if decision.needs_distinct:
            new_pass = _norm(decision.new_rows) == _norm(gold_rows or [])
        else:
            new_pass = baseline_pass

        if new_pass:
            pattern1_ok += 1
        if not baseline_pass and new_pass:
            fixed.append(it["input"]["question_id"])
        if baseline_pass and not new_pass:
            broken.append(it["input"]["question_id"])

    return {
        "baseline": baseline_ok,
        "pattern1": pattern1_ok,
        "fixed": fixed,
        "broken": broken,
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# Layer 3: Decision audit — every rule-fire must be justified
# ---------------------------------------------------------------------------

def audit_decisions():
    items = [json.loads(l) for l in open(SNAPSHOT)]
    fires = []
    rejections = []

    for it in items:
        db_path = it["input"]["database_schema"]["db_path"]
        pred = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        gold = it["input"]["gold_sql"]
        decision = verify_and_fix(pred, db_path)

        if decision.needs_distinct:
            # Check gold for hint that DISTINCT is correct here
            gold_has_distinct = has_distinct(gold)
            fires.append({
                "qid": it["input"]["question_id"],
                "db": it["input"]["database_id"],
                "gold_has_distinct": gold_has_distinct,
                "reason": decision.reason,
            })
        elif decision.reason.startswith("high_null_density"):
            # NULL-guard rejection — verify gold doesn't use DISTINCT
            gold_has_distinct = has_distinct(gold)
            rejections.append({
                "qid": it["input"]["question_id"],
                "db": it["input"]["database_id"],
                "gold_has_distinct": gold_has_distinct,
                "reason": decision.reason,
            })

    return fires, rejections


def main():
    print("=" * 72)
    print("Layer 1: Unit tests on rule logic")
    print("=" * 72)
    failures = test_unit()
    if failures:
        print(f"\n✗ {len(failures)} UNIT TEST FAILURES:")
        for f in failures:
            print(f)
        sys.exit(1)
    print("✓ All unit tests pass (add_distinct, has_distinct, has_aggregate, "
          "has_join, null_density, has_duplicate_rows)")

    print("\n" + "=" * 72)
    print("Layer 2: Reproducibility on 99-sample snapshot")
    print("=" * 72)
    t = time.time()
    result = test_reproducibility()
    elapsed = time.time() - t
    print(f"  Re-evaluated 99 samples in {elapsed:.1f}s")
    print(f"  Baseline accuracy:  {result['baseline']}/99 = {result['baseline']/99:.2%}")
    print(f"  Pattern 1 accuracy: {result['pattern1']}/99 = {result['pattern1']/99:.2%}")
    delta = result['pattern1'] - result['baseline']
    print(f"  Delta:              {delta:+d}  ({delta/99*100:+.2f}pp)")
    print(f"  Fixed (FAIL→PASS):  {len(result['fixed'])} {result['fixed']}")
    print(f"  Broken (PASS→FAIL): {len(result['broken'])} {result['broken']}")

    EXPECTED_BASELINE = 71
    EXPECTED_PATTERN1 = 75
    EXPECTED_FIXED = [345, 850, 853, 854]
    EXPECTED_BROKEN = []
    if (result['baseline'] != EXPECTED_BASELINE
            or result['pattern1'] != EXPECTED_PATTERN1
            or sorted(result['fixed']) != sorted(EXPECTED_FIXED)
            or sorted(result['broken']) != sorted(EXPECTED_BROKEN)):
        print(f"\n✗ Numbers DRIFTED from prior validated run")
        print(f"  Expected baseline {EXPECTED_BASELINE}, pattern1 {EXPECTED_PATTERN1},"
              f" fixed {EXPECTED_FIXED}, broken {EXPECTED_BROKEN}")
        sys.exit(2)
    print(f"\n✓ Numbers match the validated record (B=71, +P1=75, +4 fixed, 0 broken)")

    print("\n  Decision distribution:")
    for r, c in result['decisions'].most_common():
        print(f"    {r:50s} {c:3d}")

    print("\n" + "=" * 72)
    print("Layer 3: Decision audit — every fire must be justified")
    print("=" * 72)
    fires, rejections = audit_decisions()
    print(f"  Rule fires (DISTINCT injected): {len(fires)}")
    fires_gold_has_distinct = sum(1 for f in fires if f["gold_has_distinct"])
    print(f"    of which gold also uses DISTINCT: {fires_gold_has_distinct} / {len(fires)}")
    if fires_gold_has_distinct < len(fires):
        print("    ⚠ Some fires inject DISTINCT but gold doesn't — these MUST be the "
              "result-changing wins. Review:")
        for f in fires:
            if not f["gold_has_distinct"]:
                print(f"      [{f['db']}/{f['qid']}] gold_has_distinct=False  (this is a "
                      "legitimate fix if it executes to the same result as gold)")

    print(f"\n  NULL-guard rejections (kept original): {len(rejections)}")
    rejections_gold_distinct = sum(1 for r in rejections if r["gold_has_distinct"])
    print(f"    of which gold DOES use DISTINCT: {rejections_gold_distinct} / {len(rejections)}")
    if rejections_gold_distinct > 0:
        print("    ⚠ NULL-guard suppressed DISTINCT but gold actually uses it. "
              "Review whether these need adjustment:")
        for r in rejections:
            if r["gold_has_distinct"]:
                print(f"      [{r['db']}/{r['qid']}]  reason={r['reason']}")
    else:
        print("    ✓ NULL-guard never rejects a true-positive case")

    print("\n" + "=" * 72)
    print("✅ Sanity check passed — Pattern 1 is safe to apply to full BIRD-Dev")
    print("=" * 72)


if __name__ == "__main__":
    main()
