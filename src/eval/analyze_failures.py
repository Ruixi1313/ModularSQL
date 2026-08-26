#!/usr/bin/env python3
"""
Analyze DeepEye-SQL failure cases on the 99-sample baseline.

Compares predicted SQL execution results vs gold, categorizes failures by:
  - Result mismatch type (VALUE / ROW_COUNT / COLUMN_COUNT / EMPTY / EXEC_ERROR)
  - Per-database breakdown
  - Surface SQL pattern hints (missing WHERE, wrong aggregation, etc.)

Focus: 3 worst-performing DBs (formula_1, card_games, european_football_2).
"""

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


SNAPSHOT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/sql_selection/bird/dev.snapshot.data/items.jsonl"
FOCUS_DBS = {"formula_1", "card_games", "european_football_2"}


def execute(conn, sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return sorted([tuple(r) for r in cur.fetchall()]), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def classify_mismatch(pred_rows, gold_rows, pred_err, gold_err):
    if pred_err:
        return "PRED_EXEC_ERROR"
    if pred_rows == gold_rows:
        return "MATCH"
    if not pred_rows and gold_rows:
        return "PRED_EMPTY_GT_NONEMPTY"
    if not gold_rows and pred_rows:
        return "PRED_NONEMPTY_GT_EMPTY"
    # Compare structure
    if pred_rows and gold_rows:
        p_cols = len(pred_rows[0]) if isinstance(pred_rows[0], tuple) else 1
        g_cols = len(gold_rows[0]) if isinstance(gold_rows[0], tuple) else 1
        if p_cols != g_cols:
            return "COLUMN_COUNT_MISMATCH"
        if len(pred_rows) != len(gold_rows):
            return "ROW_COUNT_MISMATCH"
        return "VALUE_MISMATCH"
    return "UNKNOWN"


def detect_sql_patterns(pred_sql, gold_sql):
    """Surface hints about what's different."""
    hints = []
    p = pred_sql.upper()
    g = gold_sql.upper()

    # Gold has WHERE that pred doesn't (missing filter)
    g_wheres = re.findall(r"WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|HAVING|UNION|$)", g, re.DOTALL)
    p_wheres = re.findall(r"WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|HAVING|UNION|$)", p, re.DOTALL)
    if g_wheres and not p_wheres:
        hints.append("MISSING_WHERE")

    # Aggregation pattern differences
    if ("SUM(CASE" in g or "SUM(IIF" in g) and "SUM(CASE" not in p and "SUM(IIF" not in p:
        hints.append("AGG_STYLE_DIFF (gold uses SUM(CASE/IIF), pred doesn't)")
    if "COUNT(DISTINCT" in g and "COUNT(DISTINCT" not in p:
        hints.append("MISSING_DISTINCT")

    # Subquery / nesting
    g_subq = g.count("(SELECT")
    p_subq = p.count("(SELECT")
    if abs(g_subq - p_subq) >= 1:
        hints.append(f"SUBQUERY_DIFF (gold has {g_subq}, pred has {p_subq})")

    # ORDER + LIMIT
    if "ORDER BY" in g and "ORDER BY" not in p:
        hints.append("MISSING_ORDER_BY")
    if "LIMIT" in g and "LIMIT" not in p:
        hints.append("MISSING_LIMIT")

    # CAST as REAL (for percentage)
    if "CAST(" in g and "CAST(" not in p:
        hints.append("MISSING_CAST (likely percentage)")

    return hints


def main():
    if not SNAPSHOT.exists():
        print(f"Snapshot not found: {SNAPSHOT}")
        sys.exit(1)

    items = [json.loads(l) for l in open(SNAPSHOT)]
    print(f"Loaded {len(items)} samples\n")

    by_db = defaultdict(list)
    for it in items:
        by_db[it["input"]["database_id"]].append(it)

    overall_cats = Counter()
    overall_hints = Counter()
    per_db_results = {}

    failure_details_focus = []  # for the 3 focus DBs

    for db_id, db_items in sorted(by_db.items()):
        db_path = db_items[0]["input"]["database_schema"]["db_path"]
        conn = sqlite3.connect(db_path)
        cats = Counter()
        hints_in_db = Counter()
        for it in db_items:
            gold = it["input"]["gold_sql"]
            pred = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
            gold_rows, gold_err = execute(conn, gold)
            pred_rows, pred_err = execute(conn, pred)
            cat = classify_mismatch(pred_rows, gold_rows, pred_err, gold_err)
            cats[cat] += 1
            if cat != "MATCH":
                hints = detect_sql_patterns(pred, gold)
                for h in hints:
                    hints_in_db[h] += 1
                    overall_hints[h] += 1
                if db_id in FOCUS_DBS:
                    failure_details_focus.append({
                        "db": db_id,
                        "q_id": it["input"]["question_id"],
                        "difficulty": it["input"]["difficulty"],
                        "category": cat,
                        "hints": hints,
                        "question": it["input"]["question"][:120],
                        "evidence": it["input"].get("evidence", "")[:80],
                        "pred_sql": pred[:200],
                        "gold_sql": gold[:200],
                    })
        overall_cats += cats
        per_db_results[db_id] = (cats, hints_in_db)

    # Print overview
    print("=" * 70)
    print("Failure category breakdown (overall)")
    print("=" * 70)
    total = sum(overall_cats.values())
    for cat, n in overall_cats.most_common():
        print(f"  {cat:30s} {n:3d} ({n/total:.1%})")

    print("\n" + "=" * 70)
    print("SQL pattern hints (across all failures)")
    print("=" * 70)
    for hint, n in overall_hints.most_common():
        print(f"  {hint:50s} {n:3d}")

    print("\n" + "=" * 70)
    print("Per-DB failure mode")
    print("=" * 70)
    for db_id, (cats, hints) in sorted(per_db_results.items()):
        match = cats.get("MATCH", 0)
        total = sum(cats.values())
        marker = " ⭐" if db_id in FOCUS_DBS else ""
        print(f"\n{db_id} ({match}/{total} = {match/total:.0%}){marker}")
        for cat, n in cats.most_common():
            if cat != "MATCH":
                print(f"    {cat:30s} {n}")
        if hints:
            print(f"  Top hints:")
            for h, n in hints.most_common(3):
                print(f"    - {h}: {n}")

    # Detailed failures for the 3 focus DBs
    print("\n\n" + "=" * 70)
    print(f"FOCUS DB FAILURE DETAILS ({', '.join(FOCUS_DBS)})")
    print("=" * 70)
    for f in failure_details_focus:
        print(f"\n[{f['db']}/{f['q_id']}] {f['difficulty']} | {f['category']}")
        print(f"  Hints: {f['hints'] or '(none)'}")
        print(f"  Q: {f['question']}")
        if f["evidence"]:
            print(f"  E: {f['evidence']}")
        print(f"  PRED: {f['pred_sql']}")
        print(f"  GOLD: {f['gold_sql']}")


if __name__ == "__main__":
    main()
