#!/usr/bin/env python3
"""Probe: try injecting CAST AS REAL around divisions in pred SQL for all
`missing_cast` failures. Measure the actual fix rate — this is the true
upper bound for a P3 = "CAST injector" plugin.

Patch strategy 1 (simple): wrap the LHS of every '/' operator with CAST(... AS REAL)
Patch strategy 2 (safer): only wrap when LHS is a column reference or SUM/COUNT
"""
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_refined.csv"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"
CLUSTERS = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/failure_clusters.csv"

CAST_RE = re.compile(r"\bCAST\s*\(", re.IGNORECASE)


def norm(rows):
    if rows is None:
        return None
    return sorted([tuple(r) for r in rows], key=lambda r: str(r))


def execute(db, sql, timeout=8):
    try:
        c = sqlite3.connect(db, timeout=timeout)
        cur = c.cursor()
        cur.execute(sql)
        r = [tuple(x) for x in cur.fetchall()]
        c.close()
        return r
    except Exception as e:
        return None


def patch_cast_simple(sql):
    """Wrap LHS of every '/' with CAST(... AS REAL) if not already cast.
    LHS is taken as the bracketed/quoted/identifier expression immediately
    preceding ' / '. Returns None if no change made."""
    if CAST_RE.search(sql):
        return None  # already has CAST somewhere — skip naive injection

    # Match: (column_or_expr) / (column_or_expr)
    # Heuristic: find ' / ' and look backward for a balanced expr
    out = []
    i = 0
    changed = False
    while i < len(sql):
        # Find next ' / '
        j = sql.find('/', i)
        if j == -1:
            out.append(sql[i:])
            break

        # Skip if it's part of /* comment or // — none typical in SQL
        # Walk back to find LHS expression boundary
        k = j - 1
        while k >= 0 and sql[k] == ' ':
            k -= 1
        if k < 0:
            out.append(sql[i:j+1])
            i = j + 1
            continue

        # Determine LHS: walk back through identifiers, dots, brackets, quotes, parens
        end = k + 1  # exclusive
        depth = 0
        if sql[k] == ')':
            depth = 1
            k -= 1
            while k >= 0 and depth > 0:
                if sql[k] == ')':
                    depth += 1
                elif sql[k] == '(':
                    depth -= 1
                k -= 1
            start = k + 1
        elif sql[k] in '`"':
            quote = sql[k]
            k -= 1
            while k >= 0 and sql[k] != quote:
                k -= 1
            start = k
        else:
            while k >= 0 and (sql[k].isalnum() or sql[k] in "_.`\""):
                if sql[k] in '`"':
                    quote = sql[k]
                    k -= 1
                    while k >= 0 and sql[k] != quote:
                        k -= 1
                k -= 1
            start = k + 1

        if start >= end:
            out.append(sql[i:j+1])
            i = j + 1
            continue

        lhs = sql[start:end]
        # Skip wrapping numeric literals (e.g., "100.0 / X")
        if re.match(r"^[\d.]+$", lhs.strip()):
            out.append(sql[i:j+1])
            i = j + 1
            continue

        # Emit: prefix + CAST(lhs AS REAL) + " / "
        out.append(sql[i:start])
        out.append(f"CAST({lhs} AS REAL)")
        out.append(sql[end:j+1])
        changed = True
        i = j + 1
    return "".join(out) if changed else None


def main():
    # Find missing_cast failures
    target_qids = set()
    with CLUSTERS.open() as f:
        for r in csv.DictReader(f):
            if "missing_cast" in r["labels"].split(";"):
                target_qids.add(int(r["qid"]))
    print(f"Found {len(target_qids)} missing_cast failures\n")

    items = {}
    with S7.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in target_qids:
                items[qid] = it

    fixed = []
    patched_no_match = []
    no_patch = []
    patched_error = []

    for qid in sorted(target_qids):
        it = items[qid]
        db = it["input"]["database_schema"]["db_path"]
        gold = norm(execute(db, it["input"]["gold_sql"]))
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        patched = patch_cast_simple(pred_sql)
        if patched is None:
            no_patch.append(qid)
            continue
        new_rows = execute(db, patched)
        if new_rows is None:
            patched_error.append(qid)
            continue
        if norm(new_rows) == gold:
            fixed.append((qid, pred_sql, patched))
        else:
            patched_no_match.append(qid)

    print(f"Total missing_cast failures: {len(target_qids)}")
    print(f"  Patched & matched gold:    {len(fixed):>3}  ← actual P3 yield")
    print(f"  Patched but still wrong:   {len(patched_no_match):>3}")
    print(f"  Patch caused SQL error:    {len(patched_error):>3}")
    print(f"  No patch applied (already had CAST or no /): {len(no_patch):>3}")
    print()
    print(f"Estimated P3 max yield: +{len(fixed)} samples on dev1534 ({len(fixed)/1534*100:.2f}pp)")
    print()
    print(f"Fixed qids: {[q for q, _, _ in fixed]}")
    if fixed:
        q, orig, patched = fixed[0]
        print(f"\nExample fix (qid={q}):")
        print(f"  ORIG:    {orig[:200]}")
        print(f"  PATCHED: {patched[:200]}")


if __name__ == "__main__":
    main()
