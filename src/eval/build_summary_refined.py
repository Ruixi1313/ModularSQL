#!/usr/bin/env python3
"""Re-evaluate refined Pattern 1 on full dev1534 across multiple τ values.

For each item, executes baseline once, computes dup_ratio, and decides for each
candidate τ whether to fire. Outputs:
  - per-sample summary CSV with p1_match for each τ
  - headline comparison table
"""
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.distinct_verifier import (
    has_distinct, has_aggregate, has_groupby, has_join,
    has_duplicate_rows, duplication_ratio, add_distinct, execute_sql,
)

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_refined.csv"

TAUS = [0.00, 0.50, 0.65, 0.70, 0.75, 0.80]


def norm(rows):
    if rows is None:
        return None
    return sorted([tuple(r) for r in rows], key=lambda r: str(r))


def decide_fires(pred_sql, pred_rows):
    """Return (fires_at_tau dict, dup_ratio). Pre-conditions checked once."""
    fires = {tau: False for tau in TAUS}
    if has_distinct(pred_sql) or has_aggregate(pred_sql) or has_groupby(pred_sql):
        return fires, 0.0
    if not has_join(pred_sql):
        return fires, 0.0
    if pred_rows is None or not has_duplicate_rows(pred_rows):
        return fires, 0.0
    dup_r = duplication_ratio(pred_rows)
    for tau in TAUS:
        if dup_r >= tau:
            fires[tau] = True
    return fires, dup_r


def main():
    items = [json.loads(l) for l in open(S)]
    print(f"Loaded {len(items)} items\n", flush=True)

    rows = []
    # counters[tau] = {"fire": n, "fix": n, "break": n, "neutral_pass": n, "neutral_fail": n}
    counters = {tau: defaultdict(int) for tau in TAUS}
    base_pass_total = 0
    eval_total = 0
    by_diff = {tau: defaultdict(lambda: [0, 0, 0]) for tau in TAUS}  # [base_pass, p1_pass, n]

    for i, it in enumerate(items, 1):
        db = it["input"]["database_schema"]["db_path"]
        gold_rows, _ = execute_sql(db, it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        gold = norm(gold_rows)
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        pred_rows, _ = execute_sql(db, pred_sql)
        base_pass = pred_rows is not None and norm(pred_rows) == gold

        eval_total += 1
        base_pass_total += int(base_pass)
        diff = it["input"].get("difficulty", "?")

        fires, dup_r = decide_fires(pred_sql, pred_rows)
        # Only compute DISTINCT once if any τ fires
        p1_pass_at = {}
        new_rows_normed = None
        if any(fires.values()):
            new_sql = add_distinct(pred_sql)
            if new_sql is not None:
                new_r, _ = execute_sql(db, new_sql)
                new_rows_normed = norm(new_r) if new_r is not None else None

        for tau in TAUS:
            if fires[tau] and new_rows_normed is not None:
                p1_pass = new_rows_normed == gold
                counters[tau]["fire"] += 1
                if p1_pass and not base_pass:
                    counters[tau]["fix"] += 1
                elif not p1_pass and base_pass:
                    counters[tau]["break"] += 1
                elif p1_pass and base_pass:
                    counters[tau]["neutral_pass"] += 1
                else:
                    counters[tau]["neutral_fail"] += 1
            else:
                p1_pass = base_pass
            p1_pass_at[tau] = p1_pass
            by_diff[tau][diff][0] += int(base_pass)
            by_diff[tau][diff][1] += int(p1_pass)
            by_diff[tau][diff][2] += 1

        row = {
            "qid": it["input"]["question_id"],
            "db_id": it["input"]["database_id"],
            "difficulty": diff,
            "baseline_match": "PASS" if base_pass else "FAIL",
            "dup_ratio": round(dup_r, 3),
        }
        for tau in TAUS:
            row[f"p1_match_tau{tau:.2f}"] = "PASS" if p1_pass_at[tau] else "FAIL"
            row[f"fired_tau{tau:.2f}"] = "YES" if fires[tau] else "NO"
        rows.append(row)

        if i % 200 == 0:
            print(f"  ...{i}/{len(items)} processed", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Wrote {OUT} ({len(rows)} rows)\n", flush=True)

    print("=" * 84)
    print(f"Refined Pattern 1 on full dev1534 (baseline = {base_pass_total}/{eval_total} = {base_pass_total/eval_total:.2%})")
    print("=" * 84)
    print(f"  {'τ':>6}  {'fired':>7}  {'fixes':>6}  {'breaks':>7}  {'net':>5}  "
          f"{'total_pass':>11}  {'accuracy':>10}")
    for tau in TAUS:
        c = counters[tau]
        net = c["fix"] - c["break"]
        total_pass = base_pass_total + net
        acc = total_pass / eval_total
        marker = "  ← current" if tau == 0.00 else ("  ← deployed" if tau == 0.70 else "")
        print(f"  {tau:>6.2f}  {c['fire']:>7}  {c['fix']:>6}  {c['break']:>7}  "
              f"{net:>+5}  {total_pass:>11}  {acc:>9.2%}{marker}")
    print()

    print("By difficulty (deployed τ=0.70 vs baseline):")
    tau = 0.70
    for d in ["simple", "moderate", "challenging"]:
        b, p, n = by_diff[tau][d]
        if n:
            print(f"  {d:<14}  base {b:>4}/{n}={b/n:.1%}   τ=0.70 {p:>4}/{n}={p/n:.1%}   Δ={p-b:+d}")


if __name__ == "__main__":
    main()
