#!/usr/bin/env python3
"""Evaluate Pattern 3 (DISTINCT Remover) on full dev1534.

Same methodology as P1 v2 evaluation:
  1. For each item, run baseline + P1v2 to get current state.
  2. Apply P3 with multiple dup_ratio thresholds.
  3. Report fires/fixes/breaks per τ.
  4. Output per-sample features for LOO-DB CV.
"""
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.distinct_verifier import verify_and_fix as p1_verify, execute_sql
from src.improvements.distinct_remover import (
    has_distinct, has_aggregate, has_groupby, remove_distinct, duplication_ratio,
)

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p3.csv"

TAUS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]


def norm(rows):
    if rows is None:
        return None
    return sorted([tuple(r) for r in rows], key=lambda r: str(r))


def main():
    items = [json.loads(l) for l in open(S)]
    print(f"Loaded {len(items)} items\n", flush=True)

    rows = []
    counters = {tau: defaultdict(int) for tau in TAUS}
    base_pass_total = 0
    p1_pass_total = 0
    eval_total = 0
    by_diff = {tau: defaultdict(lambda: [0, 0, 0]) for tau in TAUS}

    for i, it in enumerate(items, 1):
        db = it["input"]["database_schema"]["db_path"]
        gold_rows, _ = execute_sql(db, it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        gold = norm(gold_rows)
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        pred_rows, _ = execute_sql(db, pred_sql)
        base_pass = pred_rows is not None and norm(pred_rows) == gold

        p1_dec = p1_verify(pred_sql, db, pred_rows=pred_rows, dup_ratio_threshold=0.80)
        if p1_dec.needs_distinct:
            current_sql = p1_dec.new_sql
            current_rows = p1_dec.new_rows
            current_pass = norm(current_rows) == gold
        else:
            current_sql = pred_sql
            current_rows = pred_rows
            current_pass = base_pass

        eval_total += 1
        base_pass_total += int(base_pass)
        p1_pass_total += int(current_pass)
        diff = it["input"].get("difficulty", "?")

        p3_eligible = (has_distinct(current_sql) and
                       not has_aggregate(current_sql) and
                       not has_groupby(current_sql))
        dup_r_without_distinct = None
        new_rows_no_distinct = None
        if p3_eligible:
            new_sql = remove_distinct(current_sql)
            if new_sql is not None:
                new_rows_no_distinct, _ = execute_sql(db, new_sql)
                if new_rows_no_distinct is not None and len(new_rows_no_distinct) > len(current_rows or []):
                    dup_r_without_distinct = duplication_ratio(new_rows_no_distinct)

        new_pass_normed = (norm(new_rows_no_distinct) == gold) if new_rows_no_distinct is not None else None

        for tau in TAUS:
            fires = (dup_r_without_distinct is not None and dup_r_without_distinct <= tau)
            if fires:
                counters[tau]["fire"] += 1
                p3_pass = new_pass_normed
                if p3_pass and not current_pass:
                    counters[tau]["fix"] += 1
                elif not p3_pass and current_pass:
                    counters[tau]["break"] += 1
                elif p3_pass and current_pass:
                    counters[tau]["neutral_pass"] += 1
                else:
                    counters[tau]["neutral_fail"] += 1
            else:
                p3_pass = current_pass
            by_diff[tau][diff][0] += int(current_pass)
            by_diff[tau][diff][1] += int(p3_pass)
            by_diff[tau][diff][2] += 1

        rows.append({
            "qid": it["input"]["question_id"],
            "db_id": it["input"]["database_id"],
            "difficulty": diff,
            "base_pass": base_pass,
            "p1_pass": current_pass,
            "p3_eligible": p3_eligible,
            "dup_ratio_no_distinct": dup_r_without_distinct,
            "fires_at_tau0.50": (dup_r_without_distinct is not None and dup_r_without_distinct <= 0.50),
        })

        if i % 200 == 0:
            print(f"  ...{i}/{len(items)} processed", flush=True)

    print(f"\nBaseline: {base_pass_total}/{eval_total} = {base_pass_total/eval_total:.2%}")
    print(f"P1 v2:    {p1_pass_total}/{eval_total} = {p1_pass_total/eval_total:.2%} (+{p1_pass_total - base_pass_total})\n")

    print("=" * 88)
    print(f"P3 (DISTINCT Remover) sweep")
    print("=" * 88)
    print(f"  {'τ':>6}  {'fired':>7}  {'fixes':>6}  {'breaks':>7}  {'net':>5}  "
          f"{'p1+p3_pass':>11}  {'accuracy':>10}")
    best_tau, best_net = None, -1e9
    for tau in TAUS:
        c = counters[tau]
        net = c["fix"] - c["break"]
        total_pass = p1_pass_total + net
        acc = total_pass / eval_total
        marker = ""
        if net > best_net:
            best_tau, best_net = tau, net
            marker = "  ← best"
        print(f"  {tau:>6.2f}  {c['fire']:>7}  {c['fix']:>6}  {c['break']:>7}  "
              f"{net:>+5}  {total_pass:>11}  {acc:>9.2%}{marker}")

    print(f"\nBest τ={best_tau}, net=+{best_net}")
    print(f"Final P1+P3 accuracy at best τ: "
          f"{p1_pass_total + best_net}/{eval_total} = {(p1_pass_total + best_net)/eval_total:.2%}")
    print(f"Total gain over baseline: +{p1_pass_total + best_net - base_pass_total} "
          f"(+{(p1_pass_total + best_net - base_pass_total)/eval_total*100:.2f}pp)")

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Wrote {OUT}")


if __name__ == "__main__":
    main()
