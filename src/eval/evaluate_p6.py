#!/usr/bin/env python3
"""Evaluate Pattern 6 (Better Selector via execution-based majority vote)
on full dev1534. Compare to current S7 baseline under set-EX.

Outputs:
  - total fix/break/net counts
  - by-difficulty breakdown
  - by-db breakdown
  - per-sample CSV with current_pass / p6_pass / selected_idx
"""
import csv
import json
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.better_selector import select_by_majority, execute

ROOT = Path(__file__).resolve().parents[2]
# S6 (sql_revision) has BOTH raw S5 candidates AND revised candidates available.
# S7 baseline picks from REVISED candidates, so for fair comparison we MUST
# vote over revised candidates too (not raw S5 candidates).
S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p6_revised.csv"


def set_match(pred, gold):
    if gold is None or pred is None:
        return False
    return set(pred) == set(gold)


def echo(msg):
    print(msg, flush=True)


def main():
    # Load S7 baseline (current pred SQL per qid) and S6 (REVISED candidates)
    echo("Loading datasets...")
    s7_by_qid = {}
    with S7.open() as f:
        for line in f:
            it = json.loads(line)
            s7_by_qid[it["input"]["question_id"]] = it
    s6_by_qid = {}
    with S6.open() as f:
        for line in f:
            it = json.loads(line)
            s6_by_qid[it["input"]["question_id"]] = it
    echo(f"  {len(s7_by_qid)} S7 items, {len(s6_by_qid)} S6 items (revised candidates)")

    rows = []
    n_eval = 0
    base_pass = p6_pass = 0
    fix = brk = neutral_pass = neutral_fail = 0
    by_diff = defaultdict(lambda: [0, 0, 0])  # [base_pass, p6_pass, n]
    by_db = defaultdict(lambda: [0, 0, 0])
    fallback_counts = defaultdict(int)

    for i, qid in enumerate(sorted(s7_by_qid.keys()), 1):
        echo(f"  [{i}/{len(s7_by_qid)}] qid={qid}")
        s7_it = s7_by_qid[qid]
        s6_it = s6_by_qid.get(qid)
        if s6_it is None:
            continue
        db = s7_it["input"]["database_schema"]["db_path"]
        gold_rows = execute(db, s7_it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        diff = s7_it["input"].get("difficulty", "?")
        db_id = s7_it["input"]["database_id"]

        # Current S7 result
        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        s7_rows = execute(db, s7_sql)
        s7_correct = set_match(s7_rows, gold_rows)

        # Pattern 6: pick by majority vote over S6 REVISED candidates (fair)
        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        if not cands:
            p6_sql = s7_sql
            p6_rows = s7_rows
            p6_correct = s7_correct
            fb = "no_candidates"
            selected_idx = -1
            group_size = 0
            n_groups = 0
        else:
            dec = select_by_majority(cands, db)
            p6_sql = dec.selected_sql
            p6_rows = execute(db, p6_sql)
            p6_correct = set_match(p6_rows, gold_rows)
            fb = dec.fallback
            selected_idx = dec.selected_idx
            group_size = dec.group_size
            n_groups = dec.n_distinct_groups

        n_eval += 1
        base_pass += int(s7_correct)
        p6_pass += int(p6_correct)
        if p6_correct and not s7_correct: fix += 1
        elif not p6_correct and s7_correct: brk += 1
        elif p6_correct and s7_correct: neutral_pass += 1
        else: neutral_fail += 1

        by_diff[diff][0] += int(s7_correct); by_diff[diff][1] += int(p6_correct); by_diff[diff][2] += 1
        by_db[db_id][0] += int(s7_correct); by_db[db_id][1] += int(p6_correct); by_db[db_id][2] += 1
        fallback_counts[fb] += 1

        rows.append({
            "qid": qid, "db_id": db_id, "difficulty": diff,
            "s7_pass": s7_correct, "p6_pass": p6_correct,
            "p6_selected_idx": selected_idx,
            "p6_group_size": group_size, "p6_n_groups": n_groups,
            "p6_fallback": fb,
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    echo("\n" + "=" * 80)
    echo(f"Pattern 6 (Majority Vote Selector) on full dev1534 (n={n_eval})")
    echo("=" * 80)
    echo(f"  Baseline (S7 tournament):  {base_pass}/{n_eval} = {base_pass/n_eval:.2%}")
    echo(f"  + Pattern 6:               {p6_pass}/{n_eval} = {p6_pass/n_eval:.2%}")
    echo(f"  Δ:                         {p6_pass - base_pass:+d} ({(p6_pass-base_pass)/n_eval*100:+.2f}pp)")
    echo(f"\n  fix (FAIL→PASS):        {fix}")
    echo(f"  break (PASS→FAIL):      {brk}")
    echo(f"  neutral pass:           {neutral_pass}")
    echo(f"  neutral fail:           {neutral_fail}")
    echo(f"  fix:break ratio:        {fix/max(brk,1):.2f}:1")

    echo("\nFallback distribution:")
    for fb, c in sorted(fallback_counts.items(), key=lambda x: -x[1]):
        echo(f"  {fb:<20}  {c:>5}")

    echo("\nBy difficulty:")
    echo(f"  {'diff':<14}{'n':>6}{'base':>14}{'p6':>14}{'Δ':>6}")
    for d in ["simple", "moderate", "challenging"]:
        b, p, n = by_diff[d]
        if n:
            echo(f"  {d:<14}{n:>6}  {b}/{n}={b/n:.1%}  {p}/{n}={p/n:.1%}  {p-b:+d}")

    echo("\nBy database (sorted by Δ desc):")
    db_sorted = sorted(by_db.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))
    echo(f"  {'db':<26}{'base':>10}{'p6':>10}{'Δ':>6}")
    for db_id, (b, p, n) in db_sorted:
        echo(f"  {db_id:<26}{b}/{n}={b/n*100:.0f}%  {p}/{n}={p/n*100:.0f}%  {p-b:+d}")

    echo(f"\n✓ Per-sample → {OUT}")


if __name__ == "__main__":
    main()
