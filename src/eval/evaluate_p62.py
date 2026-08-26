#!/usr/bin/env python3
"""Evaluate Pattern 6.2 (Asymmetric Veto) on full dev1534 under Set-EX.

Compares to baseline:
  - S7 tournament alone           (our reproduction's baseline)
  - + Pattern 6.0 (pure majority) (the failed naive attempt)
  - + Pattern 6.2 (asymmetric)    (this experiment)
"""
import csv
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.asymmetric_selector import select_asymmetric, execute

ROOT = Path(__file__).resolve().parents[2]
S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p62.csv"


def set_match(pred, gold):
    if gold is None or pred is None:
        return False
    return set(pred) == set(gold)


def echo(msg):
    print(msg, flush=True)


def main():
    echo("Loading datasets...")
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S7.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S6.open())}
    echo(f"  {len(s7_by_qid)} S7 items, {len(s6_by_qid)} S6 items")

    rows = []
    n_eval = base_pass = p62_pass = 0
    fix = brk = neutral_pass = neutral_fail = 0
    triggers = defaultdict(int)
    by_diff = defaultdict(lambda: [0, 0, 0])  # [base, p62, n]

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

        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        s7_rows = execute(db, s7_sql)
        s7_correct = set_match(s7_rows, gold_rows)

        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        dec = select_asymmetric(s7_sql, cands, db)
        p62_rows = execute(db, dec.selected_sql) if dec.triggered else s7_rows
        p62_correct = set_match(p62_rows, gold_rows)

        n_eval += 1
        base_pass += int(s7_correct)
        p62_pass += int(p62_correct)
        if p62_correct and not s7_correct: fix += 1
        elif not p62_correct and s7_correct: brk += 1
        elif p62_correct and s7_correct: neutral_pass += 1
        else: neutral_fail += 1

        triggers[dec.trigger_reason] += 1
        by_diff[diff][0] += int(s7_correct); by_diff[diff][1] += int(p62_correct); by_diff[diff][2] += 1

        rows.append({
            "qid": qid, "difficulty": diff,
            "s7_pass": s7_correct, "p62_pass": p62_correct,
            "triggered": dec.triggered, "reason": dec.trigger_reason,
            "rescue_group_size": dec.rescue_group_size,
            "n_safe_candidates": dec.n_safe_candidates,
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    echo("\n" + "=" * 80)
    echo(f"Pattern 6.2 (Asymmetric Veto) on full dev1534 (n={n_eval})")
    echo("=" * 80)
    echo(f"  Baseline (S7 tournament):  {base_pass}/{n_eval} = {base_pass/n_eval:.2%}")
    echo(f"  + Pattern 6.2:             {p62_pass}/{n_eval} = {p62_pass/n_eval:.2%}")
    echo(f"  Δ:                         {p62_pass - base_pass:+d} ({(p62_pass-base_pass)/n_eval*100:+.2f}pp)")
    echo(f"\n  fix (FAIL→PASS):        {fix}")
    echo(f"  break (PASS→FAIL):      {brk}")
    echo(f"  fix:break ratio:        {fix/max(brk,1):.2f}:1")
    echo(f"  neutral pass:           {neutral_pass}")
    echo(f"  neutral fail:           {neutral_fail}")

    echo("\nTrigger reason distribution:")
    for reason, c in sorted(triggers.items(), key=lambda x: -x[1]):
        echo(f"  {reason:<24}  {c:>5}  ({c/n_eval*100:.1f}%)")

    echo("\nBy difficulty:")
    echo(f"  {'diff':<14}{'n':>6}{'base':>16}{'p62':>16}{'Δ':>6}")
    for d in ["simple", "moderate", "challenging"]:
        b, p, n = by_diff[d]
        if n:
            echo(f"  {d:<14}{n:>6}  {b}/{n}={b/n:.1%}  {p}/{n}={p/n:.1%}  {p-b:+d}")

    echo(f"\n✓ Per-sample → {OUT}")


if __name__ == "__main__":
    main()
