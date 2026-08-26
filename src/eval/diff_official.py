#!/usr/bin/env python3
"""Official BIRD-style evaluation using set() comparison (matches DeepEye's
runner/evaluation.py and BIRD official eval script).

Re-runs the diff for THREE configurations:
  1. DeepEye public artifact (from results/bird-dev/qwen3-coder-30b-a3b.json)
  2. Our baseline (workspace_full/sql_selection.topk3_backup, original run)
  3. Our + P1 v2 + P3 (apply plugins on top of baseline)

All evaluated with set() comparison.
"""
import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.distinct_verifier import verify_and_fix as p1_verify
from src.improvements.distinct_remover import (
    has_distinct, has_aggregate, has_groupby, remove_distinct, duplication_ratio,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OURS_JSONL = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
THEIRS_JSON = ROOT / "external/DeepEye-SQL/results/bird-dev/qwen3-coder-30b-a3b.json"


def execute(db, sql, timeout=8):
    try:
        c = sqlite3.connect(db, timeout=timeout)
        cur = c.cursor()
        cur.execute(sql)
        r = [tuple(x) for x in cur.fetchall()]
        c.close()
        return r
    except Exception:
        return None


def ex_set(pred_rows, gold_rows):
    """Set-EX (BIRD official): compares row sets, ignores multiplicity."""
    if gold_rows is None:
        return None  # gold execution failed; skip
    if pred_rows is None:
        return 0
    return 1 if set(pred_rows) == set(gold_rows) else 0


def ex_multiset(pred_rows, gold_rows):
    """Multiset-EX (cardinality-aware): preserves row multiplicities.
    Equivalent to comparing bags-of-rows. Stricter than set-EX."""
    if gold_rows is None:
        return None
    if pred_rows is None:
        return 0
    return 1 if sorted(pred_rows, key=lambda r: str(r)) == sorted(gold_rows, key=lambda r: str(r)) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(DEFAULT_OURS_JSONL))
    args = ap.parse_args()
    print(f"Loading our predictions from: {args.ours}")

    their_preds = json.load(open(THEIRS_JSON))
    items = [json.loads(l) for l in open(args.ours)]

    n = 0
    counts = defaultdict(int)  # (eval_method, system) -> count
    by_diff = defaultdict(lambda: defaultdict(int))
    plugin_fires = defaultdict(int)

    for i, it in enumerate(items, 1):
        qid = it["input"]["question_id"]
        db = it["input"]["database_schema"]["db_path"]
        diff = it["input"].get("difficulty", "?")
        gold_sql = it["input"]["gold_sql"]
        gold_rows = execute(db, gold_sql)
        if gold_rows is None:
            continue
        n += 1

        # Baseline (our pred)
        our_pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        our_pred_rows = execute(db, our_pred_sql)

        # + P1 v2
        p1_dec = p1_verify(our_pred_sql, db, pred_rows=our_pred_rows, dup_ratio_threshold=0.80)
        if p1_dec.needs_distinct:
            p1_sql = p1_dec.new_sql
            p1_rows = p1_dec.new_rows
            plugin_fires["P1"] += 1
        else:
            p1_sql = our_pred_sql
            p1_rows = our_pred_rows

        # + P3 on top of P1 result
        p3_eligible = (has_distinct(p1_sql) and not has_aggregate(p1_sql) and not has_groupby(p1_sql))
        p1p3_rows = p1_rows
        if p3_eligible:
            new_sql = remove_distinct(p1_sql)
            if new_sql is not None:
                new_rows = execute(db, new_sql)
                if new_rows is not None and len(new_rows) > len(p1_rows or []):
                    dup_r = duplication_ratio(new_rows)
                    if dup_r <= 0.10:
                        p1p3_rows = new_rows
                        plugin_fires["P3"] += 1

        # DeepEye public artifact
        their_sql = their_preds.get(str(qid))
        their_rows = execute(db, their_sql) if their_sql else None

        # Score under both eval methods
        for eval_name, eval_fn in [("multiset", ex_multiset), ("set", ex_set)]:
            counts[(eval_name, "ours_base")] += eval_fn(our_pred_rows, gold_rows) or 0
            counts[(eval_name, "ours_p1")]   += eval_fn(p1_rows, gold_rows) or 0
            counts[(eval_name, "ours_p1p3")] += eval_fn(p1p3_rows, gold_rows) or 0
            counts[(eval_name, "deepeye")]   += eval_fn(their_rows, gold_rows) or 0
            if eval_name == "set":
                by_diff[diff]["ours_base"] += eval_fn(our_pred_rows, gold_rows) or 0
                by_diff[diff]["ours_p1p3"] += eval_fn(p1p3_rows, gold_rows) or 0
                by_diff[diff]["deepeye"]   += eval_fn(their_rows, gold_rows) or 0
                by_diff[diff]["_n"] += 1

        if i % 200 == 0:
            print(f"  ...{i}/{len(items)} evaluated", flush=True)

    print(f"\n{'='*80}")
    print(f"Comparison: Multiset-EX (cardinality-aware) vs Set-EX (BIRD official)  n={n}")
    print(f"{'='*80}")
    print(f"  {'system':<20}{'Multiset-EX':>16}{'Set-EX (official)':>20}{'Δ':>10}")
    for system in ["ours_base", "ours_p1", "ours_p1p3", "deepeye"]:
        mset = counts[("multiset", system)]
        sset = counts[("set", system)]
        print(f"  {system:<20}{mset:>8} ({mset/n:>6.2%})  "
              f"{sset:>10} ({sset/n:>6.2%}) {sset-mset:>+8d}")

    print(f"\nPlugin fire counts (on raw baseline → P1 → +P3):")
    print(f"  P1 v2 fires:  {plugin_fires['P1']}")
    print(f"  P3 fires:     {plugin_fires['P3']}")

    print(f"\n{'='*80}")
    print(f"By difficulty under Set-EX (BIRD official)")
    print(f"{'='*80}")
    print(f"  {'difficulty':<14}{'n':>6}{'ours_base':>12}{'ours_p1p3':>12}{'deepeye':>10}")
    for d in ["simple", "moderate", "challenging"]:
        bd = by_diff[d]
        nn = bd["_n"]
        if nn == 0:
            continue
        print(f"  {d:<14}{nn:>6}"
              f"  {bd['ours_base']}/{nn}={bd['ours_base']/nn:.1%}"
              f"  {bd['ours_p1p3']}/{nn}={bd['ours_p1p3']/nn:.1%}"
              f"  {bd['deepeye']}/{nn}={bd['deepeye']/nn:.1%}")


if __name__ == "__main__":
    main()
