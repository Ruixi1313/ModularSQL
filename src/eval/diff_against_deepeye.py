#!/usr/bin/env python3
"""Sample-by-sample comparison vs DeepEye's published BIRD-Dev predictions.

DeepEye officially released their Qwen3-Coder-30B predictions at
  external/DeepEye-SQL/results/bird-dev/qwen3-coder-30b-a3b.json
as a dict {qid_str: sql_str} for all 1534 dev questions.

We execute both their SQL and ours against the actual DB, compare to gold,
and categorize each sample into:
  - both_correct
  - their_only     ← THE GAP (they right, we wrong)
  - our_only       ← our independent wins
  - both_wrong

Outputs:
  - Headline counts + difficulty + DB breakdowns
  - CSV of all 1534 samples with both SQLs for manual inspection
  - List of "their_only" qids (sorted), structured by DB and difficulty

Usage:
  python3 src/eval/diff_against_deepeye.py
  python3 src/eval/diff_against_deepeye.py --ours workspace_full/sql_selection.topk3_backup/...
"""
import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OURS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"
DEFAULT_THEIRS = ROOT / "external/DeepEye-SQL/results/bird-dev/qwen3-coder-30b-a3b.json"
OUT_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/diff_vs_deepeye.csv"


def norm(rows):
    if rows is None:
        return None
    return tuple(sorted([tuple(r) for r in rows], key=lambda r: str(r)))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(DEFAULT_OURS))
    ap.add_argument("--theirs", default=str(DEFAULT_THEIRS))
    args = ap.parse_args()

    their_preds = json.load(open(args.theirs))
    print(f"Loaded {len(their_preds)} predictions from DeepEye")

    our_items = [json.loads(l) for l in open(args.ours)]
    print(f"Loaded {len(our_items)} predictions from ours\n")

    rows = []
    cats = Counter()
    by_diff = defaultdict(Counter)
    by_db = defaultdict(Counter)
    their_only_qids = []
    our_only_qids = []

    for i, it in enumerate(our_items, 1):
        qid = it["input"]["question_id"]
        db = it["input"]["database_schema"]["db_path"]
        diff = it["input"].get("difficulty", "?")
        db_id = it["input"]["database_id"]
        gold_sql = it["input"]["gold_sql"]
        our_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        their_sql = their_preds.get(str(qid))
        if their_sql is None:
            continue

        gold = norm(execute(db, gold_sql))
        if gold is None:
            continue

        ours = norm(execute(db, our_sql))
        theirs = norm(execute(db, their_sql))

        ours_correct = ours == gold
        theirs_correct = theirs == gold

        if ours_correct and theirs_correct:
            cat = "both_correct"
        elif theirs_correct and not ours_correct:
            cat = "their_only"
            their_only_qids.append((qid, diff, db_id))
        elif ours_correct and not theirs_correct:
            cat = "our_only"
            our_only_qids.append((qid, diff, db_id))
        else:
            cat = "both_wrong"

        cats[cat] += 1
        by_diff[diff][cat] += 1
        by_db[db_id][cat] += 1

        rows.append({
            "qid": qid, "db_id": db_id, "difficulty": diff,
            "category": cat,
            "ours_correct": ours_correct, "theirs_correct": theirs_correct,
            "our_sql": our_sql[:300],
            "their_sql": their_sql[:300],
            "gold_sql": gold_sql[:300],
        })

        if i % 200 == 0:
            print(f"  ...{i}/{len(our_items)} executed", flush=True)

    n = sum(cats.values())
    print(f"\n{'='*72}")
    print(f"Diff vs DeepEye published predictions (n={n})")
    print(f"{'='*72}")
    our_acc = (cats['both_correct'] + cats['our_only']) / n
    their_acc = (cats['both_correct'] + cats['their_only']) / n
    print(f"  Our  accuracy: {cats['both_correct'] + cats['our_only']:>4} / {n} = {our_acc:.2%}")
    print(f"  Their accuracy: {cats['both_correct'] + cats['their_only']:>4} / {n} = {their_acc:.2%}")
    print(f"  Gap: {(their_acc - our_acc) * 100:+.2f}pp\n")

    print(f"  {'category':<20}{'count':>8}{'pct':>8}")
    for cat in ["both_correct", "their_only", "our_only", "both_wrong"]:
        c = cats[cat]
        print(f"  {cat:<20}{c:>8}{c/n:>7.1%}")

    print(f"\n{'='*72}")
    print("By difficulty")
    print(f"{'='*72}")
    print(f"  {'diff':<14}{'both':>8}{'their_only':>12}{'our_only':>10}{'both_wrong':>12}{'gap':>8}")
    for d in ["simple", "moderate", "challenging"]:
        c = by_diff[d]
        if not c:
            continue
        gap = c["their_only"] - c["our_only"]
        print(f"  {d:<14}{c['both_correct']:>8}{c['their_only']:>12}{c['our_only']:>10}{c['both_wrong']:>12}{gap:>+8}")

    print(f"\n{'='*72}")
    print("By database (sorted by gap)")
    print(f"{'='*72}")
    db_gaps = [(db, by_db[db]["their_only"] - by_db[db]["our_only"], by_db[db]) for db in by_db]
    db_gaps.sort(key=lambda x: -x[1])
    print(f"  {'db':<26}{'their_only':>12}{'our_only':>10}{'gap':>6}")
    for db, gap, c in db_gaps:
        print(f"  {db:<26}{c['their_only']:>12}{c['our_only']:>10}{gap:>+6}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Per-sample diff → {OUT_CSV}")
    print(f"  ({cats['their_only']} 'their_only' qids — these are the gap)")


if __name__ == "__main__":
    main()
