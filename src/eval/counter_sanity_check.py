#!/usr/bin/env python3
"""Counter vs sorted-by-str sanity check for Multiset-EX implementation.

Verifies that `collections.Counter` equality and `sorted(rows, key=str)`
equality give identical results on Qwen3-Coder-30B-A3B BIRD-Dev predictions.

Expected outcome: 0 differences across 1532 evaluable samples — the two
implementations are mathematically equivalent for BIRD-Dev data.

Run-time: ~10 minutes (1 backbone × 1532 samples × 2 SQL executions each).
"""
import csv
import json
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
DEEPEYE_PREDS = ROOT / "external/DeepEye-SQL/results/bird-dev/qwen3-coder-30b-a3b.json"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/counter_sanity_check.csv"


def execute(db, sql, timeout=5):
    conn = sqlite3.connect(db, timeout=timeout)
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return [tuple(r) for r in cur.fetchall()]
    except Exception:
        return None
    finally:
        timer.cancel()
        conn.close()


def match_sorted_str(p, g):
    if p is None or g is None:
        return False
    return sorted(p, key=lambda r: str(r)) == sorted(g, key=lambda r: str(r))


def match_counter(p, g):
    if p is None or g is None:
        return False
    return Counter(p) == Counter(g)


def echo(m):
    print(m, flush=True)


def main():
    echo("Loading Qwen3-Coder-30B-A3B predictions from DeepEye release...")
    preds = json.load(DEEPEYE_PREDS.open())
    echo(f"  {len(preds)} predictions loaded")

    items = [json.loads(l) for l in ITEMS.open()]
    echo(f"  {len(items)} BIRD-Dev items loaded\n")

    n_eval = 0
    sorted_pass = 0
    counter_pass = 0
    diffs = []
    rows = []

    for i, it in enumerate(items, 1):
        qid = it["input"]["question_id"]
        if str(qid) not in preds:
            continue
        db = it["input"]["database_schema"]["db_path"]
        gold_rows = execute(db, it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        pred_sql = preds[str(qid)]
        pred_rows = execute(db, pred_sql)
        if pred_rows is None:
            continue

        n_eval += 1
        s = match_sorted_str(pred_rows, gold_rows)
        c = match_counter(pred_rows, gold_rows)
        sorted_pass += int(s)
        counter_pass += int(c)

        rows.append({
            "qid": qid,
            "sorted_match": s,
            "counter_match": c,
            "agreement": s == c,
        })

        if s != c:
            diffs.append({
                "qid": qid,
                "db_id": it["input"]["database_id"],
                "sorted_str": s,
                "counter": c,
                "n_pred": len(pred_rows),
                "n_gold": len(gold_rows),
                "pred_first5": str(pred_rows[:5]),
                "gold_first5": str(gold_rows[:5]),
            })

        if i % 200 == 0:
            echo(f"  ...{i}/{len(items)}  sorted={sorted_pass} counter={counter_pass} diffs={len(diffs)}")

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    echo("\n" + "=" * 70)
    echo(f"Counter vs sorted+str sanity check (Qwen3-Coder-30B-A3B, n={n_eval})")
    echo("=" * 70)
    echo(f"  sorted+str passes: {sorted_pass}/{n_eval} = {sorted_pass/n_eval:.2%}")
    echo(f"  Counter passes:    {counter_pass}/{n_eval} = {counter_pass/n_eval:.2%}")
    echo(f"  Differences:       {len(diffs)}")
    echo("")
    if len(diffs) == 0:
        echo("  RESULT: Two implementations are EQUIVALENT on this data.")
        echo("  Conclusion: Either implementation is correct for Multiset-EX.")
    else:
        echo(f"  RESULT: {len(diffs)} discrepancies detected — see CSV for details.")
        echo("  First 5 discrepancy qids:")
        for d in diffs[:5]:
            echo(f"    qid={d['qid']}  db={d['db_id']}  sorted={d['sorted_str']}  counter={d['counter']}")
            echo(f"      pred[:5]={d['pred_first5']}")
            echo(f"      gold[:5]={d['gold_first5']}")
    echo(f"\n  Per-sample log: {OUT}")


if __name__ == "__main__":
    main()
