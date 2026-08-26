#!/usr/bin/env python3
"""Cross-backbone Set-EX / Multiset-EX evaluation on DeepEye-SQL released
prediction artifacts for BIRD-Dev.

For each of the three open-source backbones released by DeepEye-SQL,
scores the predicted SQL under both metrics with the SAME denominator
(N=1532; gold-unevaluable qids excluded under 5s timeout).

Inputs:
  external/DeepEye-SQL/results/bird-dev/qwen2.5-coder-32b.json
  external/DeepEye-SQL/results/bird-dev/qwen3-coder-30b-a3b.json
  external/DeepEye-SQL/results/bird-dev/gemma3-27b.json

Gold + db_path: workspace_full/sql_selection.topk3_backup/.../items.jsonl

Output:
  Per-backbone Set-EX and Multiset-EX counts, gap (MBS), and console table.
"""
import json
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
RELEASES_DIR = ROOT / "external/DeepEye-SQL/results/bird-dev"

BACKBONES = [
    ("Qwen2.5-Coder-32B", "qwen2.5-coder-32b.json"),
    ("Qwen3-Coder-30B-A3B", "qwen3-coder-30b-a3b.json"),
    ("Gemma-3-27B", "gemma3-27b.json"),
]

TIMEOUT_SEC = 5


def execute(db, sql, timeout=TIMEOUT_SEC):
    if sql is None:
        return None
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


def ex_set(p, g):
    return 0 if p is None or g is None else (1 if set(p) == set(g) else 0)


def ex_multiset(p, g):
    if p is None or g is None:
        return 0
    return 1 if (sorted(p, key=lambda r: str(r)) ==
                 sorted(g, key=lambda r: str(r))) else 0


def main():
    items = [json.loads(l) for l in ITEMS.open()]
    print(f"Loaded {len(items)} items", flush=True)

    preds_by_backbone = {}
    for name, fname in BACKBONES:
        preds_by_backbone[name] = json.load((RELEASES_DIR / fname).open())
        print(f"  {name}: {len(preds_by_backbone[name])} predictions", flush=True)

    print("\nExecuting gold SQL (one pass, 5s timeout)...", flush=True)
    gold_rows_by_qid = {}
    unevaluable = []
    for i, it in enumerate(items, 1):
        qid = it["input"]["question_id"]
        db = it["input"]["database_schema"]["db_path"]
        gold_sql = it["input"]["gold_sql"]
        rows = execute(db, gold_sql)
        if rows is None:
            unevaluable.append(qid)
        else:
            gold_rows_by_qid[qid] = (db, rows)
        if i % 300 == 0:
            print(f"  ...{i}/{len(items)} (excluded so far: {len(unevaluable)})", flush=True)
    print(f"\nGold unevaluable qids (excluded): {unevaluable}")
    print(f"Evaluable n = {len(gold_rows_by_qid)}")

    print("\nScoring each backbone...\n", flush=True)
    results = {}
    for name, _ in BACKBONES:
        preds = preds_by_backbone[name]
        set_pass = 0
        mset_pass = 0
        for qid, (db, gold_rows) in gold_rows_by_qid.items():
            pred_sql = preds.get(str(qid))
            pred_rows = execute(db, pred_sql) if pred_sql else None
            set_pass += ex_set(pred_rows, gold_rows)
            mset_pass += ex_multiset(pred_rows, gold_rows)
        results[name] = (set_pass, mset_pass)
        print(f"  {name}: Set-EX {set_pass}, Multiset-EX {mset_pass}", flush=True)

    n = len(gold_rows_by_qid)
    print("\n" + "=" * 78)
    print(f"Cross-backbone Set-EX vs Multiset-EX on DeepEye-SQL release artifacts")
    print(f"  N={n}  timeout={TIMEOUT_SEC}s  excluded gold qids: {unevaluable}")
    print("=" * 78)
    print(f"  {'Backbone':<22}{'Set-EX':>16}{'Multiset-EX':>20}{'MBS gap':>14}")
    print("-" * 78)
    for name, _ in BACKBONES:
        s, m = results[name]
        gap = (s - m) / n * 100
        print(f"  {name:<22}"
              f"  {s:>4}/{n} = {s/n:>7.2%}"
              f"   {m:>4}/{n} = {m/n:>7.2%}"
              f"   {gap:>+6.2f}pp")
    print()
    mbs_min = min((results[b][0] - results[b][1]) / n * 100 for b, _ in BACKBONES)
    mbs_max = max((results[b][0] - results[b][1]) / n * 100 for b, _ in BACKBONES)
    print(f"  MBS gap range across backbones: {mbs_min:.2f}--{mbs_max:.2f}pp")


if __name__ == "__main__":
    main()
