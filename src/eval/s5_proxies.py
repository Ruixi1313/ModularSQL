#!/usr/bin/env python3
"""Compute 3 proxy accuracy metrics from S5 (SQL Generation) output:
  - Pass@12: upper bound (best-of-12 candidates)
  - Majority: most-frequent execution result among 12
  - First:    only the first candidate
Final S7 accuracy will fall somewhere between Majority and Pass@12."""
import json, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path

S5 = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace_full/sql_generation/bird/dev.snapshot.data/items.jsonl"

def norm(rows):
    if rows is None: return None
    return tuple(sorted([tuple(r) for r in rows], key=lambda r: str(r)))

def execute(db, sql):
    try:
        c = sqlite3.connect(db, timeout=8); cur = c.cursor(); cur.execute(sql)
        r = cur.fetchall(); c.close(); return norm(r)
    except Exception: return None

def main():
    items = [json.loads(l) for l in open(S5)]
    n = len(items); pass12=majority=first=0; eval_n=0
    diff = defaultdict(lambda: [0,0,0,0])  # [eval, pass12, maj, first]
    t = time.time()
    for it in items:
        gold = it["input"]["gold_sql"]
        db = it["input"]["database_schema"]["db_path"]
        d = it["input"].get("difficulty", "?")
        gold_r = execute(db, gold)
        if gold_r is None: continue
        cands = it["pipeline_artifacts"]["sql_generation"]["sql_candidates"]
        if not cands: continue
        eval_n += 1; diff[d][0] += 1

        results = [execute(db, s) for s in cands]
        # Pass@12
        if any(r == gold_r for r in results):
            pass12 += 1; diff[d][1] += 1
        # Majority
        cnt = Counter(r for r in results if r is not None)
        if cnt and cnt.most_common(1)[0][0] == gold_r:
            majority += 1; diff[d][2] += 1
        # First
        if results and results[0] == gold_r:
            first += 1; diff[d][3] += 1

        if eval_n % 200 == 0:
            print(f"  ...{eval_n}/{n}  pass@12={pass12/eval_n:.1%} maj={majority/eval_n:.1%} first={first/eval_n:.1%}", flush=True)

    print(f"\nProcessed {eval_n}/{n} items in {time.time()-t:.0f}s\n")
    print("="*60)
    print(f"S5 Candidate Proxy Accuracy (n={eval_n})")
    print("="*60)
    print(f"  Pass@12 (best of 12, UPPER BOUND): {pass12}/{eval_n} = {pass12/eval_n:.2%}")
    print(f"  Majority vote (proxy for S7):       {majority}/{eval_n} = {majority/eval_n:.2%}")
    print(f"  First candidate (LOWER BOUND):      {first}/{eval_n} = {first/eval_n:.2%}")
    print()
    print(f"{'Difficulty':<14}{'Pass@12':>10}{'Majority':>10}{'First':>10}")
    for d in ["simple","moderate","challenging"]:
        if d in diff and diff[d][0]:
            e,p,m,f = diff[d]
            print(f"  {d:<12}{p/e:>10.1%}{m/e:>10.1%}{f/e:>10.1%}")
    print("\n⚠ These are PROXIES. Final S7 accuracy will likely be near Majority value (±2pp).")

if __name__ == "__main__":
    main()
