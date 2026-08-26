#!/usr/bin/env python3
"""Verify the 'multiplicity blind spot' is an evaluator-level (model-independent)
problem by re-evaluating ALL THREE published DeepEye-SQL artifacts under both
Set-EX and Multiset-EX.

Backbones tested:
  - qwen3-coder-30b-a3b   (already partially evaluated; baseline check)
  - qwen2.5-coder-32b     (NEW: different size, different generation)
  - gemma3-27b            (NEW: different family entirely)

Cost: $0 (only re-executes their published SQL predictions, no LLM call).
Time: ~10-15 min per backbone (5s timeout per SQL with threading.Timer hard cap).
"""
import csv
import json
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
BACKBONES = {
    "qwen3-coder-30b": ROOT / "external/DeepEye-SQL/results/bird-dev/qwen3-coder-30b-a3b.json",
    "qwen2.5-coder-32b": ROOT / "external/DeepEye-SQL/results/bird-dev/qwen2.5-coder-32b.json",
    "gemma3-27b": ROOT / "external/DeepEye-SQL/results/bird-dev/gemma3-27b.json",
}
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/evaluator_level_verification.csv"


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


def set_match(p, g):
    return p is not None and g is not None and set(p) == set(g)


def multiset_match(p, g):
    if p is None or g is None: return False
    return sorted(p, key=lambda r: str(r)) == sorted(g, key=lambda r: str(r))


def echo(m):
    print(m, flush=True)


def evaluate_backbone(name, preds_path, items):
    echo(f"\n--- {name} ({preds_path.name}) ---")
    preds = json.load(preds_path.open())
    n_eval = set_pass = mset_pass = blind = 0
    by_diff = defaultdict(lambda: {"n": 0, "set": 0, "mset": 0, "blind": 0})
    rows = []
    for i, it in enumerate(items, 1):
        qid = it["input"]["question_id"]
        if str(qid) not in preds:
            continue
        db = it["input"]["database_schema"]["db_path"]
        gold = execute(db, it["input"]["gold_sql"])
        if gold is None:
            continue
        pred_sql = preds[str(qid)]
        pred = execute(db, pred_sql)
        n_eval += 1
        diff = it["input"].get("difficulty", "?")
        by_diff[diff]["n"] += 1
        s_pass = set_match(pred, gold)
        m_pass = multiset_match(pred, gold)
        is_blind = s_pass and not m_pass
        set_pass += int(s_pass)
        mset_pass += int(m_pass)
        blind += int(is_blind)
        by_diff[diff]["set"] += int(s_pass)
        by_diff[diff]["mset"] += int(m_pass)
        by_diff[diff]["blind"] += int(is_blind)
        rows.append({
            "backbone": name, "qid": qid, "db": it["input"]["database_id"],
            "difficulty": diff, "set_pass": s_pass, "mset_pass": m_pass,
            "is_blind": is_blind,
        })
        if i % 200 == 0:
            echo(f"    ...{i}/{len(items)}  set={set_pass}  mset={mset_pass}  blind={blind}")

    echo(f"\n  Total evaluated: {n_eval}")
    echo(f"  Set-EX:      {set_pass}/{n_eval} = {set_pass/n_eval:.2%}")
    echo(f"  Multiset-EX: {mset_pass}/{n_eval} = {mset_pass/n_eval:.2%}")
    echo(f"  Blind-spot:  {blind}/{n_eval} = {blind/n_eval:.2%}  "
         f"(set-correct but multiset-wrong)")
    echo(f"\n  By difficulty:")
    echo(f"    {'difficulty':<14}{'n':>6}{'set-EX':>10}{'mset-EX':>10}{'blind':>8}{'blind %':>10}")
    for d in ["simple", "moderate", "challenging"]:
        x = by_diff[d]
        if x["n"]:
            echo(f"    {d:<14}{x['n']:>6}"
                 f"  {x['set']}/{x['n']}={x['set']/x['n']:.1%}"
                 f"  {x['mset']}/{x['n']}={x['mset']/x['n']:.1%}"
                 f"  {x['blind']}  {x['blind']/x['n']:.1%}")
    return {"n": n_eval, "set": set_pass, "mset": mset_pass, "blind": blind}, rows


def main():
    items = [json.loads(l) for l in ITEMS.open()]
    echo(f"Loaded {len(items)} BIRD-Dev items")

    summary = {}
    all_rows = []
    for name, preds_path in BACKBONES.items():
        if not preds_path.exists():
            echo(f"  ⚠ skipping {name}: {preds_path} not found")
            continue
        s, rows = evaluate_backbone(name, preds_path, items)
        summary[name] = s
        all_rows.extend(rows)

    if all_rows:
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        echo(f"\n✓ Per-sample data → {OUT}")

    echo("\n" + "=" * 88)
    echo("CROSS-BACKBONE 'MULTIPLICITY BLIND SPOT' VERIFICATION")
    echo("=" * 88)
    echo(f"  {'backbone':<22}{'n':>6}{'Set-EX':>14}{'Multiset-EX':>14}{'Blind-spot':>14}{'pp':>8}")
    for name, s in summary.items():
        pp = s["blind"] / s["n"] * 100
        echo(f"  {name:<22}{s['n']:>6}"
             f"  {s['set']}/{s['n']}={s['set']/s['n']:.1%}"
             f"  {s['mset']}/{s['n']}={s['mset']/s['n']:.1%}"
             f"  {s['blind']:>10}  {pp:>+5.2f}pp")
    echo("")
    if len(summary) >= 2:
        pps = [s["blind"] / s["n"] * 100 for s in summary.values()]
        echo(f"Blind-spot range across backbones: [{min(pps):.2f}pp, {max(pps):.2f}pp]")
        echo(f"Spread: {max(pps) - min(pps):.2f}pp  (smaller = more model-independent)")
        if max(pps) - min(pps) < 3.0:
            echo("→ Consistent blind-spot magnitude across models → "
                 "**evaluator-level problem confirmed**")
        else:
            echo("→ Large variance across models → some model-level component too")


if __name__ == "__main__":
    main()
