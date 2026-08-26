#!/usr/bin/env python3
"""
Compute accuracy and pairwise comparison for the 4 ablation configurations:

  B           : DeepEye baseline (workspace/sql_selection)
  B + P1      : Baseline + Pattern 1 DISTINCT verifier (applied to B's SQL)
  B + P2      : Pattern 2 (workspace_p2/sql_selection — pipeline rerun with enriched schema)
  B + P1 + P2 : Pattern 1 verifier applied on top of Pattern 2's SQL
"""
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.distinct_verifier import verify_and_fix, execute_sql, add_distinct


WS = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace"
WS2 = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace_p2"
B_SNAP = WS / "sql_selection/bird/dev.snapshot.data/items.jsonl"
P2_SNAP = WS2 / "sql_selection/bird/dev.snapshot.data/items.jsonl"
OUT_CSV = Path(__file__).resolve().parents[2] / "results/pattern2_4config_compare.csv"


def _norm(rows):
    return sorted(rows, key=lambda r: str(r))


def evaluate(pred_sql, db_path, gold_rows):
    pred_rows, err = execute_sql(db_path, pred_sql)
    if err or pred_rows is None:
        return False
    return _norm(pred_rows) == _norm(gold_rows or [])


def apply_pattern1(pred_sql, db_path):
    """Return (new_sql, fired). fired=True means DISTINCT was injected."""
    decision = verify_and_fix(pred_sql, db_path)
    if decision.needs_distinct:
        return add_distinct(pred_sql), True
    return pred_sql, False


def main():
    if not P2_SNAP.exists():
        print(f"P2 snapshot not found: {P2_SNAP}")
        sys.exit(1)

    b_items = {json.loads(l)["input"]["question_id"]: json.loads(l) for l in open(B_SNAP)}
    p2_items = {json.loads(l)["input"]["question_id"]: json.loads(l) for l in open(P2_SNAP)}
    common_qids = sorted(set(b_items) & set(p2_items))
    print(f"Common questions: {len(common_qids)}\n")

    by_config = {cfg: defaultdict(lambda: [0, 0]) for cfg in
                 ["B", "B+P1", "B+P2", "B+P1+P2"]}
    overall = {cfg: 0 for cfg in by_config}
    rows = []

    for qid in common_qids:
        b = b_items[qid]
        p2 = p2_items[qid]
        db_id = b["input"]["database_id"]
        diff = b["input"]["difficulty"]
        db_path = b["input"]["database_schema"]["db_path"]
        gold = b["input"]["gold_sql"]
        gold_rows, _ = execute_sql(db_path, gold)

        b_sql = b["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        p2_sql = p2["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]

        b_p1_sql, b_p1_fired = apply_pattern1(b_sql, db_path)
        p2_p1_sql, p2_p1_fired = apply_pattern1(p2_sql, db_path)

        results = {
            "B":          evaluate(b_sql,    db_path, gold_rows),
            "B+P1":       evaluate(b_p1_sql, db_path, gold_rows),
            "B+P2":       evaluate(p2_sql,   db_path, gold_rows),
            "B+P1+P2":    evaluate(p2_p1_sql, db_path, gold_rows),
        }
        for cfg, ok in results.items():
            if ok: overall[cfg] += 1
            by_config[cfg][db_id][1] += 1
            if ok: by_config[cfg][db_id][0] += 1
            by_config[cfg][diff][1] += 1
            if ok: by_config[cfg][diff][0] += 1

        rows.append({
            "qid": qid, "db_id": db_id, "difficulty": diff,
            "B":           "PASS" if results["B"]       else "FAIL",
            "B+P1":        "PASS" if results["B+P1"]    else "FAIL",
            "B+P2":        "PASS" if results["B+P2"]    else "FAIL",
            "B+P1+P2":     "PASS" if results["B+P1+P2"] else "FAIL",
            "p1_fired_on_B":  "Y" if b_p1_fired  else "",
            "p1_fired_on_P2": "Y" if p2_p1_fired else "",
            "b_sql":          b_sql[:200],
            "p2_sql":         p2_sql[:200],
            "gold_sql":       gold[:200],
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

    n = len(common_qids)
    print("=" * 70)
    print(f"Accuracy on 99-sample dev (each DB × 9 questions)")
    print("=" * 70)
    for cfg in ["B", "B+P1", "B+P2", "B+P1+P2"]:
        ok = overall[cfg]
        print(f"  {cfg:12s} {ok}/{n} = {ok/n:.2%}")

    print("\nDeltas vs B:")
    base = overall["B"]
    for cfg in ["B+P1", "B+P2", "B+P1+P2"]:
        d = overall[cfg] - base
        print(f"  {cfg:12s} Δ = {d:+d}  ({d/n*100:+.2f}pp)")

    # Per-difficulty
    print("\nBy difficulty (PASS/total):")
    print(f"  {'difficulty':12s} {'B':>10s} {'B+P1':>10s} {'B+P2':>10s} {'B+P1+P2':>12s}")
    for d in ["simple", "moderate", "challenging"]:
        row = [f"  {d:12s}"]
        for cfg in ["B", "B+P1", "B+P2", "B+P1+P2"]:
            ok, tot = by_config[cfg].get(d, [0, 0])
            row.append(f"{ok}/{tot}".rjust(10))
        # B+P1+P2 column padded to 12
        row[-1] = row[-1].rjust(12)
        print("".join(row))

    # Per-DB
    print("\nBy database (PASS/total):")
    print(f"  {'db_id':28s} {'B':>10s} {'B+P1':>10s} {'B+P2':>10s} {'B+P1+P2':>12s}")
    db_ids = sorted({k for cfg in by_config.values() for k in cfg.keys()
                     if k not in ("simple", "moderate", "challenging")})
    for db in db_ids:
        row = [f"  {db:28s}"]
        for cfg in ["B", "B+P1", "B+P2", "B+P1+P2"]:
            ok, tot = by_config[cfg].get(db, [0, 0])
            row.append(f"{ok}/{tot}".rjust(10))
        row[-1] = row[-1].rjust(12)
        print("".join(row))

    print(f"\n📄 Per-sample CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
