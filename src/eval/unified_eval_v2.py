#!/usr/bin/env python3
"""Variant of unified_eval.py with P6.3 applied BEFORE P1+P3.

Pipeline order:
  S7 raw SQL
  -> if qid in P6.3 triggered (77): override with LLM-picked candidate
  -> apply P1 v2 (DISTINCT inject)
  -> apply P3 (DISTINCT remove)
  -> ModularSQL final

Rationale: P6.3's probe runs on raw S7. If LLM picks a candidate, that
candidate may still benefit from deterministic DISTINCT verification.
Applying P1+P3 AFTER P6.3 lets deterministic patches refine the LLM pick.
"""
import csv
import json
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improvements.distinct_verifier import verify_and_fix as p1_verify
from src.improvements.distinct_remover import (
    has_distinct, has_aggregate, has_groupby, remove_distinct, duplication_ratio,
)

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
S6_REVISED = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
P63_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p63.csv"
OUT_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/unified_eval_v2.csv"

TIMEOUT_SEC = 5


def execute(db, sql, timeout=TIMEOUT_SEC):
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


def load_p63_overrides():
    p63_rows = list(csv.DictReader(P63_CSV.open()))
    s6_by_qid = {it["input"]["question_id"]: it
                 for it in (json.loads(l) for l in S6_REVISED.open())}
    overrides = {}
    for r in p63_rows:
        qid = int(r["qid"])
        picked = int(r["llm_picked_idx"])
        if picked < 0:
            continue
        s6_it = s6_by_qid.get(qid)
        if s6_it is None:
            continue
        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        if picked < len(cands):
            overrides[qid] = cands[picked]
    return overrides


def apply_p1p3(sql, rows, db, fires):
    p1_dec = p1_verify(sql, db, pred_rows=rows, dup_ratio_threshold=0.80)
    if p1_dec.needs_distinct:
        sql2 = p1_dec.new_sql
        rows2 = p1_dec.new_rows
        fires["P1"] += 1
    else:
        sql2, rows2 = sql, rows

    if has_distinct(sql2) and not has_aggregate(sql2) and not has_groupby(sql2):
        new_sql = remove_distinct(sql2)
        if new_sql is not None:
            new_rows = execute(db, new_sql)
            if new_rows is not None and len(new_rows) > len(rows2 or []):
                if duplication_ratio(new_rows) <= 0.10:
                    sql2, rows2 = new_sql, new_rows
                    fires["P3"] += 1
    return sql2, rows2


def main():
    items = [json.loads(l) for l in ITEMS.open()]
    overrides = load_p63_overrides()
    print(f"{len(items)} items, {len(overrides)} P6.3 overrides", flush=True)

    n_eval = 0
    counts = defaultdict(int)
    by_diff = defaultdict(lambda: defaultdict(int))
    fires = defaultdict(int)
    per_row = []

    for i, it in enumerate(items, 1):
        qid = it["input"]["question_id"]
        db = it["input"]["database_schema"]["db_path"]
        gold_sql = it["input"]["gold_sql"]
        gold_rows = execute(db, gold_sql)
        if gold_rows is None:
            continue
        n_eval += 1

        base_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        base_rows = execute(db, base_sql)

        if qid in overrides:
            start_sql = overrides[qid]
            start_rows = execute(db, start_sql)
            fires["P63"] += 1
        else:
            start_sql = base_sql
            start_rows = base_rows

        mod_sql, mod_rows = apply_p1p3(start_sql, start_rows, db, fires)

        diff = it["input"].get("difficulty", "?")
        rec = {"qid": qid, "diff": diff}
        for cfg_name, rows in [("base", base_rows), ("modularsql_v2", mod_rows)]:
            s = ex_set(rows, gold_rows)
            m = ex_multiset(rows, gold_rows)
            counts[("set", cfg_name)] += s
            counts[("multiset", cfg_name)] += m
            rec[f"{cfg_name}_set"] = s
            rec[f"{cfg_name}_mset"] = m
            by_diff[diff][f"{cfg_name}_set"] += s
            by_diff[diff][f"{cfg_name}_mset"] += m
        by_diff[diff]["_n"] += 1
        per_row.append(rec)

        if i % 200 == 0:
            print(f"  ...{i}/{len(items)} eval={n_eval} "
                  f"P1={fires['P1']} P3={fires['P3']} P63={fires['P63']}", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row[0].keys()))
        w.writeheader()
        w.writerows(per_row)

    n = n_eval
    print(f"\n{'='*80}")
    print(f"Pipeline order: P6.3 first, then P1+P3 (n={n})")
    print(f"{'='*80}")
    for cfg in ["base", "modularsql_v2"]:
        s = counts[("set", cfg)]
        m = counts[("multiset", cfg)]
        print(f"  {cfg:<16}  Set-EX {s}/{n}={s/n:.4%}   "
              f"Multiset-EX {m}/{n}={m/n:.4%}")
    bs, bm = counts[("set", "base")], counts[("multiset", "base")]
    ms, mm = counts[("set", "modularsql_v2")], counts[("multiset", "modularsql_v2")]
    print(f"\nDelta vs base:")
    print(f"  Set-EX:      {ms-bs:+d} ({(ms-bs)/n*100:+.4f}pp)")
    print(f"  Multiset-EX: {mm-bm:+d} ({(mm-bm)/n*100:+.4f}pp)")
    print(f"\nFires: P1={fires['P1']}  P3={fires['P3']}  P6.3={fires['P63']}")

    print(f"\nBy difficulty:")
    print(f"  {'difficulty':<14}{'n':>6}{'base Set':>14}{'base Mset':>14}"
          f"{'mod Set':>14}{'mod Mset':>14}")
    for d in ["simple", "moderate", "challenging"]:
        bd = by_diff[d]
        nn = bd["_n"]
        if nn == 0:
            continue
        print(f"  {d:<14}{nn:>6}"
              f"  {bd['base_set']}/{nn}={bd['base_set']/nn:.2%}"
              f"  {bd['base_mset']}/{nn}={bd['base_mset']/nn:.2%}"
              f"  {bd['modularsql_v2_set']}/{nn}={bd['modularsql_v2_set']/nn:.2%}"
              f"  {bd['modularsql_v2_mset']}/{nn}={bd['modularsql_v2_mset']/nn:.2%}")


if __name__ == "__main__":
    main()
