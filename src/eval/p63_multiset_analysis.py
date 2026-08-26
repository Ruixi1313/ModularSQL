#!/usr/bin/env python3
"""Re-evaluate Pattern 6.3 (LLM-as-Judge Rescue) under MULTISET-EX
(cardinality-aware) on the 77 trigger cases.

User's theoretical claim:
  Under multiset-EX, baseline S7 should score ~0 on these 77 because they're
  all cartesian-exploded / empty / error → wrong cardinalities.
  Therefore rescue can only ADD fixes, never produce breaks.

This script confirms or refutes that claim with data.
"""
import csv
import json
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
P63 = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p63.csv"


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


def echo(msg):
    print(msg, flush=True)


def main():
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S7.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S6.open())}

    p63_rows = list(csv.DictReader(P63.open()))
    triggered_qids = [int(r["qid"]) for r in p63_rows]
    llm_picks = {int(r["qid"]): int(r["llm_picked_idx"]) for r in p63_rows}

    echo(f"Re-evaluating {len(triggered_qids)} v6.3-triggered cases under both metrics\n")

    s7_set_pass = s7_mset_pass = p63_set_pass = p63_mset_pass = 0
    set_fix = set_break = mset_fix = mset_break = 0
    examples = {"mset_fix": [], "mset_break_hypothetical": [], "set_only_fix": []}

    for i, qid in enumerate(sorted(triggered_qids), 1):
        s7_it = s7_by_qid[qid]
        s6_it = s6_by_qid.get(qid)
        db = s7_it["input"]["database_schema"]["db_path"]
        gold_rows = execute(db, s7_it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        s7_rows = execute(db, s7_sql)

        # LLM-picked candidate (or s7 if parse_fail / no pick)
        picked_idx = llm_picks[qid]
        if picked_idx >= 0:
            cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
            p63_sql = cands[picked_idx]
        else:
            p63_sql = s7_sql
        p63_rows_exec = execute(db, p63_sql)

        # Score under both metrics
        s7_s = set_match(s7_rows, gold_rows)
        s7_m = multiset_match(s7_rows, gold_rows)
        p63_s = set_match(p63_rows_exec, gold_rows)
        p63_m = multiset_match(p63_rows_exec, gold_rows)

        s7_set_pass += int(s7_s); s7_mset_pass += int(s7_m)
        p63_set_pass += int(p63_s); p63_mset_pass += int(p63_m)

        if p63_s and not s7_s: set_fix += 1
        elif not p63_s and s7_s: set_break += 1
        if p63_m and not s7_m:
            mset_fix += 1
            if len(examples["mset_fix"]) < 5:
                examples["mset_fix"].append(qid)
        elif not p63_m and s7_m:
            mset_break += 1
            if len(examples["mset_break_hypothetical"]) < 5:
                examples["mset_break_hypothetical"].append(qid)

    n = len(triggered_qids)
    echo("=" * 80)
    echo(f"Pattern 6.3 on {n} triggered cases — both metrics")
    echo("=" * 80)
    echo(f"  {'metric':<14}{'S7 baseline':>14}{'+ P6.3 rescue':>16}{'fix':>6}{'break':>8}{'net':>6}")
    echo(f"  {'set-EX':<14}{s7_set_pass}/{n}={s7_set_pass/n:.1%}  "
         f"{p63_set_pass}/{n}={p63_set_pass/n:.1%}  {set_fix:>6}{set_break:>8}{p63_set_pass-s7_set_pass:>+6}")
    echo(f"  {'multiset-EX':<14}{s7_mset_pass}/{n}={s7_mset_pass/n:.1%}  "
         f"{p63_mset_pass}/{n}={p63_mset_pass/n:.1%}  {mset_fix:>6}{mset_break:>8}{p63_mset_pass-s7_mset_pass:>+6}")
    echo("")
    echo(f"User's hypothesis: under multiset-EX, baseline S7 should be near 0 on these")
    echo(f"  → actual baseline multiset-EX: {s7_mset_pass}/{n} = {s7_mset_pass/n:.1%}")
    echo(f"  → multiset-EX breaks: {mset_break} (theoretical claim: should be 0)")
    echo("")
    echo(f"Projected full dev1534 multiset-EX delta from v6.3:")
    echo(f"  Existing multiset baseline (with P1 v2 + P3): 1035/1534 = 67.47%")
    echo(f"  + v6.3 net multiset fix:                       +{mset_fix - mset_break}")
    new_total = 1035 + (mset_fix - mset_break)
    echo(f"  → New total: {new_total}/1534 = {new_total/1534:.2%}")


if __name__ == "__main__":
    main()
