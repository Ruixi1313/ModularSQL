#!/usr/bin/env python3
"""Compute Pass@12 ceiling on REVISED (S6) candidates for full 428 set-EX failures.

The earlier yield_probe_fast.py used a 100-sample subsample of RAW (S5) candidates
and got 35% pass rate → +9.77pp ceiling estimate. This recomputes on the FAIR pool
(S6 revised) and the FULL 428 failures to get the precise selector-bottleneck claim
for the paper.
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


def set_match(pred, gold):
    if pred is None or gold is None:
        return False
    return set(pred) == set(gold)


def echo(msg):
    print(msg, flush=True)


def main():
    echo("Loading S6 + S7 datasets...")
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S7.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S6.open())}

    # Step 1: find ALL set-EX failures in the baseline
    echo("Identifying set-EX failures on baseline...")
    fail_qids = []
    n_eval = base_pass = 0
    for qid, s7_it in s7_by_qid.items():
        db = s7_it["input"]["database_schema"]["db_path"]
        gold_rows = execute(db, s7_it["input"]["gold_sql"])
        if gold_rows is None:
            continue
        n_eval += 1
        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        s7_rows = execute(db, s7_sql)
        if set_match(s7_rows, gold_rows):
            base_pass += 1
        else:
            fail_qids.append(qid)
    echo(f"  Eval={n_eval}, Baseline pass={base_pass}, Failures={len(fail_qids)}")

    # Step 2: for each failure, check if ANY revised candidate matches gold
    echo(f"\nComputing Pass@12 ceiling on REVISED candidates for all {len(fail_qids)} failures...")
    any_match = 0
    by_diff = defaultdict(lambda: [0, 0])
    matched_qids = []
    for i, qid in enumerate(sorted(fail_qids), 1):
        echo(f"  [{i}/{len(fail_qids)}] qid={qid}")
        s7_it = s7_by_qid[qid]
        s6_it = s6_by_qid.get(qid)
        if s6_it is None:
            continue
        db = s7_it["input"]["database_schema"]["db_path"]
        gold_rows = execute(db, s7_it["input"]["gold_sql"])
        diff = s7_it["input"].get("difficulty", "?")
        by_diff[diff][1] += 1

        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        for c in cands:
            cr = execute(db, c)
            if set_match(cr, gold_rows):
                any_match += 1
                by_diff[diff][0] += 1
                matched_qids.append(qid)
                break

    echo("\n" + "=" * 80)
    echo(f"Pass@12 Ceiling on REVISED candidates (full {len(fail_qids)} set-EX failures)")
    echo("=" * 80)
    echo(f"  Failures with at least one revised-candidate match: {any_match}/{len(fail_qids)} = {any_match/len(fail_qids):.1%}")
    echo(f"  → Selector ceiling: +{any_match} samples (+{any_match/n_eval*100:.2f}pp on dev1534)")
    echo(f"  → Final EX achievable: ({base_pass}+{any_match})/{n_eval} = {(base_pass+any_match)/n_eval:.2%}")
    echo("")
    echo(f"  {'difficulty':<14}{'matched':>10}{'failures':>10}{'rate':>8}")
    for d in ["simple", "moderate", "challenging"]:
        m, t = by_diff[d]
        if t:
            echo(f"  {d:<14}{m:>10}{t:>10}{m/t:>7.1%}")

    # Compare to raw ceiling (35% / +9.77pp)
    echo("")
    echo("Comparison to RAW (S5) ceiling:")
    echo("  RAW (sampled n=100):   35.0% match → +9.77pp (extrapolated)")
    echo(f"  REVISED (full n={len(fail_qids)}): {any_match/len(fail_qids)*100:.1f}% match → +{any_match/n_eval*100:.2f}pp")


if __name__ == "__main__":
    main()
