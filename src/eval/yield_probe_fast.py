#!/usr/bin/env python3
"""Fast yield probe — partial sample of 100 set-EX failures for Pass@12,
plus full structural probes on candidate clusters.

Differences from yield_probe_set_eval.py:
- Random sample of 100 failures for Pass@12 (estimates Pattern 6 ceiling)
- flush=True on all prints + per-10-sample progress
- Use python3 -u when running
"""
import csv
import json
import random
import re
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
S5_ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_generation/bird/dev.snapshot.data/items.jsonl"
S7_ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
CLUSTERS_SET = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/failure_clusters_set.csv"

PASS12_SAMPLE_N = 100  # subsample size for Pattern 6 ceiling estimate


def echo(msg):
    print(msg, flush=True)


def execute(db, sql, timeout_sec=5):
    """Execute with HARD timeout via threading.Timer + conn.interrupt().
    SIGALRM doesn't work for SQLite (signal can't interrupt C-level execute()).
    conn.interrupt() is thread-safe and aborts query at the SQLite C layer."""
    conn = sqlite3.connect(db, timeout=timeout_sec)
    timer = threading.Timer(timeout_sec, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
        return rows
    except Exception:
        return None
    finally:
        timer.cancel()
        conn.close()


def set_match(pred, gold):
    if gold is None or pred is None:
        return False
    return set(pred) == set(gold)


JOIN_CLAUSE_RE = re.compile(
    r"\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|OUTER\s+|FULL\s+|CROSS\s+)?JOIN\s+\S+\s+(?:AS\s+\S+\s+)?ON\s+[^()]+?(?=\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|OUTER\s+|FULL\s+|CROSS\s+)?JOIN\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b|\s+LIMIT\b|\s*;|\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def patch_remove_each_join(sql):
    matches = list(JOIN_CLAUSE_RE.finditer(sql))
    for m in matches:
        yield sql[:m.start()] + sql[m.end():]


SELECT_RE = re.compile(r"(?i)^(\s*SELECT)\s+(.+?)\s+(FROM\b.*)$", re.DOTALL)


def patch_wrap_first_col_in_sum(sql):
    m = SELECT_RE.match(sql)
    if not m:
        return None
    proj = m.group(2)
    first_col = proj.split(",", 1)[0].strip()
    if re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", first_col, re.IGNORECASE):
        return None
    new_proj = f"SUM({first_col})"
    if "," in proj:
        new_proj += ", " + proj.split(",", 1)[1]
    return f"{m.group(1)} {new_proj} {m.group(3)}"


def main():
    fail_rows = list(csv.DictReader(CLUSTERS_SET.open()))
    echo(f"Loaded {len(fail_rows)} set-EX failures")

    label_to_qids = defaultdict(set)
    for r in fail_rows:
        for lab in r["labels"].split(";"):
            label_to_qids[lab].add(int(r["qid"]))

    fail_qid_set = {int(r["qid"]) for r in fail_rows}

    # Subsample for Pass@12
    sampled_qids = set(random.sample(sorted(fail_qid_set), PASS12_SAMPLE_N))
    echo(f"Pass@12 subsample: {len(sampled_qids)} of {len(fail_qid_set)} failures")

    echo("\nLoading S5 candidates...")
    s5_items = {}
    with S5_ITEMS.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in fail_qid_set:
                s5_items[qid] = it
    echo(f"  Got {len(s5_items)} items")

    # === (A) Pass@12 / selection-bug analysis (subsample) ===
    echo(f"\n{'='*80}")
    echo(f"(A) Pattern 6 ceiling: how many subsample failures have a set-EX-matching")
    echo(f"    candidate among S5's 12 generation attempts?")
    echo(f"{'='*80}")

    any_match = 0
    by_diff = defaultdict(lambda: [0, 0])
    selection_fix_qids = []
    processed = 0
    for qid in sorted(sampled_qids):
        if qid not in s5_items:
            continue
        it = s5_items[qid]
        echo(f"    [{processed+1}/{len(sampled_qids)}] qid={qid} db={it['input']['database_id']}")
        gold_rows = execute(it["input"]["database_schema"]["db_path"], it["input"]["gold_sql"])
        cands = it["pipeline_artifacts"]["sql_generation"]["sql_candidates"]
        diff = it["input"].get("difficulty", "?")
        by_diff[diff][1] += 1
        for c in cands:
            cr = execute(it["input"]["database_schema"]["db_path"], c)
            if set_match(cr, gold_rows):
                any_match += 1
                by_diff[diff][0] += 1
                selection_fix_qids.append(qid)
                break
        processed += 1
        if processed % 10 == 0:
            echo(f"  ...{processed}/{len(sampled_qids)}  any_match={any_match} ({any_match/processed:.1%})")

    echo(f"\n  Subsample n={processed}, fixable by better selector: {any_match} ({any_match/processed:.1%})")
    echo(f"  Extrapolated to all 428 failures: ~{int(any_match/processed * 428)} ({any_match/processed * 428/1534*100:+.2f}pp on dev1534)")
    echo(f"\n  {'difficulty':<14}{'any_match':>10}{'total':>8}{'rate':>8}")
    for d in ["simple", "moderate", "challenging"]:
        am, tot = by_diff[d]
        if tot:
            echo(f"  {d:<14}{am:>10}{tot:>8}{am/tot:>7.1%}")

    # === (B) Naive structural fix probes (FULL cluster, not subsample) ===
    echo(f"\n{'='*80}")
    echo(f"(B) Naive structural fix probes (full cluster size)")
    echo(f"{'='*80}")

    s7_items_by_qid = {}
    with S7_ITEMS.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in fail_qid_set:
                s7_items_by_qid[qid] = it

    probes = [
        ("extra_joins_+1", "remove_each_join", patch_remove_each_join),
        ("missing_sum",    "wrap_first_col_in_sum", patch_wrap_first_col_in_sum),
    ]

    for label, name, fn in probes:
        qids = sorted(label_to_qids.get(label, []))
        n_cluster = len(qids)
        n_patched = n_fixed = 0
        fixed_qids = []
        for idx, qid in enumerate(qids, 1):
            if qid not in s7_items_by_qid:
                continue
            it = s7_items_by_qid[qid]
            db = it["input"]["database_schema"]["db_path"]
            pred = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
            gold_rows = execute(db, it["input"]["gold_sql"])

            result = fn(pred)
            if result is None:
                continue
            candidates = [result] if isinstance(result, str) else list(result)
            if not candidates:
                continue
            n_patched += 1
            for cand in candidates:
                cr = execute(db, cand)
                if set_match(cr, gold_rows):
                    n_fixed += 1
                    fixed_qids.append(qid)
                    break
            if idx % 20 == 0:
                echo(f"    {label}: {idx}/{n_cluster}  fixed_so_far={n_fixed}")
        echo(f"\n  {label:<22} cluster={n_cluster:>3}  patched={n_patched:>3}  "
             f"fixed_under_SET={n_fixed:>3}  yield={n_fixed/n_cluster:.1%}  "
             f"pp_on_dev1534={n_fixed/1534*100:+.2f}")
        if fixed_qids:
            echo(f"    fixed qids: {fixed_qids[:15]}{'...' if len(fixed_qids) > 15 else ''}")


if __name__ == "__main__":
    main()
