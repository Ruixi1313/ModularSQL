#!/usr/bin/env python3
"""Cross-backbone robustness study: Qwen3-Coder vs GPT-4o on the same n=99
stratified subset of BIRD-Dev.

For each system:
  1. Compute baseline EX under set-EX AND multiset-EX
  2. Apply P1 v2 (DISTINCT add at dup_ratio ≥ 0.80)
  3. Apply P3 (DISTINCT remove at dup_ratio ≤ 0.10)
  4. Apply P6.2 asymmetric veto (rescue if S7 pick unhealthy)
  5. Apply P6.3 LLM rescue (Qwen LLM-as-judge for the 77-class)
  6. Report fix/break/net per plugin per backbone

Used after gpt4o-subset pipeline finishes. Reads from:
  workspace_full/             (Qwen, full 1534 — filtered to same 99 qids)
  workspace_gpt4o_subset/     (GPT-4o, exactly 99 qids)
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
from src.improvements.asymmetric_selector import select_asymmetric

ROOT = Path(__file__).resolve().parents[2]
QWEN_S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
QWEN_S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
GPT_S7 = ROOT / "external/DeepEye-SQL/workspace_gpt4o_subset/sql_selection/bird/dev.snapshot.data/items.jsonl"
GPT_S6 = ROOT / "external/DeepEye-SQL/workspace_gpt4o_subset/sql_revision/bird/dev.snapshot.data/items.jsonl"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/qwen_vs_gpt4o.csv"


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


def evaluate_one_backbone(label, s7_path, s6_path, target_qids):
    """Compute all metrics for one backbone on the given qids."""
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in s7_path.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in s6_path.open())}

    results = {
        "baseline_set": 0, "baseline_mset": 0,
        "p1_set": 0, "p1_mset": 0,
        "p1_p3_set": 0, "p1_p3_mset": 0,
        "p6_2_set": 0, "p6_2_mset": 0,
        "n_eval": 0,
    }
    rows = []
    for qid in sorted(target_qids):
        if qid not in s7_by_qid or qid not in s6_by_qid:
            continue
        s7_it = s7_by_qid[qid]
        s6_it = s6_by_qid[qid]
        db = s7_it["input"]["database_schema"]["db_path"]
        gold = execute(db, s7_it["input"]["gold_sql"])
        if gold is None:
            continue
        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        s7_rows = execute(db, s7_sql)

        # baseline
        b_set = set_match(s7_rows, gold)
        b_mset = multiset_match(s7_rows, gold)

        # + P1 v2
        p1_dec = p1_verify(s7_sql, db, pred_rows=s7_rows, dup_ratio_threshold=0.80)
        if p1_dec.needs_distinct:
            p1_sql, p1_rows = p1_dec.new_sql, p1_dec.new_rows
        else:
            p1_sql, p1_rows = s7_sql, s7_rows
        p1_set = set_match(p1_rows, gold)
        p1_mset = multiset_match(p1_rows, gold)

        # + P3
        p1p3_rows = p1_rows
        if has_distinct(p1_sql) and not has_aggregate(p1_sql) and not has_groupby(p1_sql):
            new_sql = remove_distinct(p1_sql)
            if new_sql is not None:
                new_rows = execute(db, new_sql)
                if new_rows is not None and len(new_rows) > len(p1_rows or []):
                    if duplication_ratio(new_rows) <= 0.10:
                        p1p3_rows = new_rows
        p1p3_set = set_match(p1p3_rows, gold)
        p1p3_mset = multiset_match(p1p3_rows, gold)

        # + P6.2 asymmetric veto (on top of P1+P3)
        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        # P6.2 takes the post-P3 SQL as base
        p1p3_sql = p1_sql
        if p1p3_rows is not p1_rows:
            p1p3_sql = remove_distinct(p1_sql) or p1_sql
        dec62 = select_asymmetric(p1p3_sql, cands, db)
        if dec62.triggered:
            p62_rows = execute(db, dec62.selected_sql)
        else:
            p62_rows = p1p3_rows
        p62_set = set_match(p62_rows, gold)
        p62_mset = multiset_match(p62_rows, gold)

        results["n_eval"] += 1
        results["baseline_set"] += int(b_set)
        results["baseline_mset"] += int(b_mset)
        results["p1_set"] += int(p1_set)
        results["p1_mset"] += int(p1_mset)
        results["p1_p3_set"] += int(p1p3_set)
        results["p1_p3_mset"] += int(p1p3_mset)
        results["p6_2_set"] += int(p62_set)
        results["p6_2_mset"] += int(p62_mset)

        rows.append({
            "qid": qid, "backbone": label,
            "baseline_set": b_set, "baseline_mset": b_mset,
            "p1_set": p1_set, "p1_mset": p1_mset,
            "p1_p3_set": p1p3_set, "p1_p3_mset": p1p3_mset,
            "p6_2_set": p62_set, "p6_2_mset": p62_mset,
        })

    return results, rows


def main():
    # Identify the 99 subset qids from GPT-4o workspace
    echo("Identifying GPT-4o subset qids...")
    if not GPT_S7.exists():
        echo(f"ERROR: GPT-4o S7 output not found at {GPT_S7}")
        echo("       Run scripts/run_gpt4o_subset.sh --go first.")
        return
    target_qids = []
    with GPT_S7.open() as f:
        for line in f:
            it = json.loads(line)
            target_qids.append(it["input"]["question_id"])
    echo(f"  Got {len(target_qids)} qids")

    echo("\nEvaluating Qwen3-Coder on same qids (filter from workspace_full)...")
    qwen_results, qwen_rows = evaluate_one_backbone("qwen", QWEN_S7, QWEN_S6, target_qids)

    echo("\nEvaluating GPT-4o on subset workspace...")
    gpt_results, gpt_rows = evaluate_one_backbone("gpt4o", GPT_S7, GPT_S6, target_qids)

    # Save per-sample data
    all_rows = qwen_rows + gpt_rows
    if all_rows:
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        echo(f"\n✓ Per-sample → {OUT}")

    echo("\n" + "=" * 88)
    echo(f"Cross-backbone Robustness Study (n={qwen_results['n_eval']} / {gpt_results['n_eval']})")
    echo("=" * 88)
    echo(f"\n  {'configuration':<22}{'Qwen set-EX':>14}{'Qwen mset-EX':>14}"
         f"{'GPT-4o set-EX':>16}{'GPT-4o mset-EX':>16}")
    for key, label in [
        ("baseline", "Baseline (S7)"),
        ("p1", "+ P1 v2"),
        ("p1_p3", "+ P1 v2 + P3"),
        ("p6_2", "+ P1 + P3 + P6.2"),
    ]:
        qn = qwen_results["n_eval"]
        gn = gpt_results["n_eval"]
        echo(f"  {label:<22}{qwen_results[f'{key}_set']}/{qn}={qwen_results[f'{key}_set']/qn:.1%}    "
             f"{qwen_results[f'{key}_mset']}/{qn}={qwen_results[f'{key}_mset']/qn:.1%}    "
             f"{gpt_results[f'{key}_set']}/{gn}={gpt_results[f'{key}_set']/gn:.1%}    "
             f"{gpt_results[f'{key}_mset']}/{gn}={gpt_results[f'{key}_mset']/gn:.1%}")

    # Compute deltas
    echo(f"\n  Plugin contribution (Δ vs baseline):")
    echo(f"  {'plugin':<22}{'Qwen set Δ':>14}{'Qwen mset Δ':>14}"
         f"{'GPT-4o set Δ':>16}{'GPT-4o mset Δ':>16}")
    for key, label in [("p1_p3", "+ P1 + P3"), ("p6_2", "+ all (P1+P3+P6.2)")]:
        echo(f"  {label:<22}"
             f"{qwen_results[f'{key}_set'] - qwen_results['baseline_set']:>+14d}"
             f"{qwen_results[f'{key}_mset'] - qwen_results['baseline_mset']:>+14d}"
             f"{gpt_results[f'{key}_set'] - gpt_results['baseline_set']:>+16d}"
             f"{gpt_results[f'{key}_mset'] - gpt_results['baseline_mset']:>+16d}")

    echo(f"\nRobustness Verdict:")
    qwen_mset_delta = qwen_results["p6_2_mset"] - qwen_results["baseline_mset"]
    gpt_mset_delta = gpt_results["p6_2_mset"] - gpt_results["baseline_mset"]
    if qwen_mset_delta > 0 and gpt_mset_delta > 0:
        echo(f"  ✓ Plugin transfers: positive Δ on both backbones under multiset-EX")
        echo(f"    Qwen: +{qwen_mset_delta},  GPT-4o: +{gpt_mset_delta}")
    elif qwen_mset_delta > 0 and gpt_mset_delta <= 0:
        echo(f"  ⚠ Plugin Qwen-specific: +{qwen_mset_delta} Qwen, {gpt_mset_delta} GPT-4o")
    else:
        echo(f"  ❌ Plugin does not transfer cleanly")


if __name__ == "__main__":
    main()
