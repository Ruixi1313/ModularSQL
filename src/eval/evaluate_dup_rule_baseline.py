#!/usr/bin/env python3
"""Non-LLM rescue baseline: on the 77 rescue-triggered
samples, instead of asking an LLM to pick among the 12 S6 candidates, pick the
healthy candidate with the LOWEST duplication ratio (ties -> lowest index).
If no candidate is healthy, keep the S7 base selection.

Scores under both Set-EX (BIRD official semantics) and Multiset-EX, and
compares per-qid against the LLM rescue results (summary_p63.csv /
unified_eval_v2.csv).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.improvements.llm_rescue_selector import execute, probe_health, dup_ratio

S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
P62_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p62.csv"
P63_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p63.csv"
UNIFIED = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/unified_eval_v2.csv"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_dup_rule_baseline.csv"


def set_match(pred, gold):
    return pred is not None and gold is not None and set(pred) == set(gold)


def mset_match(pred, gold):
    return pred is not None and gold is not None and Counter(pred) == Counter(gold)


def main():
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S7.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S6.open())}

    rescue_qids = [int(r["qid"]) for r in csv.DictReader(P62_CSV.open())
                   if r["triggered"] == "True"]
    p63_by_qid = {int(r["qid"]): r for r in csv.DictReader(P63_CSV.open())}
    unified_by_qid = {int(r["qid"]): r for r in csv.DictReader(UNIFIED.open())}

    # Full-dev baseline/LLM totals for projection.
    base_set_total = sum(int(r["base_set"]) for r in unified_by_qid.values())
    base_mset_total = sum(int(r["base_mset"]) for r in unified_by_qid.values())
    llm_set_total = sum(int(r["modularsql_v2_set"]) for r in unified_by_qid.values())
    llm_mset_total = sum(int(r["modularsql_v2_mset"]) for r in unified_by_qid.values())
    n_total = len(unified_by_qid)
    print(f"Full dev ({n_total}): base Set {base_set_total}, base Mset {base_mset_total}, "
          f"LLM-rescue Set {llm_set_total}, LLM-rescue Mset {llm_mset_total}")
    print(f"Rescue-triggered qids: {len(rescue_qids)}\n")

    rows = []
    set_fix = set_brk = mset_fix = mset_brk = 0
    n_eval = n_no_healthy = n_agree_llm = n_ties = 0

    for i, qid in enumerate(sorted(rescue_qids), 1):
        s7_it = s7_by_qid[qid]
        s6_it = s6_by_qid.get(qid)
        if s6_it is None:
            continue
        db = s7_it["input"]["database_schema"]["db_path"]
        gold_rows = execute(db, s7_it["input"]["gold_sql"])
        if gold_rows is None:
            continue

        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        s7_rows = execute(db, s7_sql)
        base_set = set_match(s7_rows, gold_rows)
        base_mset = mset_match(s7_rows, gold_rows)

        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        cand_stats = []
        for idx, sql in enumerate(cands):
            rows_c = execute(db, sql)
            health = probe_health(rows_c)
            cand_stats.append((idx, sql, rows_c, health,
                               dup_ratio(rows_c) if rows_c else 0.0))

        healthy = [c for c in cand_stats if c[3] == "healthy"]
        if healthy:
            min_ratio = min(c[4] for c in healthy)
            tied = [c for c in healthy if c[4] == min_ratio]
            n_ties += int(len(tied) > 1)
            pick_idx, pick_sql, pick_rows, _, pick_ratio = tied[0]
        else:
            n_no_healthy += 1
            pick_idx, pick_sql, pick_rows, pick_ratio = -1, s7_sql, s7_rows, None

        rule_set = set_match(pick_rows, gold_rows)
        rule_mset = mset_match(pick_rows, gold_rows)

        n_eval += 1
        if rule_set and not base_set: set_fix += 1
        elif not rule_set and base_set: set_brk += 1
        if rule_mset and not base_mset: mset_fix += 1
        elif not rule_mset and base_mset: mset_brk += 1

        llm_idx = int(p63_by_qid[qid]["llm_picked_idx"]) if qid in p63_by_qid else -2
        n_agree_llm += int(pick_idx == llm_idx)

        print(f"  [{i}/{len(rescue_qids)}] qid={qid} picked={pick_idx} "
              f"(dup={pick_ratio if pick_ratio is not None else 'n/a'}) "
              f"set:{int(base_set)}->{int(rule_set)} mset:{int(base_mset)}->{int(rule_mset)} "
              f"llm_picked={llm_idx} n_healthy={len(healthy)}", flush=True)

        rows.append({
            "qid": qid,
            "picked_idx": pick_idx, "picked_dup_ratio": pick_ratio,
            "n_healthy": len(healthy),
            "base_set": int(base_set), "rule_set": int(rule_set),
            "base_mset": int(base_mset), "rule_mset": int(rule_mset),
            "llm_picked_idx": llm_idx,
            "agrees_with_llm": int(pick_idx == llm_idx),
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rule_set_total = base_set_total + set_fix - set_brk
    rule_mset_total = base_mset_total + mset_fix - mset_brk
    print("\n" + "=" * 80)
    print(f"Dup-ratio rule baseline on {n_eval} rescue cases "
          f"(no healthy candidate: {n_no_healthy}, ties at min ratio: {n_ties})")
    print("=" * 80)
    print(f"  Set-EX:  fix {set_fix}, break {set_brk}  "
          f"-> full dev {rule_set_total}/{n_total} = {rule_set_total/n_total:.2%} "
          f"(base {base_set_total}, LLM {llm_set_total})")
    print(f"  Mset-EX: fix {mset_fix}, break {mset_brk}  "
          f"-> full dev {rule_mset_total}/{n_total} = {rule_mset_total/n_total:.2%} "
          f"(base {base_mset_total}, LLM {llm_mset_total})")
    print(f"  Picks agreeing with LLM rescue: {n_agree_llm}/{n_eval}")
    print(f"\nPer-qid results -> {OUT}")


if __name__ == "__main__":
    main()
