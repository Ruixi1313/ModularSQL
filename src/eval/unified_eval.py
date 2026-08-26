#!/usr/bin/env python3
"""Unified Set-EX / Multiset-EX evaluation on full BIRD-Dev.

For each evaluable sample (gold SQL executes successfully under 5s),
scores four pipeline configurations under both Set-EX and Multiset-EX
with the SAME denominator:

  config              SQL source
  -----------------   --------------------------------------------------
  ours_base           DeepEye-SQL reproduction (S7 final_selected_sql)
  ours_p1             ours_base + Pattern 1 v2 (DISTINCT inject @ tau=0.80)
  ours_p1p3           ours_p1 + Pattern 3 (DISTINCT remove @ tau=0.10)
  ours_modularsql     ours_p1p3 + Pattern 6.3 LLM rescue
                        (for the 77 P6.3-triggered qids, override SQL
                         with the LLM-picked candidate from summary_p63)

Outputs:
  - Console: integer counts and percentages, same denominator both metrics
  - CSV: per-sample passes for both metrics across all four configs

Timeout: 5s (matches evaluate_p63.py canonical Set-EX baseline 1104/1532).
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
OUT_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/unified_eval.csv"

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


def ex_set(pred_rows, gold_rows):
    if pred_rows is None or gold_rows is None:
        return 0
    return 1 if set(pred_rows) == set(gold_rows) else 0


def ex_multiset(pred_rows, gold_rows):
    if pred_rows is None or gold_rows is None:
        return 0
    return 1 if (sorted(pred_rows, key=lambda r: str(r)) ==
                 sorted(gold_rows, key=lambda r: str(r))) else 0


def load_p63_overrides():
    """Returns dict {qid: override_sql} for the 77 P6.3-triggered cases.
    For each, the SQL is the LLM-picked candidate from sql_candidates_after_revision.
    """
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


def main():
    print(f"Loading items from: {ITEMS}", flush=True)
    items = [json.loads(l) for l in ITEMS.open()]
    print(f"  {len(items)} items loaded", flush=True)

    print(f"Loading P6.3 overrides from: {P63_CSV}", flush=True)
    p63_overrides = load_p63_overrides()
    print(f"  {len(p63_overrides)} P6.3 SQL overrides loaded", flush=True)

    n_total = len(items)
    n_eval = 0
    n_gold_unevaluable = 0
    counts = defaultdict(int)
    by_diff = defaultdict(lambda: defaultdict(int))
    per_row = []
    plugin_fires = defaultdict(int)

    for i, it in enumerate(items, 1):
        qid = it["input"]["question_id"]
        db = it["input"]["database_schema"]["db_path"]
        diff = it["input"].get("difficulty", "?")
        gold_sql = it["input"]["gold_sql"]

        gold_rows = execute(db, gold_sql)
        if gold_rows is None:
            n_gold_unevaluable += 1
            continue
        n_eval += 1

        base_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        base_rows = execute(db, base_sql)

        p1_dec = p1_verify(base_sql, db, pred_rows=base_rows, dup_ratio_threshold=0.80)
        if p1_dec.needs_distinct:
            p1_sql = p1_dec.new_sql
            p1_rows = p1_dec.new_rows
            plugin_fires["P1"] += 1
        else:
            p1_sql = base_sql
            p1_rows = base_rows

        p1p3_sql = p1_sql
        p1p3_rows = p1_rows
        if has_distinct(p1_sql) and not has_aggregate(p1_sql) and not has_groupby(p1_sql):
            new_sql = remove_distinct(p1_sql)
            if new_sql is not None:
                new_rows = execute(db, new_sql)
                if new_rows is not None and len(new_rows) > len(p1_rows or []):
                    dup_r = duplication_ratio(new_rows)
                    if dup_r <= 0.10:
                        p1p3_sql = new_sql
                        p1p3_rows = new_rows
                        plugin_fires["P3"] += 1

        if qid in p63_overrides:
            modsql_sql = p63_overrides[qid]
            modsql_rows = execute(db, modsql_sql)
            plugin_fires["P63"] += 1
        else:
            modsql_sql = p1p3_sql
            modsql_rows = p1p3_rows

        rec = {
            "qid": qid, "db_id": it["input"].get("database_id", "?"),
            "difficulty": diff,
        }
        for cfg_name, rows in [
            ("base", base_rows), ("p1", p1_rows),
            ("p1p3", p1p3_rows), ("modularsql", modsql_rows),
        ]:
            s = ex_set(rows, gold_rows)
            m = ex_multiset(rows, gold_rows)
            counts[("set", cfg_name)] += s
            counts[("multiset", cfg_name)] += m
            rec[f"{cfg_name}_set"] = s
            rec[f"{cfg_name}_mset"] = m
            if cfg_name == "modularsql":
                by_diff[diff]["set"] += s
                by_diff[diff]["mset"] += m
                by_diff[diff]["_n"] += 1
        per_row.append(rec)

        if i % 200 == 0:
            print(f"  ...{i}/{n_total}  evaluated={n_eval}  "
                  f"P1_fires={plugin_fires['P1']}  P3_fires={plugin_fires['P3']}  "
                  f"P6.3_fires={plugin_fires['P63']}", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row[0].keys()))
        w.writeheader()
        w.writerows(per_row)

    n = n_eval
    print("\n" + "=" * 80)
    print(f"Unified Set-EX / Multiset-EX evaluation (n={n}, timeout={TIMEOUT_SEC}s)")
    print(f"  total items: {n_total};  gold unevaluable (excluded): {n_gold_unevaluable}")
    print("=" * 80)
    print(f"  {'config':<14}{'Set-EX':>22}{'Multiset-EX':>22}{'gap':>10}")
    print(f"  {'':<14}{'count / pct':>22}{'count / pct':>22}{'pp':>10}")
    print("-" * 80)
    for cfg in ["base", "p1", "p1p3", "modularsql"]:
        s = counts[("set", cfg)]
        m = counts[("multiset", cfg)]
        gap = (s - m) / n * 100
        print(f"  {cfg:<14}"
              f"  {s:>5}/{n}={s/n:>7.2%}"
              f"   {m:>5}/{n}={m/n:>7.2%}"
              f"   {gap:>+6.2f}")

    base_s = counts[("set", "base")]
    base_m = counts[("multiset", "base")]
    mod_s = counts[("set", "modularsql")]
    mod_m = counts[("multiset", "modularsql")]
    print()
    print(f"ModularSQL vs DeepEye baseline (same n={n}):")
    print(f"  Set-EX:      {base_s}/{n} = {base_s/n:.4%}  -->  "
          f"{mod_s}/{n} = {mod_s/n:.4%}   "
          f"Delta = {mod_s-base_s:+d} ({(mod_s-base_s)/n*100:+.4f}pp)")
    print(f"  Multiset-EX: {base_m}/{n} = {base_m/n:.4%}  -->  "
          f"{mod_m}/{n} = {mod_m/n:.4%}   "
          f"Delta = {mod_m-base_m:+d} ({(mod_m-base_m)/n*100:+.4f}pp)")
    print()
    print(f"MBS gap (Set-EX minus Multiset-EX, same predictions):")
    print(f"  DeepEye baseline:  {(base_s-base_m)/n*100:+.4f}pp")
    print(f"  ModularSQL:        {(mod_s-mod_m)/n*100:+.4f}pp")
    print(f"  MBS closed by:     {((base_s-base_m) - (mod_s-mod_m))/n*100:+.4f}pp")
    print()
    print(f"Plugin fire counts:")
    print(f"  P1 v2 fires:  {plugin_fires['P1']}")
    print(f"  P3 fires:     {plugin_fires['P3']}")
    print(f"  P6.3 fires:   {plugin_fires['P63']}")
    print()
    print(f"By difficulty (ModularSQL):")
    print(f"  {'difficulty':<14}{'n':>6}{'Set-EX':>16}{'Multiset-EX':>16}")
    for d in ["simple", "moderate", "challenging"]:
        bd = by_diff[d]
        nn = bd["_n"]
        if nn == 0:
            continue
        print(f"  {d:<14}{nn:>6}  {bd['set']}/{nn}={bd['set']/nn:.2%}  "
              f"{bd['mset']}/{nn}={bd['mset']/nn:.2%}")
    print(f"\nPer-sample CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
