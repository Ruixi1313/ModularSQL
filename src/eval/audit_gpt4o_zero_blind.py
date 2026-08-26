#!/usr/bin/env python3
"""Audit the 'GPT-4o 0% blind spot' result before accepting it.

Three checks:
  1. Execution success rate — are GPT-4o SQLs actually running or just erroring?
  2. Failure mode shift — did GPT-4o commit DIFFERENT errors than Qwen on the
     same qids (schema/value errors vs multiplicity errors)?
  3. Manual sample inspection — for 5-10 representative cases from
     missing_distinct / extra_distinct / extra_joins_+1, dump:
       Question, gold SQL, GPT-4o SQL, gold row count, GPT-4o row count
     and check whether any case has the Set-EX==pass / Multiset-EX==fail
     pattern (i.e. multiplicity-only failure).
"""
import csv
import json
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPT_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/gpt4o_qualitative.csv"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"


def execute(db, sql, timeout=5):
    conn = sqlite3.connect(db, timeout=timeout)
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return [tuple(r) for r in cur.fetchall()], None
    except Exception as e:
        return None, str(e)
    finally:
        timer.cancel()
        conn.close()


def echo(m):
    print(m, flush=True)


def main():
    items_by_qid = {it["input"]["question_id"]: it
                    for it in (json.loads(l) for l in S7.open())}
    rows = list(csv.DictReader(GPT_CSV.open()))

    # === CHECK 1: Execution success rate ===
    echo("=" * 80)
    echo("CHECK 1: GPT-4o execution success rate by stratum")
    echo("=" * 80)
    by_stratum = defaultdict(lambda: {"n": 0, "exec_ok": 0, "set_pass": 0,
                                       "mset_pass": 0, "blind": 0})
    for r in rows:
        s = r["stratum"]
        by_stratum[s]["n"] += 1
        if r["exec_err"] != "True":
            by_stratum[s]["exec_ok"] += 1
        if r["set_pass"] == "True":
            by_stratum[s]["set_pass"] += 1
        if r["mset_pass"] == "True":
            by_stratum[s]["mset_pass"] += 1
        if r["blind"] == "True":
            by_stratum[s]["blind"] += 1

    echo(f"  {'stratum':<24}{'n':>4}{'exec_ok':>10}{'set_pass':>10}"
         f"{'mset_pass':>11}{'blind':>8}")
    for s, c in by_stratum.items():
        echo(f"  {s:<24}{c['n']:>4}  {c['exec_ok']}/{c['n']}  "
             f"{c['set_pass']}/{c['n']}  {c['mset_pass']}/{c['n']}  {c['blind']}/{c['n']}")

    total_exec_ok = sum(c["exec_ok"] for c in by_stratum.values())
    total_n = sum(c["n"] for c in by_stratum.values())
    echo(f"\n  Overall exec success: {total_exec_ok}/{total_n} = {total_exec_ok/total_n:.0%}")
    echo(f"  → Excluding exec errors leaves n={total_exec_ok} for blind-spot detection")

    # === CHECK 2: Of the exec_ok SQLs, how many are valid-but-wrong (set fail)?
    echo("\n" + "=" * 80)
    echo("CHECK 2: Among exec-success SQLs, breakdown by outcome")
    echo("=" * 80)
    echo("  set=PASS, mset=PASS:  SQL is fully correct (no multiplicity issue)")
    echo("  set=PASS, mset=FAIL:  BLIND SPOT (this is what we're looking for)")
    echo("  set=FAIL:             SQL is semantically wrong (different gold set)")
    echo("")

    outcomes = defaultdict(lambda: defaultdict(int))
    for r in rows:
        s = r["stratum"]
        if r["exec_err"] == "True":
            outcomes[s]["exec_err"] += 1
            continue
        if r["set_pass"] == "True" and r["mset_pass"] == "True":
            outcomes[s]["both_pass"] += 1
        elif r["set_pass"] == "True" and r["mset_pass"] != "True":
            outcomes[s]["BLIND"] += 1
        else:
            outcomes[s]["set_fail"] += 1

    echo(f"  {'stratum':<24}{'both_pass':>11}{'BLIND':>8}{'set_fail':>10}{'exec_err':>10}")
    for s, c in outcomes.items():
        echo(f"  {s:<24}{c['both_pass']:>11}{c['BLIND']:>8}{c['set_fail']:>10}{c['exec_err']:>10}")

    total_both_pass = sum(c["both_pass"] for c in outcomes.values())
    total_blind = sum(c["BLIND"] for c in outcomes.values())
    total_set_fail = sum(c["set_fail"] for c in outcomes.values())
    total_err = sum(c["exec_err"] for c in outcomes.values())
    echo(f"\n  OVERALL: both_pass={total_both_pass}  BLIND={total_blind}  "
         f"set_fail={total_set_fail}  exec_err={total_err}")
    echo(f"\n  Interpretation:")
    echo(f"    If set_fail+exec_err is large → most queries get a different result set")
    echo(f"      → blind spot can't manifest because pred_set != gold_set already")
    echo(f"    If both_pass is high but blind=0 → GPT-4o really is multiplicity-clean")

    # === CHECK 3: Manual inspection of 5-10 cases (high-priority strata) ===
    echo("\n" + "=" * 80)
    echo("CHECK 3: Manual sample inspection (DISTINCT/JOIN strata)")
    echo("=" * 80)
    inspect_strata = {"missing_distinct", "extra_distinct", "extra_joins_+1"}
    inspected = 0
    for r in rows:
        if inspected >= 9:
            break
        if r["stratum"] not in inspect_strata:
            continue
        qid = int(r["qid"])
        it = items_by_qid[qid]
        db = it["input"]["database_schema"]["db_path"]
        gold_sql = it["input"]["gold_sql"]
        gpt_sql = r["gpt4o_sql"]
        gold_rows, _ = execute(db, gold_sql)
        gpt_rows, gpt_err = execute(db, gpt_sql)
        echo(f"\n  --- qid={qid}  stratum={r['stratum']}  set={r['set_pass']}  mset={r['mset_pass']} ---")
        echo(f"     Q:    {it['input']['question'][:130]}")
        echo(f"     GOLD: {gold_sql[:200]}")
        echo(f"     GPT4o: {gpt_sql[:200]}")
        echo(f"     gold rows: n={len(gold_rows) if gold_rows is not None else 'ERR'}")
        echo(f"     gpt rows:  n={len(gpt_rows) if gpt_rows is not None else 'ERR'}"
             f"  {'(error: ' + (gpt_err or '')[:60] + ')' if gpt_err else ''}")
        # diagnose the failure mode
        if gpt_rows is None:
            echo(f"     diagnosis: GPT-4o SQL errored — NOT a blind-spot candidate")
        elif gold_rows is None:
            echo(f"     diagnosis: gold timed out — skip")
        else:
            gold_set = set(gold_rows)
            gpt_set = set(gpt_rows)
            if gpt_set == gold_set:
                if sorted(gold_rows, key=str) == sorted(gpt_rows, key=str):
                    echo(f"     diagnosis: both pass — GPT-4o produced clean output")
                else:
                    echo(f"     diagnosis: **BLIND SPOT** — sets match but multisets differ")
            else:
                # check WHY sets differ
                only_in_gold = gold_set - gpt_set
                only_in_gpt = gpt_set - gold_set
                echo(f"     diagnosis: sets differ — gold has {len(only_in_gold)} unique rows GPT lacks; "
                     f"GPT has {len(only_in_gpt)} extra unique rows")
        inspected += 1

    # === Final synthesis ===
    echo("\n" + "=" * 80)
    echo("AUDIT VERDICT")
    echo("=" * 80)
    if total_set_fail + total_err > 30:
        echo(f"  ⚠ {total_set_fail + total_err}/{total_n} = "
             f"{(total_set_fail + total_err)/total_n:.0%} of SQLs have wrong set "
             f"or errored — too sparse to reliably measure blind-spot rate.")
        echo(f"  → 0% blind spot is likely a sample-size artifact, NOT GPT-4o robustness.")
    elif total_blind == 0 and total_both_pass > 15:
        echo(f"  ✓ {total_both_pass} clean passes with 0 blind spots → "
             f"GPT-4o produces multiplicity-clean outputs (on this stratum).")
    else:
        echo(f"  ? Mixed signals — see Check 3 manual inspection.")


if __name__ == "__main__":
    main()
