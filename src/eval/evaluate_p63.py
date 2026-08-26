#!/usr/bin/env python3
"""Evaluate Pattern 6.3 (LLM-as-Judge Rescue) on the 77 rescue-triggered
samples identified in v6.2. For each, ask Qwen3-Coder via OpenRouter to
pick the best candidate. Score under Set-EX.
"""
import csv
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(ROOT))
from src.improvements.llm_rescue_selector import select_with_llm_rescue, execute, probe_health
S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
P62_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p62.csv"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p63.csv"


def set_match(pred, gold):
    return pred is not None and gold is not None and set(pred) == set(gold)


def echo(msg):
    print(msg, flush=True)


def format_schema(it):
    """Extract a concise schema summary. Tables and columns are dicts."""
    tables = it["input"]["database_schema"].get("tables", {})
    if not isinstance(tables, dict):
        return ""
    parts = []
    for tbl_name, tbl in list(tables.items())[:10]:
        cols = list(tbl.get("columns", {}).keys())[:15]
        parts.append(f"  {tbl_name}({', '.join(cols)})")
    return "\n".join(parts)


def main():
    echo("Loading datasets...")
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S7.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S6.open())}

    rescue_qids = []
    with P62_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["triggered"] == "True":
                rescue_qids.append(int(r["qid"]))
    echo(f"  Loaded {len(rescue_qids)} rescue-triggered qids from v6.2")

    if not os.environ.get("OPENROUTER_API_KEY"):
        echo("ERROR: OPENROUTER_API_KEY not set")
        return

    rows = []
    n_eval = base_pass = p63_pass = 0
    fix = brk = neutral_pass = neutral_fail = 0
    total_prompt_tok = total_completion_tok = 0

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
        s7_correct = set_match(s7_rows, gold_rows)

        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]
        question = s7_it["input"]["question"]
        hint = s7_it["input"].get("knowledge", "") or ""
        schema_summary = format_schema(s6_it)

        echo(f"  [{i}/{len(rescue_qids)}] qid={qid} db={s7_it['input']['database_id']} ...")

        dec = select_with_llm_rescue(
            question=question, hint=hint, schema_summary=schema_summary,
            base_sql=s7_sql, revised_candidates=cands, db_path=db,
        )

        if dec.triggered and dec.llm_picked_idx >= 0:
            p63_rows = execute(db, dec.selected_sql)
        else:
            p63_rows = s7_rows
        p63_correct = set_match(p63_rows, gold_rows)

        n_eval += 1
        base_pass += int(s7_correct)
        p63_pass += int(p63_correct)
        if p63_correct and not s7_correct: fix += 1
        elif not p63_correct and s7_correct: brk += 1
        elif p63_correct and s7_correct: neutral_pass += 1
        else: neutral_fail += 1
        total_prompt_tok += dec.prompt_tokens
        total_completion_tok += dec.completion_tokens

        echo(f"      base_pass={s7_correct} p63_pass={p63_correct} llm_picked={dec.llm_picked_idx} reason={dec.trigger_reason}")

        rows.append({
            "qid": qid,
            "s7_pass": s7_correct, "p63_pass": p63_correct,
            "llm_picked_idx": dec.llm_picked_idx,
            "reason": dec.trigger_reason,
            "prompt_tok": dec.prompt_tokens,
            "completion_tok": dec.completion_tokens,
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # cost: qwen3-coder $0.07 in, $0.27 out per M
    cost = total_prompt_tok / 1e6 * 0.07 + total_completion_tok / 1e6 * 0.27
    echo("\n" + "=" * 80)
    echo(f"Pattern 6.3 (LLM-as-Judge Rescue) on {n_eval} rescue cases")
    echo("=" * 80)
    echo(f"  S7 baseline on these 77:      {base_pass}/{n_eval} = {base_pass/n_eval:.2%}")
    echo(f"  + LLM rescue:                  {p63_pass}/{n_eval} = {p63_pass/n_eval:.2%}")
    echo(f"  Δ within rescue subset:        {p63_pass-base_pass:+d}")
    echo(f"  fix (FAIL→PASS):              {fix}")
    echo(f"  break (PASS→FAIL):            {brk}")
    echo(f"  fix:break ratio:              {fix/max(brk,1):.2f}:1")
    echo("")
    echo(f"Projected on full dev1534 (Pattern 6.3 = healthy:trust S7 + unhealthy:LLM rescue):")
    echo(f"  baseline:        1104/1532 = 72.06%")
    echo(f"  + P6.3:          {1104+fix-brk}/1532 = {(1104+fix-brk)/1532:.2%}")
    echo(f"  Δ:               {fix-brk:+d} ({(fix-brk)/1532*100:+.2f}pp)")
    echo("")
    echo(f"LLM cost: {total_prompt_tok} prompt + {total_completion_tok} completion tokens → ${cost:.4f}")


if __name__ == "__main__":
    main()
