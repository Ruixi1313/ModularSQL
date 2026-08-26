#!/usr/bin/env python3
"""Measure wall-clock latency of the LLM rescue path.

Re-runs the rescue on the 77 flagged queries with timing instrumentation,
decomposing end-to-end rescue latency into (a) candidate-pool execution
(12 SQLs, 5s timeout each) and (b) the LLM API call itself.
Cost: same as the paper's reported rescue cost (~$0.008 total).
"""
import csv
import json
import os
import sys
import time
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
import src.improvements.llm_rescue_selector as rescue
from src.eval.evaluate_p63 import format_schema

S6 = ROOT / "external/DeepEye-SQL/workspace_full/sql_revision/bird/dev.snapshot.data/items.jsonl"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
P62_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_p62.csv"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/llm_rescue_latency.csv"

_orig_call = rescue.call_openrouter
_llm_time = {"t": 0.0}


def timed_call(prompt, **kw):
    t0 = time.perf_counter()
    try:
        return _orig_call(prompt, **kw)
    finally:
        _llm_time["t"] = time.perf_counter() - t0


rescue.call_openrouter = timed_call


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def main():
    s7_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S7.open())}
    s6_by_qid = {it["input"]["question_id"]: it for it in (json.loads(l) for l in S6.open())}
    rescue_qids = [int(r["qid"]) for r in csv.DictReader(P62_CSV.open())
                   if r["triggered"] == "True"]

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    rows, total_ms, llm_ms, exec_ms = [], [], [], []
    for i, qid in enumerate(sorted(rescue_qids), 1):
        s7_it = s7_by_qid[qid]
        s6_it = s6_by_qid.get(qid)
        if s6_it is None:
            continue
        db = s7_it["input"]["database_schema"]["db_path"]
        s7_sql = s7_it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        cands = s6_it["pipeline_artifacts"]["sql_revision"]["sql_candidates_after_revision"]

        _llm_time["t"] = 0.0
        t0 = time.perf_counter()
        dec = rescue.select_with_llm_rescue(
            question=s7_it["input"]["question"],
            hint=s7_it["input"].get("knowledge", "") or "",
            schema_summary=format_schema(s6_it),
            base_sql=s7_sql, revised_candidates=cands, db_path=db,
        )
        total = (time.perf_counter() - t0) * 1e3
        llm = _llm_time["t"] * 1e3
        total_ms.append(total)
        llm_ms.append(llm)
        exec_ms.append(total - llm)
        print(f"  [{i}/{len(rescue_qids)}] qid={qid} total {total:.0f}ms "
              f"(candidates+base exec {total-llm:.0f}ms, LLM {llm:.0f}ms) "
              f"reason={dec.trigger_reason} picked={dec.llm_picked_idx}", flush=True)
        rows.append({"qid": qid, "total_ms": f"{total:.0f}",
                     "exec_ms": f"{total-llm:.0f}", "llm_ms": f"{llm:.0f}",
                     "reason": dec.trigger_reason, "picked": dec.llm_picked_idx})

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(total_ms)
    print(f"\n=== LLM rescue latency over {n} flagged queries ===")
    for label, v in (("end-to-end", total_ms), ("candidate execution", exec_ms),
                     ("LLM call", llm_ms)):
        print(f"  {label:20s} p50 {pct(v,.5):6.0f}ms  p90 {pct(v,.9):6.0f}ms  "
              f"max {max(v):6.0f}ms  mean {sum(v)/n:6.0f}ms")
    print(f"  Amortized over 1532 queries: {sum(total_ms)/1532:.1f}ms per query")
    print(f"Per-qid -> {OUT}")


if __name__ == "__main__":
    main()
