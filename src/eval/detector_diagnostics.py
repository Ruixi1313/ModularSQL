#!/usr/bin/env python3
"""Detector diagnostic metrics:
precision / recall / F1 of the anomaly detector, per-reason precision,
a threshold sweep for the dup-ratio trigger, and wall-clock latency of the
detection check and the DISTINCT-patch execution.

Ground truth for "should have been flagged": the base SQL is wrong under
Multiset-EX (primary; also reported vs Set-EX). Runs over three systems:
the DeepEye-SQL reproduction and the two released cross-pipeline artifacts.
"""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.improvements.llm_rescue_selector import execute, dup_ratio
from src.improvements.distinct_verifier import (
    has_distinct, has_aggregate, has_groupby, has_join, add_distinct,
)
from src.eval.cross_pipeline_guardrail import load_predictions, _LIMIT_RE

DEV = ROOT / "data/bird/dev/dev.json"
DB_ROOT = ROOT / "data/bird/dev/dev_databases"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
OUT_DIR = ROOT / "results/ModularSQL_Baseline_dev1534_20260514"
TAU_DEPLOYED = 0.80
TAU_SWEEP = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def set_match(p, g):
    return p is not None and g is not None and set(p) == set(g)


def mset_match(p, g):
    return p is not None and g is not None and Counter(p) == Counter(g)


def deepeye_preds():
    for line in S7.open():
        it = json.loads(line)
        yield (int(it["input"]["question_id"]),
               it["input"]["database_schema"]["db_path"],
               it["input"]["gold_sql"],
               it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"],
               )


def file_preds(path):
    dev = json.load(DEV.open())
    preds = load_predictions(Path(path))
    for item in dev:
        qid = int(item["question_id"])
        db = str(DB_ROOT / item["db_id"] / f"{item['db_id']}.sqlite")
        yield qid, db, item["SQL"], preds[qid]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def run_system(name, records):
    rows = []
    detect_us, patch_ms = [], []
    for qid, db, gold_sql, pred_sql in records:
        gold_rows = execute(db, gold_sql)
        if gold_rows is None:
            continue
        pred_rows = execute(db, pred_sql)

        t0 = time.perf_counter()
        if pred_rows is None:
            health, dr = "exec_error", None
        elif len(pred_rows) == 0:
            health, dr = "empty_result", 0.0
        else:
            dr = dup_ratio(pred_rows)
            health = "cartesian_explosion" if dr >= TAU_DEPLOYED else "healthy"
        detect_us.append((time.perf_counter() - t0) * 1e6)

        p_fired = 0
        p_time = None
        if health == "cartesian_explosion" and not has_distinct(pred_sql) \
                and not has_aggregate(pred_sql) and not has_groupby(pred_sql) \
                and has_join(pred_sql) and not _LIMIT_RE.search(pred_sql):
            patched = add_distinct(pred_sql)
            if patched is not None:
                t0 = time.perf_counter()
                execute(db, patched)
                p_time = (time.perf_counter() - t0) * 1e3
                patch_ms.append(p_time)
                p_fired = 1

        rows.append({
            "qid": qid, "health": health,
            "dup_ratio": f"{dr:.4f}" if dr is not None else "",
            "set_ok": int(set_match(pred_rows, gold_rows)),
            "mset_ok": int(mset_match(pred_rows, gold_rows)),
            "patch_fired": p_fired,
            "detect_us": f"{detect_us[-1]:.1f}",
            "patch_ms": f"{p_time:.1f}" if p_time is not None else "",
        })

    out_csv = OUT_DIR / f"detector_diag_{name}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"\n=== {name} ({n} evaluable) ===")
    for metric in ("mset_ok", "set_ok"):
        tp = sum(1 for r in rows if r["health"] != "healthy" and not r[metric])
        fp = sum(1 for r in rows if r["health"] != "healthy" and r[metric])
        fn = sum(1 for r in rows if r["health"] == "healthy" and not r[metric])
        p, rr, f1 = prf(tp, fp, fn)
        print(f"  Detector vs {metric[:-3]}-wrong: TP {tp} FP {fp} FN {fn} "
              f"-> P {p:.3f} R {rr:.3f} F1 {f1:.3f}")
    print("  Per-reason precision (vs mset-wrong):")
    for reason in ("exec_error", "empty_result", "cartesian_explosion"):
        sub = [r for r in rows if r["health"] == reason]
        if sub:
            tp = sum(1 for r in sub if not r["mset_ok"])
            print(f"    {reason:22s} {tp}/{len(sub)} = {tp/len(sub):.3f}")
    print("  Dup-ratio threshold sweep (flag = err|empty|dup>=tau, vs mset-wrong):")
    for tau in TAU_SWEEP:
        tp = fp = fn = 0
        for r in rows:
            flagged = (r["health"] in ("exec_error", "empty_result")
                       or (r["dup_ratio"] and float(r["dup_ratio"]) >= tau))
            wrong = not int(r["mset_ok"])
            if flagged and wrong: tp += 1
            elif flagged: fp += 1
            elif wrong: fn += 1
        p, rr, f1 = prf(tp, fp, fn)
        star = " <- deployed" if tau == TAU_DEPLOYED else ""
        print(f"    tau={tau:.2f}: flags {tp+fp:4d}  P {p:.3f} R {rr:.3f} F1 {f1:.3f}{star}")
    print(f"  Latency: detect check p50 {pct(detect_us,.5):.0f}us "
          f"p90 {pct(detect_us,.9):.0f}us max {max(detect_us)/1000:.1f}ms; "
          f"patch exec n={len(patch_ms)} p50 {pct(patch_ms,.5):.0f}ms "
          f"p90 {pct(patch_ms,.9):.0f}ms max {max(patch_ms) if patch_ms else 0:.0f}ms")
    print(f"  Per-qid -> {out_csv}")


def main():
    run_system("deepeye", deepeye_preds())
    run_system("dailsql", file_preds(ROOT / "data/xpipe_predictions/dail_bird.txt"))
    run_system("turbo_kg", file_preds(ROOT / "data/xpipe_predictions/turbo_kg_predict_dev.json"))


if __name__ == "__main__":
    main()
