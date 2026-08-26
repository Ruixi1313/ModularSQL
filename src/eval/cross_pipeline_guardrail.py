#!/usr/bin/env python3
"""Cross-pipeline validation of the ModularSQL guardrail.

Applies the anomaly detector (probe_health: exec error / empty result /
dup_ratio >= 0.80) and the deterministic DISTINCT patch (Pattern 1, identical
gating to distinct_verifier.verify_and_fix, tau = 0.80) to the released final
SQLs of other Text-to-SQL pipelines on BIRD-Dev. The LLM rescue stage is not
applicable because these releases contain a single SQL per question (no
candidate pool); this is the detection + deterministic-patch layer only.

Inputs (single final SQL per question, qid-aligned with dev.json):
  --pred file.txt   one SQL per line (DAIL-SQL release format), or
  --pred file.json  {qid: "SQL\t----- bird -----\tdb_id"} (BIRD baseline format)

Reports: base Set-EX / Multiset-EX (MBS gap), flagged counts by reason,
patch fires, fix/break under both metrics.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.improvements.llm_rescue_selector import execute, probe_health, dup_ratio
from src.improvements.distinct_verifier import (
    has_distinct, has_aggregate, has_groupby, has_join, add_distinct,
)

DEV = ROOT / "data/bird/dev/dev.json"
DB_ROOT = ROOT / "data/bird/dev/dev_databases"
TAU = 0.80


def set_match(pred, gold):
    return pred is not None and gold is not None and set(pred) == set(gold)


def mset_match(pred, gold):
    return pred is not None and gold is not None and Counter(pred) == Counter(gold)


def load_predictions(path: Path):
    """Return {qid(int): sql(str)}."""
    if path.suffix == ".json":
        raw = json.load(path.open())
        return {int(k): str(v).split("\t")[0].strip() for k, v in raw.items()}
    sqls = [l.strip() for l in path.open()]
    return {i: s for i, s in enumerate(sqls)}


_LIMIT_RE = __import__("re").compile(r"\bLIMIT\b", __import__("re").IGNORECASE)


def distinct_patch(sql, rows, limit_guard=False):
    """Pattern-1 gating, identical to verify_and_fix but with the caller
    owning (timeout-guarded) execution. Returns patched sql or None.

    limit_guard: skip queries containing LIMIT — DISTINCT dedups before the
    LIMIT cut, so injection can change even the set-semantics result
    (found via qid 1122 on DAIL-SQL predictions)."""
    if rows is None or has_distinct(sql) or has_aggregate(sql) \
            or has_groupby(sql) or not has_join(sql):
        return None
    if limit_guard and _LIMIT_RE.search(sql):
        return None
    if len(rows) < 2 or len(set(rows)) == len(rows):
        return None
    if dup_ratio(rows) < TAU:
        return None
    return add_distinct(sql)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit-guard", action="store_true")
    args = ap.parse_args()

    dev = json.load(DEV.open())
    preds = load_predictions(Path(args.pred))
    assert len(preds) == len(dev), (len(preds), len(dev))

    out_rows = []
    n_eval = 0
    base_set = base_mset = new_set = new_mset = 0
    flags = Counter()
    n_patch_fired = 0
    set_fix = set_brk = mset_fix = mset_brk = 0

    for item in dev:
        qid = int(item["question_id"])
        db = str(DB_ROOT / item["db_id"] / f"{item['db_id']}.sqlite")
        gold_rows = execute(db, item["SQL"])
        if gold_rows is None:
            continue
        n_eval += 1

        sql = preds[qid]
        rows = execute(db, sql)
        b_set, b_mset = set_match(rows, gold_rows), mset_match(rows, gold_rows)

        health = probe_health(rows)
        flags[health] += 1

        patched_sql = (distinct_patch(sql, rows, limit_guard=args.limit_guard)
                       if health != "healthy" else None)
        # Pattern 1 in the paper also fires on sub-threshold-free healthy dups?
        # No: gating requires dup_ratio >= 0.80, which implies flagged. Patch
        # is attempted whenever gating passes; health!=healthy check above is
        # redundant but keeps flagged/patched accounting aligned.
        if patched_sql is not None:
            p_rows = execute(db, patched_sql)
            if p_rows is None:
                patched_sql = None  # distinct caused error -> fail closed
        if patched_sql is not None:
            n_patch_fired += 1
            f_rows, f_sql = p_rows, patched_sql
        else:
            f_rows, f_sql = rows, sql

        a_set, a_mset = set_match(f_rows, gold_rows), mset_match(f_rows, gold_rows)

        base_set += b_set; base_mset += b_mset
        new_set += a_set; new_mset += a_mset
        if a_set and not b_set: set_fix += 1
        if not a_set and b_set: set_brk += 1
        if a_mset and not b_mset: mset_fix += 1
        if not a_mset and b_mset: mset_brk += 1

        out_rows.append({
            "qid": qid, "difficulty": item["difficulty"], "health": health,
            "patched": int(f_sql is not sql),
            "base_set": int(b_set), "new_set": int(a_set),
            "base_mset": int(b_mset), "new_mset": int(a_mset),
        })

    out_csv = ROOT / f"results/ModularSQL_Baseline_dev1534_20260514/xpipe_{args.name}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n_flagged = sum(v for k, v in flags.items() if k != "healthy")
    print(f"\n=== {args.name} on BIRD-Dev ({n_eval} evaluable) ===")
    print(f"Base:   Set-EX {base_set}/{n_eval} = {base_set/n_eval:.2%}   "
          f"Multiset-EX {base_mset}/{n_eval} = {base_mset/n_eval:.2%}   "
          f"MBS gap {(base_set-base_mset)/n_eval*100:.2f}pp")
    print(f"Flagged: {n_flagged} ({dict(flags)})")
    print(f"DISTINCT patch fired: {n_patch_fired}")
    print(f"Guarded: Set-EX {new_set}/{n_eval} = {new_set/n_eval:.2%} "
          f"(fix {set_fix}, brk {set_brk})   "
          f"Multiset-EX {new_mset}/{n_eval} = {new_mset/n_eval:.2%} "
          f"(fix {mset_fix}, brk {mset_brk})")
    print(f"Per-qid -> {out_csv}")


if __name__ == "__main__":
    main()
