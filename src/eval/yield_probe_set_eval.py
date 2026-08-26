#!/usr/bin/env python3
"""Yield probe under SET evaluation — explore what can fix BIRD-EX failures.

Two angles:
(A) Selection-bug analysis (Pattern 6 candidate):
    For each set-eval failure, check if ANY of the 12 S5 candidates already
    matches gold under set-eval. If many do, a better selector could close
    the gap without any structural rewriting.

(B) Naive structural fix probe:
    For each cluster of structural failures, try a naive fix transform,
    re-execute, check set-match.
"""
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S5_ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_generation/bird/dev.snapshot.data/items.jsonl"
S7_ITEMS = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
CLUSTERS_SET = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/failure_clusters_set.csv"


def execute(db, sql, timeout=8):
    try:
        c = sqlite3.connect(db, timeout=timeout)
        cur = c.cursor()
        cur.execute(sql)
        r = [tuple(x) for x in cur.fetchall()]
        c.close()
        return r
    except Exception:
        return None


def set_match(pred, gold):
    if gold is None or pred is None:
        return False
    return set(pred) == set(gold)


# === Naive transforms ===

JOIN_CLAUSE_RE = re.compile(
    r"\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|OUTER\s+|FULL\s+|CROSS\s+)?JOIN\s+\S+\s+(?:AS\s+\S+\s+)?ON\s+[^()]+?(?=\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|OUTER\s+|FULL\s+|CROSS\s+)?JOIN\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b|\s+LIMIT\b|\s*;|\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def patch_remove_each_join(sql):
    """Yield variants with each JOIN clause removed one at a time."""
    matches = list(JOIN_CLAUSE_RE.finditer(sql))
    for m in matches:
        yield sql[:m.start()] + sql[m.end():]


SELECT_RE = re.compile(r"(?i)^(\s*SELECT)\s+(.+?)\s+(FROM\b.*)$", re.DOTALL)


def patch_wrap_first_col_in_sum(sql):
    """Wrap first projected column in SUM(...)."""
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


# === Main ===

def main():
    # Load failures by label
    fail_rows = list(csv.DictReader(CLUSTERS_SET.open()))
    print(f"Loaded {len(fail_rows)} set-eval failures\n")

    label_to_qids = defaultdict(set)
    for r in fail_rows:
        for lab in r["labels"].split(";"):
            label_to_qids[lab].add(int(r["qid"]))

    fail_qid_set = {int(r["qid"]) for r in fail_rows}

    # Load S5 candidates for fail qids
    print("Loading S5 candidates for fail qids...")
    s5_items = {}
    with S5_ITEMS.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in fail_qid_set:
                s5_items[qid] = it
    print(f"  Got {len(s5_items)} items\n")

    # === (A) Pass@12 / selection-bug analysis under set-eval ===
    print("=" * 80)
    print("(A) Selection-bug analysis: how many set-eval failures have a")
    print("    correct candidate among S5's 12 attempts (under set-eval)?")
    print("=" * 80)
    any_match = 0
    by_diff = defaultdict(lambda: [0, 0])  # [any_match, total]
    selection_fix_qids = []
    for qid, it in s5_items.items():
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
    print(f"  Failures with at least one set-matching candidate: "
          f"{any_match}/{len(s5_items)} ({any_match/len(s5_items):.1%})")
    print(f"  (This is the Pattern 6 'better selector' upper bound)")
    print()
    print(f"  {'difficulty':<14}{'any_match':>10}{'total':>8}{'rate':>8}")
    for d in ["simple", "moderate", "challenging"]:
        am, tot = by_diff[d]
        if tot:
            print(f"  {d:<14}{am:>10}{tot:>8}{am/tot:>7.1%}")
    print()

    # === (B) Naive structural fix probes ===
    print("=" * 80)
    print("(B) Naive structural fix probes on top candidate clusters")
    print("=" * 80)

    # Load S7 (where pred SQL lives) for naive transform tests
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
        for qid in qids:
            if qid not in s7_items_by_qid:
                continue
            it = s7_items_by_qid[qid]
            db = it["input"]["database_schema"]["db_path"]
            pred = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
            gold_rows = execute(db, it["input"]["gold_sql"])

            # Generator vs single transform
            patched_iter = fn(pred) if hasattr(fn, "__call__") else None
            if hasattr(fn, "__call__"):
                # may be a generator
                result = fn(pred)
                if result is None:
                    continue
                if isinstance(result, str):
                    candidates = [result]
                else:
                    candidates = list(result)
                if not candidates:
                    continue
                n_patched += 1
                for cand in candidates:
                    cr = execute(db, cand)
                    if set_match(cr, gold_rows):
                        n_fixed += 1
                        fixed_qids.append(qid)
                        break
        print(f"\n  {label:<22} cluster={n_cluster:>3}  patched={n_patched:>3}  "
              f"fixed_under_SET={n_fixed:>3}  yield={n_fixed/n_cluster:.1%}  "
              f"pp_on_dev1534={n_fixed/1534*100:+.2f}")
        if fixed_qids:
            print(f"    fixed qids: {fixed_qids[:15]}{'...' if len(fixed_qids) > 15 else ''}")


if __name__ == "__main__":
    main()
