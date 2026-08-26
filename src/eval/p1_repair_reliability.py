#!/usr/bin/env python3
"""Pattern 1 Repair Reliability analysis.

For each of the 46 P1-fixed samples, ask:
  Did ANY of the 12 LLM candidates already match gold?
    YES → "selection correction" (P1 indirectly fixed a S6/S7 selection bug)
    NO  → "truly novel fix"        (LLM 12 attempts all wrong; rule contributed
                                    capability orthogonal to LLM scaling)

This is the headline number for Pattern 1's contribution claim in the paper.
"""
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S5 = ROOT / "external/DeepEye-SQL/workspace_full/sql_generation/bird/dev.snapshot.data/items.jsonl"
SUMMARY = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary.csv"


def norm(rows):
    if rows is None:
        return None
    return tuple(sorted([tuple(r) for r in rows], key=lambda r: str(r)))


def execute(db, sql, timeout=8):
    try:
        c = sqlite3.connect(db, timeout=timeout)
        cur = c.cursor()
        cur.execute(sql)
        r = cur.fetchall()
        c.close()
        return norm(r)
    except Exception:
        return None


def main():
    # P1 fired & flipped FAIL→PASS = "P1 fix"
    fixed_qids = set()
    with SUMMARY.open() as f:
        for row in csv.DictReader(f):
            if row["p1_fired"] == "YES" and row["baseline_match"] == "FAIL" and row["p1_match"] == "PASS":
                fixed_qids.add(int(row["qid"]))
    print(f"Loaded {len(fixed_qids)} P1-fixed qids from summary.csv\n")

    items = {}
    with S5.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in fixed_qids:
                items[qid] = it
    print(f"Located {len(items)}/{len(fixed_qids)} matching items in S5 artifact\n")

    truly_novel = []
    selection_correction = []
    by_diff = defaultdict(lambda: [0, 0])  # [novel, selection]

    for qid, it in items.items():
        gold_sql = it["input"]["gold_sql"]
        db = it["input"]["database_schema"]["db_path"]
        diff = it["input"].get("difficulty", "?")
        gold_r = execute(db, gold_sql)
        if gold_r is None:
            print(f"  ! skipping qid={qid}: gold execution failed")
            continue
        cands = it["pipeline_artifacts"]["sql_generation"]["sql_candidates"]
        any_correct = any(execute(db, c) == gold_r for c in cands)
        if any_correct:
            selection_correction.append(qid)
            by_diff[diff][1] += 1
        else:
            truly_novel.append(qid)
            by_diff[diff][0] += 1

    n = len(truly_novel) + len(selection_correction)
    print("=" * 64)
    print(f"Pattern 1 Repair Reliability (n={n})")
    print("=" * 64)
    print(f"  Truly novel (LLM 12 candidates all wrong):   "
          f"{len(truly_novel):>3} / {n}  ({len(truly_novel)/n:.1%})")
    print(f"  Selection correction (correct cand existed): "
          f"{len(selection_correction):>3} / {n}  ({len(selection_correction)/n:.1%})")
    print()
    print(f"  {'Difficulty':<14}{'Novel':>8}{'Selection':>12}{'Total':>8}")
    for d in ["simple", "moderate", "challenging"]:
        nov, sel = by_diff[d]
        if nov + sel:
            print(f"  {d:<14}{nov:>8}{sel:>12}{nov+sel:>8}")
    print()
    print(f"Novel-fix qids:     {sorted(truly_novel)}")
    print(f"Selection-fix qids: {sorted(selection_correction)}")


if __name__ == "__main__":
    main()
