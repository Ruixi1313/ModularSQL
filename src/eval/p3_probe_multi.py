#!/usr/bin/env python3
"""Probe yield for multiple candidate patterns by applying naive fix +
re-execute. This gives the upper bound on what a rule-based plugin could
achieve. We test:

  - P3a: extra_distinct → remove DISTINCT, see if matches gold
  - P3b: extra_orderby  → remove ORDER BY, see if matches gold
  - P3c: extra_limit    → remove LIMIT, see if matches gold
  - P3d: missing_limit  → not patchable without knowing the limit value (skip)
  - P3e: missing_distinct (P1 didn't fire) → add DISTINCT, see if matches gold

For each, report yield and what fraction of cluster is "purely rule-fixable."
"""
import csv
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"
CLUSTERS = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/failure_clusters.csv"


def norm(rows):
    if rows is None:
        return None
    return sorted([tuple(r) for r in rows], key=lambda r: str(r))


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


DISTINCT_RE = re.compile(r"(\bSELECT)\s+DISTINCT\s+", re.IGNORECASE)
ORDERBY_RE = re.compile(r"\s*\bORDER\s+BY\b[^)]+?(?=\bLIMIT\b|\bUNION\b|\)|;|$)",
                        re.IGNORECASE)
LIMIT_RE = re.compile(r"\s*\bLIMIT\b\s+\d+\s*(?:OFFSET\s+\d+)?\s*", re.IGNORECASE)


def patch_remove_distinct(sql):
    new = DISTINCT_RE.sub(r"\1 ", sql, count=1)
    return new if new != sql else None


def patch_remove_orderby(sql):
    new = ORDERBY_RE.sub("", sql, count=1)
    return new if new != sql else None


def patch_remove_limit(sql):
    new = LIMIT_RE.sub(" ", sql, count=1)
    return new if new != sql else None


def patch_add_distinct(sql):
    if DISTINCT_RE.search(sql):
        return None
    new = re.sub(r"^(\s*SELECT)(\s+)(?!DISTINCT\b)", r"\1 DISTINCT ", sql, count=1)
    return new if new != sql else None


PATTERNS = [
    ("extra_distinct",   "remove_distinct", patch_remove_distinct),
    ("extra_orderby",    "remove_orderby",  patch_remove_orderby),
    ("extra_limit",      "remove_limit",    patch_remove_limit),
    ("missing_distinct", "add_distinct",    patch_add_distinct),
]


def load_failures_by_label():
    out = {}
    with CLUSTERS.open() as f:
        for r in csv.DictReader(f):
            for lab in r["labels"].split(";"):
                out.setdefault(lab, []).append(int(r["qid"]))
    return out


def main():
    label_to_qids = load_failures_by_label()
    relevant_qids = set()
    for label, _, _ in PATTERNS:
        relevant_qids.update(label_to_qids.get(label, []))

    items = {}
    with S7.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in relevant_qids:
                items[qid] = it
    print(f"Loaded {len(items)} relevant failures\n")

    print(f"{'cluster':<20}{'patch':<20}{'cluster_n':>10}{'patched':>10}{'fixed':>8}"
          f"{'yield':>8}{'pp':>8}")
    print("-" * 88)
    for label, patch_name, patch_fn in PATTERNS:
        qids = label_to_qids.get(label, [])
        cluster_n = len(qids)
        patched = fixed = 0
        fixed_qids = []
        for qid in qids:
            if qid not in items:
                continue
            it = items[qid]
            db = it["input"]["database_schema"]["db_path"]
            pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
            patched_sql = patch_fn(pred_sql)
            if patched_sql is None:
                continue
            patched += 1
            new_rows = execute(db, patched_sql)
            if new_rows is None:
                continue
            gold = norm(execute(db, it["input"]["gold_sql"]))
            if norm(new_rows) == gold:
                fixed += 1
                fixed_qids.append(qid)
        yield_rate = fixed / cluster_n if cluster_n else 0
        pp = fixed / 1534 * 100
        print(f"{label:<20}{patch_name:<20}{cluster_n:>10}{patched:>10}{fixed:>8}"
              f"{yield_rate:>7.1%}{pp:>+7.2f}")
        if fixed_qids:
            print(f"  fixed qids: {fixed_qids[:15]}{'...' if len(fixed_qids) > 15 else ''}")


if __name__ == "__main__":
    main()
