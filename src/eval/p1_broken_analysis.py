#!/usr/bin/env python3
"""Characterize the 41 P1 broken cases (baseline PASS -> P1 FAIL).

For each broken sample, extract features that might separate them from the 46
P1-fixed cases. The goal: find a refined guard that keeps fixes but rejects
breakages.

Features per sample:
  - n_baseline_rows, n_gold_rows, n_p1_rows
  - duplication_ratio = (n_baseline - n_unique) / n_baseline
  - null_density of baseline rows
  - per-column null fraction (which projected columns contain NULLs)
  - n_joins (count of JOIN keywords)
  - n_select_cols (rough — count commas in SELECT projection)
  - gold_has_distinct (boolean)
  - gold_has_groupby (boolean)
  - intersect_ratio: |gold ∩ p1| / |gold| (how much of gold survived DISTINCT)
  - row count parity: n_gold vs n_p1 (does gold have MORE rows than DISTINCT'd?)
"""
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary.csv"
S5 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"

JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
DISTINCT_RE = re.compile(r"\bSELECT\s+DISTINCT\b", re.IGNORECASE)
GROUPBY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
SELECT_PROJ_RE = re.compile(r"^\s*SELECT\s+(.+?)\s+FROM\b", re.IGNORECASE | re.DOTALL)


def execute(db, sql):
    try:
        c = sqlite3.connect(db, timeout=8)
        cur = c.cursor()
        cur.execute(sql)
        r = [tuple(x) for x in cur.fetchall()]
        c.close()
        return r
    except Exception:
        return None


def null_frac(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if any(v is None for v in r)) / len(rows)


def col_null_frac(rows):
    """Per-column null fraction. Returns list aligned to projection."""
    if not rows:
        return []
    n_cols = len(rows[0])
    return [sum(1 for r in rows if r[i] is None) / len(rows) for i in range(n_cols)]


def proj_col_count(sql):
    m = SELECT_PROJ_RE.match(sql)
    if not m:
        return -1
    proj = m.group(1)
    # rough — split on commas not inside parens
    depth = 0
    cols = 1
    for ch in proj:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cols += 1
    return cols


def load_broken_qids():
    qids = []
    with SUMMARY.open() as f:
        for row in csv.DictReader(f):
            if row["p1_fired"] == "YES" and row["baseline_match"] == "PASS" and row["p1_match"] == "FAIL":
                qids.append(int(row["qid"]))
    return qids


def load_fixed_qids():
    qids = []
    with SUMMARY.open() as f:
        for row in csv.DictReader(f):
            if row["p1_fired"] == "YES" and row["baseline_match"] == "FAIL" and row["p1_match"] == "PASS":
                qids.append(int(row["qid"]))
    return qids


def load_items(target_qids):
    target = set(target_qids)
    items = {}
    with S5.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid in target:
                items[qid] = it
    return items


def featurize(it):
    db = it["input"]["database_schema"]["db_path"]
    gold_sql = it["input"]["gold_sql"]
    pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]

    baseline_rows = execute(db, pred_sql) or []
    gold_rows = execute(db, gold_sql) or []

    n_b = len(baseline_rows)
    n_g = len(gold_rows)
    n_unique = len(set(baseline_rows))
    dup_ratio = (n_b - n_unique) / n_b if n_b else 0.0

    g_set = set(gold_rows)
    p1_unique = set(baseline_rows)  # DISTINCT'd baseline ≈ p1 rows
    intersect = len(g_set & p1_unique)

    feats = {
        "qid": it["input"]["question_id"],
        "db_id": it["input"]["database_id"],
        "difficulty": it["input"].get("difficulty"),
        "n_baseline": n_b,
        "n_gold": n_g,
        "n_unique_baseline": n_unique,
        "dup_ratio": round(dup_ratio, 3),
        "null_density": round(null_frac(baseline_rows), 3),
        "col_null_fracs": [round(x, 2) for x in col_null_frac(baseline_rows)],
        "max_col_null": round(max(col_null_frac(baseline_rows), default=0.0), 2),
        "n_joins": len(JOIN_RE.findall(pred_sql)),
        "n_proj_cols": proj_col_count(pred_sql),
        "gold_has_distinct": bool(DISTINCT_RE.search(gold_sql)),
        "gold_has_groupby": bool(GROUPBY_RE.search(gold_sql)),
        "gold_ge_p1": n_g >= n_unique,  # gold has at least as many rows as DISTINCT'd → distinct WRONG direction
        "intersect_over_gold": round(intersect / n_g, 3) if n_g else 0.0,
    }
    return feats


def summarize(label, feats_list):
    print(f"\n{'='*80}\n{label} (n={len(feats_list)})\n{'='*80}")
    if not feats_list:
        return

    keys_num = ["n_baseline", "n_gold", "n_unique_baseline", "dup_ratio",
                "null_density", "max_col_null", "n_joins", "n_proj_cols",
                "intersect_over_gold"]
    keys_bool = ["gold_has_distinct", "gold_has_groupby", "gold_ge_p1"]

    print(f"  {'feature':<22}{'mean':>10}{'median':>10}{'min':>10}{'max':>10}")
    for k in keys_num:
        vals = [f[k] for f in feats_list if f[k] is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        med = vals_sorted[len(vals) // 2]
        print(f"  {k:<22}{sum(vals)/len(vals):>10.2f}{med:>10.2f}{min(vals):>10.2f}{max(vals):>10.2f}")
    print()
    for k in keys_bool:
        n_true = sum(1 for f in feats_list if f[k])
        print(f"  {k:<22}{n_true:>4}/{len(feats_list)}  ({n_true/len(feats_list):.0%})")

    db_counter = Counter(f["db_id"] for f in feats_list)
    print(f"\n  by db: {dict(db_counter.most_common())}")
    diff_counter = Counter(f["difficulty"] for f in feats_list)
    print(f"  by difficulty: {dict(diff_counter)}")


def main():
    broken_qids = load_broken_qids()
    fixed_qids = load_fixed_qids()
    print(f"Loaded {len(broken_qids)} broken qids, {len(fixed_qids)} fixed qids")

    all_qids = list(set(broken_qids) | set(fixed_qids))
    items = load_items(all_qids)
    print(f"Featurizing {len(items)} items...\n")

    broken_feats = [featurize(items[q]) for q in broken_qids if q in items]
    fixed_feats = [featurize(items[q]) for q in fixed_qids if q in items]

    summarize("BROKEN (P1 mis-fired, PASS→FAIL)", broken_feats)
    summarize("FIXED  (P1 helped,   FAIL→PASS)", fixed_feats)

    # Differential features — look for features where broken vs fixed diverge
    print(f"\n{'='*80}\nDISCRIMINATIVE SIGNALS (broken_mean - fixed_mean)\n{'='*80}")
    for k in ["n_baseline", "n_gold", "n_unique_baseline", "dup_ratio",
              "null_density", "max_col_null", "n_joins", "n_proj_cols",
              "intersect_over_gold"]:
        b = sum(f[k] for f in broken_feats) / len(broken_feats)
        x = sum(f[k] for f in fixed_feats) / len(fixed_feats)
        print(f"  {k:<22}  broken={b:7.2f}  fixed={x:7.2f}  Δ={b-x:+7.2f}")
    for k in ["gold_has_distinct", "gold_has_groupby", "gold_ge_p1"]:
        b = sum(1 for f in broken_feats if f[k]) / len(broken_feats)
        x = sum(1 for f in fixed_feats if f[k]) / len(fixed_feats)
        print(f"  {k:<22}  broken={b:7.2%}  fixed={x:7.2%}  Δ={b-x:+7.2%}")

    # Dump per-sample for spot-check
    out = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/p1_broken_features.csv"
    with out.open("w", newline="") as f:
        flat = []
        for src, lbl in [(broken_feats, "broken"), (fixed_feats, "fixed")]:
            for d in src:
                d2 = {**d, "label": lbl, "col_null_fracs": str(d["col_null_fracs"])}
                flat.append(d2)
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"\nPer-sample features → {out}")


if __name__ == "__main__":
    main()
