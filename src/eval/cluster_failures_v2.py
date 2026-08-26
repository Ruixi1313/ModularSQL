#!/usr/bin/env python3
"""Cluster failures under either Multiset-EX (cardinality-aware) or
Set-EX (BIRD official) evaluation.

Replaces cluster_failures.py. Reads pred SQL directly from S7 items.jsonl,
executes against DB, compares to gold under chosen eval method, then
clusters failures by structural diff.

Usage:
  python3 src/eval/cluster_failures_v2.py --eval multiset
  python3 src/eval/cluster_failures_v2.py --eval set
"""
import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
OUT_DIR = ROOT / "results/ModularSQL_Baseline_dev1534_20260514"


FEATURES = {
    "distinct":  re.compile(r"\bDISTINCT\b", re.IGNORECASE),
    "groupby":   re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
    "having":    re.compile(r"\bHAVING\b", re.IGNORECASE),
    "case":      re.compile(r"\bCASE\s+WHEN\b", re.IGNORECASE),
    "subquery":  re.compile(r"\(\s*SELECT\b", re.IGNORECASE),
    "orderby":   re.compile(r"\bORDER\s+BY\b", re.IGNORECASE),
    "limit":     re.compile(r"\bLIMIT\b", re.IGNORECASE),
    "count":     re.compile(r"\bCOUNT\s*\(", re.IGNORECASE),
    "sum":       re.compile(r"\bSUM\s*\(", re.IGNORECASE),
    "avg":       re.compile(r"\bAVG\s*\(", re.IGNORECASE),
    "min":       re.compile(r"\bMIN\s*\(", re.IGNORECASE),
    "max":       re.compile(r"\bMAX\s*\(", re.IGNORECASE),
    "like":      re.compile(r"\bLIKE\b", re.IGNORECASE),
    "in_clause": re.compile(r"\bIN\s*\(", re.IGNORECASE),
    "between":   re.compile(r"\bBETWEEN\b", re.IGNORECASE),
    "not":       re.compile(r"\bNOT\b", re.IGNORECASE),
    "cte":       re.compile(r"\bWITH\b.*?\bAS\s*\(", re.IGNORECASE | re.DOTALL),
    "union":     re.compile(r"\bUNION\b", re.IGNORECASE),
    "cast":      re.compile(r"\bCAST\s*\(", re.IGNORECASE),
    "strftime":  re.compile(r"\bstrftime\b", re.IGNORECASE),
}
JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)


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


def match_multiset(pred, gold):
    """Multiset-EX (cardinality-aware): preserves row multiplicities."""
    if gold is None: return None
    if pred is None: return False
    return sorted(pred, key=lambda r: str(r)) == sorted(gold, key=lambda r: str(r))


def match_set(pred, gold):
    if gold is None: return None
    if pred is None: return False
    return set(pred) == set(gold)


def featurize(sql):
    if not sql:
        return {f: False for f in FEATURES}, 0
    feats = {name: bool(rx.search(sql)) for name, rx in FEATURES.items()}
    return feats, len(JOIN_RE.findall(sql))


def label_diff(g_feats, p_feats, g_joins, p_joins):
    labels = []
    for k in FEATURES:
        if g_feats[k] and not p_feats[k]:
            labels.append(f"missing_{k}")
        elif p_feats[k] and not g_feats[k]:
            labels.append(f"extra_{k}")
    if g_joins > p_joins:
        labels.append(f"missing_joins_+{g_joins - p_joins}")
    elif p_joins > g_joins:
        labels.append(f"extra_joins_+{p_joins - g_joins}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["multiset", "set"], required=True)
    ap.add_argument("--ours", default=str(DEFAULT_S7))
    args = ap.parse_args()

    match_fn = match_multiset if args.eval == "multiset" else match_set
    out_csv = OUT_DIR / f"failure_clusters_{args.eval}.csv"
    print(f"Eval method: {args.eval}\nInput: {args.ours}\nOutput: {out_csv}\n")

    items = [json.loads(l) for l in open(args.ours)]
    failures = []
    n_eval = 0
    for i, it in enumerate(items, 1):
        db = it["input"]["database_schema"]["db_path"]
        gold_sql = it["input"]["gold_sql"]
        pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
        gold_rows = execute(db, gold_sql)
        pred_rows = execute(db, pred_sql)
        if gold_rows is None:
            continue
        n_eval += 1
        if match_fn(pred_rows, gold_rows):
            continue
        failures.append({
            "qid": it["input"]["question_id"],
            "db_id": it["input"]["database_id"],
            "difficulty": it["input"].get("difficulty"),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
        })
        if i % 200 == 0:
            print(f"  ...{i}/{len(items)}  failures so far: {len(failures)}", flush=True)

    print(f"\nTotal evaluated: {n_eval}, failures: {len(failures)} ({len(failures)/n_eval*100:.2f}%)\n")

    label_counts = Counter()
    label_qids = defaultdict(list)
    label_by_diff = defaultdict(Counter)
    rows = []
    for f in failures:
        g_feats, g_j = featurize(f["gold_sql"])
        p_feats, p_j = featurize(f["pred_sql"])
        labels = label_diff(g_feats, p_feats, g_j, p_j) or ["structurally_identical"]
        for lab in labels:
            label_counts[lab] += 1
            if len(label_qids[lab]) < 5:
                label_qids[lab].append(f["qid"])
            label_by_diff[lab][f["difficulty"]] += 1
        rows.append({**f, "labels": ";".join(labels),
                     "gold_sql": f["gold_sql"][:300], "pred_sql": f["pred_sql"][:300]})

    print("=" * 80)
    print(f"Top failure clusters under {args.eval} eval (n_failures={len(failures)})")
    print("=" * 80)
    print(f"  {'label':<35}{'count':>8}{'pct':>8}    example qids")
    for label, cnt in label_counts.most_common(20):
        pct = cnt / len(failures)
        ex = ", ".join(str(q) for q in label_qids[label][:5])
        print(f"  {label:<35}{cnt:>8}{pct:>7.1%}    [{ex}]")

    print("\n" + "=" * 80)
    print("Top 10 clusters × difficulty")
    print("=" * 80)
    print(f"  {'label':<35}{'simple':>8}{'moderate':>10}{'challenging':>13}")
    for label, _ in label_counts.most_common(10):
        d = label_by_diff[label]
        print(f"  {label:<35}{d.get('simple', 0):>8}{d.get('moderate', 0):>10}{d.get('challenging', 0):>13}")

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Per-failure data → {out_csv}")


if __name__ == "__main__":
    main()
