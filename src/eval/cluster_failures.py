#!/usr/bin/env python3
"""Cluster the 502 P1-v2 failures by structural diff between gold and pred SQL.

Each failure gets a multi-label tag based on which SQL features appear in gold
but are missing/wrong in pred. Output:
  - top-N clusters ranked by count
  - per-cluster example qids for manual triage
  - per-cluster cost-benefit ratio (count × difficulty weight)
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / ""
SUMMARY = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/summary_refined.csv"
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection/bird/dev.snapshot.data/items.jsonl"
OUT_CSV = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/failure_clusters.csv"


FEATURES = {
    "distinct":  re.compile(r"\bDISTINCT\b", re.IGNORECASE),
    "groupby":   re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
    "having":    re.compile(r"\bHAVING\b", re.IGNORECASE),
    "case":      re.compile(r"\bCASE\s+WHEN\b", re.IGNORECASE),
    "subquery":  re.compile(r"\(\s*SELECT\b", re.IGNORECASE),  # nested SELECT
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


def featurize(sql):
    if not sql:
        return {f: False for f in FEATURES}, 0
    feats = {name: bool(rx.search(sql)) for name, rx in FEATURES.items()}
    n_joins = len(JOIN_RE.findall(sql))
    return feats, n_joins


def label_diff(gold_feats, pred_feats, gold_joins, pred_joins):
    """Multi-label categorization of where pred diverges from gold."""
    labels = []
    for k in FEATURES:
        if gold_feats[k] and not pred_feats[k]:
            labels.append(f"missing_{k}")
        elif pred_feats[k] and not gold_feats[k]:
            labels.append(f"extra_{k}")
    if gold_joins > pred_joins:
        labels.append(f"missing_joins_+{gold_joins - pred_joins}")
    elif pred_joins > gold_joins:
        labels.append(f"extra_joins_+{pred_joins - gold_joins}")
    return labels


def main():
    fail_qids = set()
    diff_lookup = {}
    with SUMMARY.open() as f:
        for r in csv.DictReader(f):
            if r["p1_match_tau0.80"] == "FAIL":
                fail_qids.add(int(r["qid"]))
                diff_lookup[int(r["qid"])] = r["difficulty"]
    print(f"Loaded {len(fail_qids)} failures (after P1 v2 τ=0.80)\n")

    rows = []
    label_counts = Counter()
    label_qids = defaultdict(list)
    label_by_diff = defaultdict(lambda: Counter())

    with S7.open() as f:
        for line in f:
            it = json.loads(line)
            qid = it["input"]["question_id"]
            if qid not in fail_qids:
                continue
            gold_sql = it["input"]["gold_sql"]
            pred_sql = it["pipeline_artifacts"]["sql_selection"]["final_selected_sql"]
            db_id = it["input"]["database_id"]
            diff = it["input"].get("difficulty", "?")

            g_feats, g_joins = featurize(gold_sql)
            p_feats, p_joins = featurize(pred_sql)
            labels = label_diff(g_feats, p_feats, g_joins, p_joins)

            if not labels:
                labels = ["structurally_identical"]  # same skeleton, different details
            for lab in labels:
                label_counts[lab] += 1
                if len(label_qids[lab]) < 5:
                    label_qids[lab].append(qid)
                label_by_diff[lab][diff] += 1

            rows.append({
                "qid": qid, "db_id": db_id, "difficulty": diff,
                "labels": ";".join(labels),
                "gold_sql": gold_sql[:300],
                "pred_sql": pred_sql[:300],
            })

    print(f"Featurized {len(rows)} failure rows\n")
    print("=" * 80)
    print(f"Top failure clusters (label → count, % of 502 failures)")
    print("=" * 80)
    print(f"  {'label':<35}{'count':>8}{'pct':>8}    examples")
    for label, cnt in label_counts.most_common(25):
        pct = cnt / len(fail_qids)
        ex = ", ".join(str(q) for q in label_qids[label][:5])
        print(f"  {label:<35}{cnt:>8}{pct:>7.1%}    [{ex}]")

    print()
    print("=" * 80)
    print("Top 10 clusters × difficulty breakdown")
    print("=" * 80)
    print(f"  {'label':<35}{'simple':>8}{'moderate':>10}{'challenging':>13}")
    for label, _ in label_counts.most_common(10):
        d = label_by_diff[label]
        print(f"  {label:<35}{d.get('simple', 0):>8}{d.get('moderate', 0):>10}{d.get('challenging', 0):>13}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Per-failure labels → {OUT_CSV}")


if __name__ == "__main__":
    main()
