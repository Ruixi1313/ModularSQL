#!/usr/bin/env python3
"""
Compute token usage and dollar cost from a DeepEye-SQL pipeline snapshot.

Each item's `pipeline_artifacts.metrics.total_llm_cost` has:
    {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}

For the Qwen3-Coder-30B-A3B-Instruct model on OpenRouter:
    input  = $0.07 / M tokens
    output = $0.27 / M tokens

Usage:
    python3 src/eval/compute_cost.py <path/to/snapshot/items.jsonl> [--label NAME]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PRICE_IN_PER_M = 0.07
PRICE_OUT_PER_M = 0.27

COST_LOG = Path(__file__).resolve().parents[2] / "results/cost_log.csv"


def compute_cost(snapshot_path: Path) -> dict:
    items = [json.loads(l) for l in open(snapshot_path)]
    in_tokens = out_tokens = 0
    per_db = defaultdict(lambda: [0, 0])  # db_id → [input, output]
    for it in items:
        m = it.get("pipeline_artifacts", {}).get("metrics", {}).get("total_llm_cost", {})
        pt = m.get("prompt_tokens", 0)
        ct = m.get("completion_tokens", 0)
        in_tokens += pt
        out_tokens += ct
        db = it.get("input", {}).get("database_id", "")
        per_db[db][0] += pt
        per_db[db][1] += ct

    in_dollars = in_tokens * PRICE_IN_PER_M / 1_000_000
    out_dollars = out_tokens * PRICE_OUT_PER_M / 1_000_000

    return {
        "n_items": len(items),
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "input_cost": round(in_dollars, 4),
        "output_cost": round(out_dollars, 4),
        "total_cost": round(in_dollars + out_dollars, 4),
        "per_db": dict(per_db),
    }


def log_cost(label: str, snapshot_path: Path, cost: dict) -> None:
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not COST_LOG.exists()
    with COST_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "label", "snapshot", "n_items",
                        "prompt_tokens", "completion_tokens", "total_tokens",
                        "input_cost_usd", "output_cost_usd", "total_cost_usd"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            label, str(snapshot_path), cost["n_items"],
            cost["prompt_tokens"], cost["completion_tokens"], cost["total_tokens"],
            cost["input_cost"], cost["output_cost"], cost["total_cost"],
        ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", help="Path to sql_selection items.jsonl")
    p.add_argument("--label", default="unlabeled", help="Run label (e.g., ModularSQL_Baseline_dev1534)")
    p.add_argument("--log", action="store_true", help="Append to results/cost_log.csv")
    args = p.parse_args()

    snap = Path(args.snapshot)
    if not snap.exists():
        print(f"Not found: {snap}", file=sys.stderr)
        sys.exit(1)

    cost = compute_cost(snap)
    print(f"=== Cost report: {args.label} ===")
    print(f"  Items:               {cost['n_items']}")
    print(f"  Prompt tokens:       {cost['prompt_tokens']:>12,}")
    print(f"  Completion tokens:   {cost['completion_tokens']:>12,}")
    print(f"  Total tokens:        {cost['total_tokens']:>12,}")
    print(f"  Input cost  (USD):   ${cost['input_cost']:.4f}")
    print(f"  Output cost (USD):   ${cost['output_cost']:.4f}")
    print(f"  TOTAL cost  (USD):   ${cost['total_cost']:.4f}")
    print(f"  Per-item avg:        ${cost['total_cost']/cost['n_items']:.4f}")

    if args.log:
        log_cost(args.label, snap, cost)
        print(f"\n  Appended to {COST_LOG}")


if __name__ == "__main__":
    main()
