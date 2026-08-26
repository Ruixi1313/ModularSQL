#!/usr/bin/env python3
"""Cost comparison: ModularSQL (Qwen3-Coder) vs hypothetical GPT-4o / others.

Reads our actual token usage from cost_log.csv and re-prices it against
several backbones, assuming identical token counts (which is a STRONG
assumption — different models would produce different completion lengths
in practice). We document this caveat in the output.
"""
import csv
from pathlib import Path

COST_LOG = Path(__file__).resolve().parents[2] / "results/cost_log.csv"

# USD per 1M tokens. Sources noted; update if pricing changes.
PRICING = {
    "Qwen3-Coder-30B (OpenRouter, actual)": (0.07, 0.27),
    "GPT-4o-mini":                          (0.15, 0.60),
    "GPT-4o":                               (2.50, 10.00),
    "Claude Sonnet 4.6":                    (3.00, 15.00),
    "Claude Opus 4.7":                      (15.00, 75.00),
}


def reprice(prompt_tok, completion_tok, in_rate, out_rate):
    return prompt_tok / 1e6 * in_rate, completion_tok / 1e6 * out_rate


def main():
    runs = list(csv.DictReader(COST_LOG.open()))
    print("=" * 78)
    print("ModularSQL Cost Re-pricing Across Backbones")
    print("=" * 78)
    print("Assumption: identical token counts across backbones (rough estimate only).")
    print()

    for r in runs:
        label = r["label"]
        n = int(r["n_items"])
        p_tok = int(r["prompt_tokens"])
        c_tok = int(r["completion_tokens"])
        actual_total = float(r["total_cost_usd"])

        print(f"--- {label}  (n={n}, prompt={p_tok/1e6:.1f}M, completion={c_tok/1e6:.1f}M) ---")
        print(f"  {'Backbone':<42}{'In':>10}{'Out':>10}{'Total':>10}{'vs Ours':>10}")
        ours_total = None
        for name, (in_r, out_r) in PRICING.items():
            in_c, out_c = reprice(p_tok, c_tok, in_r, out_r)
            tot = in_c + out_c
            if name.startswith("Qwen"):
                ours_total = tot
                ratio = "1.0×"
            else:
                ratio = f"{tot / ours_total:.1f}×" if ours_total else "—"
            print(f"  {name:<42}{in_c:>10.2f}{out_c:>10.2f}{tot:>10.2f}{ratio:>10}")
        print()

    # Extrapolate to BIRD test set (~1789 questions) using dev1534 per-item rate
    print("=" * 78)
    print("Extrapolation to BIRD private test set (n≈1789)")
    print("=" * 78)
    dev1534 = next(r for r in runs if "dev1534" in r["label"])
    per_item_in = int(dev1534["prompt_tokens"]) / int(dev1534["n_items"])
    per_item_out = int(dev1534["completion_tokens"]) / int(dev1534["n_items"])
    test_in_tok = per_item_in * 1789
    test_out_tok = per_item_out * 1789
    print(f"  Projected tokens: {test_in_tok/1e6:.1f}M in + {test_out_tok/1e6:.1f}M out")
    print(f"  {'Backbone':<42}{'Projected total':>20}")
    for name, (in_r, out_r) in PRICING.items():
        in_c, out_c = reprice(test_in_tok, test_out_tok, in_r, out_r)
        print(f"  {name:<42}{'$' + f'{in_c+out_c:,.2f}':>20}")


if __name__ == "__main__":
    main()
