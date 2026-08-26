#!/usr/bin/env python3
"""Sweep dup_ratio threshold on the 87 P1-fire cases to find optimal guard.

For each candidate threshold τ ∈ [0.0, 1.0]:
  - 'fire' = P1 would inject DISTINCT only when dup_ratio >= τ
  - count: fixes_kept, breaks_avoided, fixes_lost, breaks_kept
  - net = fixes_kept - breaks_kept
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/p1_broken_features.csv"


def main():
    rows = list(csv.DictReader(FEATS.open()))
    for r in rows:
        r["dup_ratio"] = float(r["dup_ratio"])
        r["intersect_over_gold"] = float(r["intersect_over_gold"])
        r["n_baseline"] = int(r["n_baseline"])

    fixed = [r for r in rows if r["label"] == "fixed"]
    broken = [r for r in rows if r["label"] == "broken"]
    print(f"Loaded {len(fixed)} fixed + {len(broken)} broken cases\n")

    print(f"{'τ (dup_ratio)':<16}{'fixes_kept':>12}{'breaks_kept':>14}"
          f"{'fixes_lost':>12}{'breaks_avoided':>16}{'net':>10}")
    print("-" * 80)
    baseline_net = len(fixed) - len(broken)
    print(f"{'(current 0.0)':<16}{len(fixed):>12}{len(broken):>14}"
          f"{0:>12}{0:>16}{baseline_net:>+10}")

    best = (None, -999, None)
    for tau in [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65,
                0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        fixes_kept = sum(1 for r in fixed if r["dup_ratio"] >= tau)
        fixes_lost = len(fixed) - fixes_kept
        breaks_kept = sum(1 for r in broken if r["dup_ratio"] >= tau)
        breaks_avoided = len(broken) - breaks_kept
        net = fixes_kept - breaks_kept
        marker = "  ←" if net > best[1] else ""
        if net > best[1]:
            best = (tau, net, (fixes_kept, breaks_kept, fixes_lost, breaks_avoided))
        print(f"{tau:<16.2f}{fixes_kept:>12}{breaks_kept:>14}"
              f"{fixes_lost:>12}{breaks_avoided:>16}{net:>+10}{marker}")

    print(f"\nBest threshold: τ={best[0]}, net=+{best[1]}")
    fk, bk, fl, ba = best[2]
    print(f"  fixes_kept={fk}, breaks_kept={bk}, fixes_lost={fl}, breaks_avoided={ba}")
    print(f"  vs current net=+5 → improvement = +{best[1] - 5}")

    print("\nJoint rule: dup_ratio >= τ AND n_baseline >= N_min")
    for tau in [0.55, 0.65, 0.70, 0.75]:
        for nmin in [5, 10, 20, 50]:
            fk = sum(1 for r in fixed if r["dup_ratio"] >= tau and r["n_baseline"] >= nmin)
            bk = sum(1 for r in broken if r["dup_ratio"] >= tau and r["n_baseline"] >= nmin)
            net = fk - bk
            print(f"  τ={tau} N_min={nmin:>3}  fixes_kept={fk:>3} breaks_kept={bk:>3} net={net:+d}")


if __name__ == "__main__":
    main()
