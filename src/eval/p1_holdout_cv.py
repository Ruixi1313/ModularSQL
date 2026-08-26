#!/usr/bin/env python3
"""Leave-one-DB-out cross-validation of the dup_ratio threshold.

For each of the 9 DBs in the P1-fire set:
  - HELD-OUT: that DB's cases
  - TRAIN:    all other DBs' cases
  - Pick τ* that maximizes net (fixes - breaks) on TRAIN
  - Evaluate τ* on HELD-OUT, record (fixes, breaks)

Aggregate held-out fixes/breaks gives an unbiased net-delta estimate that
isn't overfit to specific DB patterns. This is the number to report to
reviewers as "out-of-distribution generalization."
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/p1_broken_features.csv"

CANDIDATE_TAUS = [round(t * 0.05, 2) for t in range(0, 21)]  # 0.00, 0.05, ..., 1.00


def pick_tau(samples):
    """Return τ maximizing net on `samples` (list of (label, dup_ratio))."""
    best_tau, best_net = 0.0, -1e9
    for tau in CANDIDATE_TAUS:
        fixes = sum(1 for lab, d in samples if lab == "fixed" and d >= tau)
        breaks = sum(1 for lab, d in samples if lab == "broken" and d >= tau)
        net = fixes - breaks
        if net > best_net:
            best_net, best_tau = net, tau
    return best_tau, best_net


def main():
    rows = list(csv.DictReader(FEATS.open()))
    samples = [(r["label"], float(r["dup_ratio"]), r["db_id"]) for r in rows]

    dbs = sorted({s[2] for s in samples})
    print(f"Loaded {len(samples)} P1-fire cases across {len(dbs)} DBs\n")

    held_fixes = held_breaks = 0
    held_fixes_baseline = held_breaks_baseline = 0  # τ=0 baseline (current P1)
    print(f"{'held-out DB':<24}{'n':>4}{'τ*':>6}{'train_net':>11}"
          f"{'held_fixes':>12}{'held_breaks':>12}{'held_net':>10}")
    print("-" * 80)

    for held in dbs:
        train = [(lab, d) for lab, d, db in samples if db != held]
        test = [(lab, d) for lab, d, db in samples if db == held]
        if not test:
            continue
        tau_star, train_net = pick_tau(train)
        tf = sum(1 for lab, d in test if lab == "fixed" and d >= tau_star)
        tb = sum(1 for lab, d in test if lab == "broken" and d >= tau_star)
        # baseline τ=0 (current P1 logic): all fire
        bf = sum(1 for lab, d in test if lab == "fixed")
        bb = sum(1 for lab, d in test if lab == "broken")
        held_fixes += tf
        held_breaks += tb
        held_fixes_baseline += bf
        held_breaks_baseline += bb
        print(f"  {held:<22}{len(test):>4}{tau_star:>6.2f}{train_net:>+11}"
              f"{tf:>12}{tb:>12}{tf - tb:>+10}")

    print("-" * 80)
    held_net = held_fixes - held_breaks
    base_net = held_fixes_baseline - held_breaks_baseline
    print(f"  {'AGGREGATE (held-out, refined)':<22}      "
          f"{'':>11}{held_fixes:>12}{held_breaks:>12}{held_net:>+10}")
    print(f"  {'AGGREGATE (baseline τ=0)':<22}      "
          f"{'':>11}{held_fixes_baseline:>12}{held_breaks_baseline:>12}{base_net:>+10}")
    print()
    print(f"Held-out improvement over current P1: {held_net - base_net:+d} samples")
    print(f"(This is the unbiased generalization estimate to report.)")


if __name__ == "__main__":
    main()
