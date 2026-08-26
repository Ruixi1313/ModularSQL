#!/usr/bin/env python3
"""
Extract qualitative case-study examples from a 4-config ablation CSV.

We want at least:
  * Example 1 — P1 saves a sample (P1 added DISTINCT and that flipped FAIL→PASS)
  * Example 2 — P2 saves a sample (profile hint helped LLM choose the right column or value)

Outputs:
  results/case_studies.md  (paper-ready, with question / SQLs side by side)
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


CMP_CSV = Path(__file__).resolve().parents[2] / "results/pattern2_4config_compare.csv"
OUT = Path(__file__).resolve().parents[2] / "results/case_studies.md"


def main():
    if not CMP_CSV.exists():
        print(f"comparison CSV not found: {CMP_CSV}")
        sys.exit(1)

    rows = list(csv.DictReader(open(CMP_CSV)))
    print(f"Loaded {len(rows)} rows from 4-config comparison\n")

    # P1 saves: B=FAIL, B+P1=PASS, B+P2=FAIL (P2 didn't fix it, only P1 did)
    p1_only_fix = [r for r in rows
                   if r["B"] == "FAIL" and r["B+P1"] == "PASS" and r["B+P2"] == "FAIL"]

    # P2 saves: B=FAIL, B+P2=PASS, B+P1=FAIL (P2 alone fixed it)
    p2_only_fix = [r for r in rows
                   if r["B"] == "FAIL" and r["B+P2"] == "PASS" and r["B+P1"] == "FAIL"]

    # Both pattern improve: B=FAIL, B+P1=PASS AND B+P2=PASS
    both_fix = [r for r in rows
                if r["B"] == "FAIL" and r["B+P1"] == "PASS" and r["B+P2"] == "PASS"]

    print(f"P1-only fixes: {len(p1_only_fix)}")
    print(f"P2-only fixes: {len(p2_only_fix)}")
    print(f"Both-fix:      {len(both_fix)}")

    sections = []

    def render_case(r, tag):
        return (
            f"### Case {tag} — [{r['db_id']}/{r['qid']}] ({r['difficulty']})\n\n"
            f"**Question:** _(loaded from snapshot)_\n\n"
            f"**Baseline SQL (FAIL):**\n```sql\n{r['b_sql']}\n```\n\n"
            f"**ModularSQL output (PASS):**\n```sql\n"
            f"{r.get('p2_sql') if r['B+P2'] == 'PASS' else r['b_sql']}"
            f"```\n\n"
            f"**Gold SQL:**\n```sql\n{r['gold_sql']}\n```\n"
        )

    md = ["# ModularSQL — Qualitative Case Studies\n",
          "Drawn from the 99-sample BIRD-Dev ablation. Each case is one row in `results/pattern2_4config_compare.csv`.\n"]

    if p1_only_fix:
        md.append("\n## Pattern 1 wins (DISTINCT verifier alone)\n")
        for i, r in enumerate(p1_only_fix[:3], 1):
            md.append(render_case(r, f"P1-{i}"))

    if p2_only_fix:
        md.append("\n## Pattern 2 wins (profile-augmented schema)\n")
        for i, r in enumerate(p2_only_fix[:3], 1):
            md.append(render_case(r, f"P2-{i}"))

    if both_fix:
        md.append("\n## Both patterns recover the same case\n")
        for i, r in enumerate(both_fix[:2], 1):
            md.append(render_case(r, f"BOTH-{i}"))

    if not p1_only_fix and not p2_only_fix:
        md.append("\n_No qualifying cases — neither pattern uniquely flipped any failures._\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print(f"\n📄 Wrote {OUT}")


if __name__ == "__main__":
    main()
