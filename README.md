# ModularSQL

A post-selection runtime guardrail for the **Multiplicity Blind Spot** in Text-to-SQL.
Companion code for the paper (arXiv preprint).

ModularSQL probes the executed result of each selected SQL query for multiplicity
anomalies and applies targeted interventions (deterministic patches + low-cost LLM
rescue) only on flagged queries, leaving unaffected queries unchanged.

## Canonical Reproduction Scripts

All numbers in the paper come from the following scripts, run on the executable
BIRD-Dev subset (`N=1,532`) of the canonical S7 baseline
(`workspace_full/sql_selection.topk3_backup/`).

| Paper element | Script |
|---|---|
| Table 1 (Cross-Backbone Evidence of MBS)               | `src/eval/cross_backbone_eval.py` |
| Table 2 (Cross-Pipeline Transfer)                      | `src/eval/cross_pipeline_guardrail.py` |
| Table 3 (Main Results + Ablation)                      | `src/eval/unified_eval_v2.py` (canonical v2); `src/eval/unified_eval.py` (v1 row) |
| Table 4 (By-difficulty gain)                           | `src/eval/unified_eval_v2.py` (stdout breakdown) |
| S4.5 Trigger reasons / Cost / Rescue effect            | `src/eval/evaluate_p63.py` |
| S4.6 Detector Diagnostics / Latency                   | `src/eval/detector_diagnostics.py`; `src/eval/llm_rescue_latency.py` |
| Appendix A (P1 Threshold Sensitivity, Table 5)         | `src/eval/p1_threshold_sweep.py` (in-sample); `src/eval/p1_holdout_cv.py` (LOO-DB CV) |
| Appendix B (P2 Threshold Sensitivity, Table 6)         | `src/eval/evaluate_p3.py` |
| Appendix C (LLM Rescue Prompt template)                | `src/improvements/llm_rescue_selector.py` |
| Appendix D (P2 Implementation Safeguards)              | `src/improvements/distinct_remover.py` |
| Appendix E (Intervention-Order Ablation v1 vs v2)      | `src/eval/unified_eval.py` (v1) + `src/eval/unified_eval_v2.py` (v2) |
| Appendix F (By-Difficulty Per-Stage Breakdown, Table 8) | `src/eval/unified_eval_v2.py` |
| Appendix G (Multiset Equality Sanity Check)            | `src/eval/counter_sanity_check.py` |
| Appendix H (Cross-Pipeline Details, Tables 9--10)      | `src/eval/cross_pipeline_guardrail.py` |
| Appendix I (Detector Diagnostics, Tables 11--12)       | `src/eval/detector_diagnostics.py` |

### Upstream data dependencies

A few scripts produce intermediate CSVs that the reproduction chain depends on:

```
build_full_summary.py
  → results/.../summary.csv
    → p1_broken_analysis.py
      → results/.../p1_broken_features.csv
        → p1_threshold_sweep.py  (Appendix A in-sample)
        → p1_holdout_cv.py       (Appendix A LOO-DB CV)

evaluate_p62.py
  → results/.../summary_p62.csv
    → evaluate_p63.py
      → results/.../summary_p63.csv
        → unified_eval.py     (Table 2 v1 row)
        → unified_eval_v2.py  (Table 2 v2 row, Tables 3 & 7)
```

Run order to reproduce from scratch:

```bash
python src/eval/build_full_summary.py
python src/eval/p1_broken_analysis.py
python src/eval/evaluate_p62.py
python src/eval/evaluate_p63.py
# Then any of the table scripts above.
```

Other evaluation scripts are exploratory, debugging, or intermediate
development artifacts and are not required to reproduce the paper.

## Paper-to-Codebase Naming

| Paper name | Codebase file | Notes |
|---|---|---|
| **P1** (DISTINCT injection)  | `src/improvements/distinct_verifier.py`     | `τ = 0.80` |
| **P2** (DISTINCT removal)    | `src/improvements/distinct_remover.py`      | `τ = 0.10` |
| **LLM rescue**               | `src/improvements/llm_rescue_selector.py`   | Judge-style selection over S6 candidates |

(Older P-numbering — P3 / P6.2 / P6.3 — appears in some codebase files. The
mapping above is canonical for the paper.)

## Headline Numbers (BIRD-Dev, N=1,532)

| System | Set-EX | Multiset-EX | Δ Multiset-EX |
|---|---|---|---|
| DeepEye reproduction (Qwen3-Coder-30B-A3B) | 72.06% | 65.86% | — |
| + ModularSQL v2 (Rescue → P1+P2)           | 72.06% | 67.75% | **+1.89 pp** |

- 77 high-risk anomalies flagged (5.0% of workload)
- 30.5% of the reproduced MBS gap closed
- $0.0076 in total LLM cost (under $0.0001 per affected query)

## Setup

### External Dependencies

Two external resources are not vendored in this repo and must be obtained
separately:

1. **DeepEye-SQL** (the upstream pipeline). Clone into `external/`:
   ```bash
   mkdir -p external && cd external
   git clone https://github.com/HKUSTDial/DeepEye-SQL.git
   cd ..
   ```

2. **BIRD-Dev**. Download from
   [bird-bench.github.io](https://bird-bench.github.io/) and place under
   `data/bird/dev/`:
   ```
   data/bird/dev/dev.json
   data/bird/dev/dev_tables.json
   data/bird/dev/dev_databases/
   ```

### Installation

DeepEye-SQL requires Python 3.12+ and `uv`. The setup script installs the
upstream dependencies and resolves config templates:

```bash
cd path/to/ModularSQL
cp .env.template .env          # then add OPENROUTER_API_KEY
./setup.sh
```

### Canonical S7 Baseline

All paper numbers come from the canonical S7 baseline at
`external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/`. This is
generated by running the full DeepEye-SQL pipeline on BIRD-Dev with
Qwen3-Coder-30B-A3B. Follow DeepEye-SQL's own documentation to generate the
workspace artifacts, then run the "Run order to reproduce from scratch"
sequence above to produce ModularSQL's CSVs.

## Project Layout

```
ModularSQL/
├── external/DeepEye-SQL/         # upstream pipeline (vendor-pinned)
├── src/
│   ├── improvements/             # ModularSQL plugins (P1, P2, LLM rescue)
│   ├── eval/                     # evaluation + ablation scripts
│   └── adapters/                 # config helpers
├── experiments/configs/          # per-experiment TOML configs
├── results/                      # outputs of eval scripts (gitignored)
├── data/                         # BIRD dataset (gitignored)
└── docs/
```
