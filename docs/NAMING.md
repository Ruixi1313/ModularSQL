# ModularSQL Naming Conventions

Consistent naming across all scripts, results, and paper artifacts.

## Configuration Names

| Name | Meaning | Pipeline behavior |
|---|---|---|
| **ModularSQL_Baseline** | DeepEye-SQL reproduced (no improvements) | Standard pipeline + BIRD's own descriptions only |
| **ModularSQL_P1** | Baseline + Pattern 1 only | Standard pipeline output → post-hoc DISTINCT verifier |
| **ModularSQL_P2** | Baseline + Pattern 2 (slim) only | Pipeline run with profile-augmented schema (slim hints: PK / prefix / NULL only) |
| **ModularSQL_Full** | Baseline + Pattern 1 + Pattern 2 | P2 pipeline output → post-hoc DISTINCT verifier |

## Sample-set Names

| Name | Definition |
|---|---|
| **dev99** | BIRD-Dev `max_samples_per_db = 9` (11 DBs × 9 = 99 samples) — fast iteration set |
| **holdout100** | BIRD-Dev questions 9–17 of each DB + 1 extra (100 samples) — generalization probe |
| **dev1534** | Full BIRD-Dev (1,534 questions) — final paper number |

## Result Directory Convention

```
results/<config>_<sample-set>_<date>/
```

Examples:
- `results/ModularSQL_Baseline_dev99_20260513/`
- `results/ModularSQL_P1_dev99_20260513/`
- `results/ModularSQL_P2slim_dev99_20260513/`
- `results/ModularSQL_Full_dev1534_20260601/`

Each result directory contains:
- `README.md` — experiment metadata + headline numbers
- `summary.csv` — per-sample {qid, db, difficulty, prediction, gold, match}
- `config.toml` — exact config used (with `${OPENROUTER_API_KEY}` placeholder)
- (optional) `workspace/` — stage snapshots (excluding regenerable `vector_database/`)

## CSV Field Conventions

For per-sample comparison CSVs (e.g., 4-config ablation):

| Column | Type | Example |
|---|---|---|
| `qid` | int | 42 |
| `db_id` | str | `formula_1` |
| `difficulty` | str | `simple` / `moderate` / `challenging` |
| `gold_sql` | str | the BIRD reference SQL |
| `ModularSQL_Baseline` | `PASS`/`FAIL` | match flag |
| `ModularSQL_P1` | `PASS`/`FAIL` | |
| `ModularSQL_P2` | `PASS`/`FAIL` | |
| `ModularSQL_Full` | `PASS`/`FAIL` | |
| `ModularSQL_Baseline_sql` | str | predicted SQL by baseline |
| `ModularSQL_P1_sql` | str | (etc.) |
| `notes` | str | annotations like "fixed by DISTINCT", "fixed by PK hint" |
