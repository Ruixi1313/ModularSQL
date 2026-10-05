# A five-minute multiplicity example

This self-contained, synthetic SQLite example illustrates the **Multiplicity
Blind Spot** discussed in [ModularSQL: A Runtime Guardrail for the Multiplicity
Blind Spot in Text-to-SQL](https://arxiv.org/abs/2609.29573) by Tianxin Zhou and
Ruixi Lin (2026). See the [research page](https://research.searcher.cloud/papers/modularsql/)
and the repository's [citation](../../README.md#citation).

## Run

From the repository root, with Python 3.9+ and its standard-library `sqlite3`
module available:

```bash
python3 examples/multiplicity/demo.py
python3 -m unittest discover -s examples/multiplicity -v
```

No package installation, API key, network call, model, BIRD data, or DeepEye
checkout is needed. The script creates a fresh in-memory database and closes it
after reading results. It writes no database or result files.

## What changes when duplicates matter?

The synthetic `purchases` table has three rows:

| purchase_id | customer |
| --- | --- |
| 1 | Ada |
| 2 | Ada |
| 3 | Bo |

Both purchases by Ada are real, separate purchases in this invented example.
The two queries are:

```sql
SELECT customer FROM purchases ORDER BY customer, purchase_id;
SELECT DISTINCT customer FROM purchases ORDER BY customer;
```

- For **one customer name per purchase**, the first query is the reference.
  The second incorrectly suppresses Ada's second purchase.
- For **unique customer names**, the second query is the reference.
  The first incorrectly repeats Ada.

The intended task determines the correct multiplicity. Adding or removing
`DISTINCT` everywhere is not a safe fix.

Set comparison, `set(candidate) == set(reference)`, discards repeated rows.
Bag (multiset) comparison, `Counter(candidate) == Counter(reference)`, also
ignores row order but preserves each complete row's occurrence count. `ORDER BY`
makes the printed example deterministic; neither comparator checks ordering.

Expected output (also checked exactly by the tests):

```text
One customer name per purchase (duplicates required)
  reference: [('Ada',), ('Ada',), ('Bo',)]
  candidate: [('Ada',), ('Bo',)]
  set_equal: True
  bag_equal: False

Unique customer names (duplicates unwanted)
  reference: [('Ada',), ('Bo',)]
  candidate: [('Ada',), ('Ada',), ('Bo',)]
  set_equal: True
  bag_equal: False
```

## Scope and reuse

This example demonstrates result-comparison semantics only. It does **not** run
ModularSQL's detector, P1/P2 interventions, LLM rescue, or the BIRD benchmark, and
does not reproduce or validate the paper's performance, cost, or accuracy claims.
Those workflows and their external dependencies are documented in the
[main README](../../README.md#canonical-reproduction-scripts).

`compare_rows` is intentionally small and uses Python equality on materialized
tuples. The demo uses non-null text values. It does not normalize SQL types,
collations, floating-point values, or column permutations; ordered-result tasks
need a different comparison. Query failures propagate as errors rather than
being treated as empty results. The tests additionally check row-order
invariance, different values, equal-length bags with different counts, empty
results, complete-row/column-position comparison, and fresh-database isolation.

All data here is invented for this example. The example is covered by the
repository's [MIT License](../../LICENSE); it includes no benchmark data or
third-party code.
