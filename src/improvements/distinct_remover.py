"""
Pattern 3: DISTINCT Remover (inverse of Pattern 1).

When pred SQL contains DISTINCT but the duplicates it removes are likely
semantically meaningful (gold expects them), remove DISTINCT.

Symmetric to Pattern 1's dup_ratio guard:
  - P1 (add):    fire when dup_ratio >= τ  (cartesian explosion; collapse)
  - P3 (remove): fire when dup_ratio <= τ  (sparse duplicates; preserve)

The dup_ratio here is computed on pred-WITHOUT-DISTINCT to measure how many
rows DISTINCT was suppressing.

No LLM calls. Pure rule + execution check.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any


_DISTINCT_RE = re.compile(r"(\bSELECT)\s+DISTINCT\s+", re.IGNORECASE)
_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
_GROUPBY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)


@dataclass
class RemoverDecision:
    should_remove: bool
    reason: str
    new_sql: Optional[str] = None
    new_rows: Optional[List[Tuple[Any, ...]]] = None


def has_distinct(sql: str) -> bool:
    return bool(_DISTINCT_RE.search(sql))


def has_aggregate(sql: str) -> bool:
    return bool(_AGG_RE.search(sql))


def has_groupby(sql: str) -> bool:
    return bool(_GROUPBY_RE.search(sql))


def remove_distinct(sql: str) -> str | None:
    """Strip the leading DISTINCT keyword. Returns None if no match."""
    new_sql, n = _DISTINCT_RE.subn(r"\1 ", sql, count=1)
    return new_sql if n == 1 else None


def duplication_ratio(rows):
    if not rows:
        return 0.0
    return (len(rows) - len(set(rows))) / len(rows)


def execute_sql(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
        conn.close()
        return rows, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def verify_and_remove(
    sql: str,
    db_path: str,
    pred_rows: Optional[List[Tuple[Any, ...]]] = None,
    dup_ratio_threshold: float = 0.10,
) -> RemoverDecision:
    """
    Decide whether to remove DISTINCT from `sql`.

    Default τ=0.10 chosen conservatively after dev1534 sweep: at this threshold
    the rule fires 5 times with 3 fixes / 0 breaks (perfect precision, small
    yield). Above τ=0.30 the fix:break ratio collapses to ~1:1, reflecting that
    dup_ratio is NOT a symmetric signal for DISTINCT verification — see paper
    Section 4.3 for the asymmetric-signal finding.

    Fires when:
      1. SQL contains DISTINCT
      2. SQL has no aggregate (DISTINCT-in-aggregate is different beast)
      3. SQL has no GROUP BY (DISTINCT after GROUP BY is suspicious; handle separately)
      4. Executing without DISTINCT yields strictly more rows
      5. The duplication ratio (without DISTINCT) is BELOW the threshold,
         signaling those duplicates are likely semantically meaningful.
    """
    if not has_distinct(sql):
        return RemoverDecision(False, "no_distinct")
    if has_aggregate(sql):
        return RemoverDecision(False, "has_aggregate")
    if has_groupby(sql):
        return RemoverDecision(False, "has_groupby")

    new_sql = remove_distinct(sql)
    if new_sql is None:
        return RemoverDecision(False, "cannot_remove_distinct")

    new_rows, err = execute_sql(db_path, new_sql)
    if err is not None:
        return RemoverDecision(False, f"remove_caused_error: {err}")

    if pred_rows is None:
        pred_rows, _ = execute_sql(db_path, sql)
    if pred_rows is None:
        return RemoverDecision(False, "pred_exec_failed")

    if len(new_rows) <= len(pred_rows):
        return RemoverDecision(False, "distinct_was_noop")

    # Compute dup_ratio on the WITHOUT-DISTINCT rows
    dup_r = duplication_ratio(new_rows)
    if dup_r > dup_ratio_threshold:
        return RemoverDecision(False, f"high_dup_ratio_{dup_r:.2f}_keep_distinct")

    return RemoverDecision(
        should_remove=True,
        reason=f"low_dup_ratio_{dup_r:.2f}_duplicates_likely_meaningful",
        new_sql=new_sql,
        new_rows=sorted(new_rows, key=lambda r: str(r)),
    )
