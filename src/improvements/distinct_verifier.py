"""
DISTINCT-aware Semantic Verifier (Pattern 1).

Rule: if a SELECT query (a) has a JOIN, (b) contains no aggregate / GROUP BY,
(c) does not already use DISTINCT, and (d) the executed result contains
duplicate rows, then injecting DISTINCT is highly likely to be correct.

Rationale: when SQL joins a 1-to-many relationship without DISTINCT, the
result multiplies; gold answers for "list / names / values of X" almost
always want unique rows.

No LLM calls. Pure rule + execution check.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any


_DISTINCT_RE = re.compile(r"\bSELECT\s+DISTINCT\b", re.IGNORECASE)
_FIRST_SELECT_RE = re.compile(r"^(\s*SELECT)(\s+)(?!DISTINCT\b)", re.IGNORECASE)
_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
_GROUPBY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)


@dataclass
class VerifierDecision:
    needs_distinct: bool
    reason: str
    new_sql: Optional[str] = None
    new_rows: Optional[List[Tuple[Any, ...]]] = None
    exec_error: Optional[str] = None


def has_distinct(sql: str) -> bool:
    return bool(_DISTINCT_RE.search(sql))


def has_join(sql: str) -> bool:
    return bool(_JOIN_RE.search(sql))


def has_aggregate(sql: str) -> bool:
    """Outer SELECT has aggregate function (single-row result, DISTINCT irrelevant)."""
    return bool(_AGG_RE.search(sql))


def has_groupby(sql: str) -> bool:
    return bool(_GROUPBY_RE.search(sql))


def add_distinct(sql: str) -> str | None:
    """Insert DISTINCT after the leading SELECT keyword.

    Returns the rewritten SQL, or None if the regex could not find a leading
    SELECT (e.g. CTE-prefixed `WITH ... SELECT`, parenthesized subqueries).
    Callers must treat None as "injection unsupported" and fall back to the
    original SQL.
    """
    new_sql, n = _FIRST_SELECT_RE.subn(r"\1 DISTINCT ", sql, count=1)
    return new_sql if n == 1 else None


def has_duplicate_rows(rows: List[Tuple[Any, ...]]) -> bool:
    if len(rows) < 2:
        return False
    return len(rows) > len(set(rows))


def duplication_ratio(rows: List[Tuple[Any, ...]]) -> float:
    """Fraction of rows that are non-unique: (n - n_unique) / n.
    High ratio (>=0.7) indicates JOIN-induced cartesian explosion where DISTINCT
    is highly likely correct. Low ratio (<0.5) suggests sparse, semantically
    meaningful duplicates that gold typically preserves."""
    if not rows:
        return 0.0
    return (len(rows) - len(set(rows))) / len(rows)


def null_density(rows: List[Tuple[Any, ...]]) -> float:
    """Fraction of rows where at least one cell is None. Retained for diagnostic
    use; the dev1534 analysis showed null_density=0 across all 87 P1-fire cases,
    so this guard never fires in practice on BIRD-Dev."""
    if not rows:
        return 0.0
    n_with_null = sum(1 for r in rows if any(v is None for v in r))
    return n_with_null / len(rows)


def execute_sql(db_path: str, sql: str) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
        conn.close()
        return rows, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def verify_and_fix(
    sql: str,
    db_path: str,
    pred_rows: Optional[List[Tuple[Any, ...]]] = None,
    dup_ratio_threshold: float = 0.80,
) -> VerifierDecision:
    """
    Decide whether to inject DISTINCT into `sql` and, if so, return the
    rewritten SQL and its execution result.

    `dup_ratio_threshold` controls how dense the duplicates must be before we
    fire. Calibrated on dev1534 P1-fire analysis: at τ=0.80, in-sample net
    delta improves from +5 to +21 (1032/1534 = 67.28% vs baseline 65.91%);
    leave-one-DB-out CV gives +17 net (unbiased generalization estimate).
    Default τ=0.80 chosen for higher fix:break precision (3.33:1 vs 2.31:1
    at τ=0.65) — safer for test-set distribution shift.
    """
    if has_distinct(sql):
        return VerifierDecision(False, "already_has_distinct")
    if has_aggregate(sql):
        return VerifierDecision(False, "aggregate_query")
    if has_groupby(sql):
        return VerifierDecision(False, "has_group_by")
    if not has_join(sql):
        return VerifierDecision(False, "no_join")

    if pred_rows is None:
        pred_rows, err = execute_sql(db_path, sql)
        if err is not None:
            return VerifierDecision(False, f"pred_exec_error: {err}")

    if not has_duplicate_rows(pred_rows):
        return VerifierDecision(False, "no_duplicates")

    # Duplication-ratio guard: low dup_ratio signals semantically meaningful
    # duplicates (e.g., two distinct entities sharing a display name) that gold
    # preserves; high dup_ratio signals JOIN-induced cartesian explosion.
    dup_r = duplication_ratio(pred_rows)
    if dup_r < dup_ratio_threshold:
        return VerifierDecision(False, f"low_dup_ratio_{dup_r:.2f}")

    new_sql = add_distinct(sql)
    if new_sql is None:
        # Regex couldn't inject (CTE / wrapped SELECT) — fail closed
        return VerifierDecision(False, "cannot_inject_distinct_unsupported_form")
    new_rows, err = execute_sql(db_path, new_sql)
    if err is not None:
        return VerifierDecision(False, f"distinct_caused_exec_error: {err}")

    return VerifierDecision(
        needs_distinct=True,
        reason="join_without_distinct_produces_duplicates",
        new_sql=new_sql,
        new_rows=sorted(new_rows, key=lambda r: str(r)),
    )
