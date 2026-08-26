"""
Pattern 6: Better Selector via execution-based majority vote.

Replaces DeepEye's S7 tournament selection. For each question:
  1. Execute all 12 S5 candidates (with hard timeout)
  2. Group candidates by their set(rows) result (BIRD official EX semantics)
  3. Pick the group with the most candidates (majority vote)
  4. Tie-breaker: pick the candidate with the lowest original index

No LLM calls. Pure rule + execution check.
"""
from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple


@dataclass
class SelectorDecision:
    selected_sql: str
    selected_idx: int
    group_size: int
    n_distinct_groups: int
    fallback: str  # "majority" | "all_errors" | "single_candidate"


def execute(db_path: str, sql: str, timeout_sec: int = 5) -> Optional[List[Tuple[Any, ...]]]:
    """Execute SQL with hard timeout via threading.Timer + conn.interrupt."""
    conn = sqlite3.connect(db_path, timeout=timeout_sec)
    timer = threading.Timer(timeout_sec, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return [tuple(r) for r in cur.fetchall()]
    except Exception:
        return None
    finally:
        timer.cancel()
        conn.close()


def select_by_majority(
    candidates: List[str],
    db_path: str,
    timeout_sec: int = 5,
) -> SelectorDecision:
    """Execute all candidates, group by set(rows), pick majority group."""
    if not candidates:
        raise ValueError("no candidates")
    if len(candidates) == 1:
        return SelectorDecision(
            selected_sql=candidates[0], selected_idx=0,
            group_size=1, n_distinct_groups=1, fallback="single_candidate",
        )

    # Execute each candidate and bucket by result-set
    groups: dict = defaultdict(list)  # frozenset(rows) -> [(idx, sql)]
    for idx, sql in enumerate(candidates):
        rows = execute(db_path, sql, timeout_sec=timeout_sec)
        if rows is None:
            key = "__ERR__"
        else:
            key = frozenset(rows)
        groups[key].append((idx, sql))

    # Remove error group from voting (unless ALL candidates errored)
    valid_groups = {k: v for k, v in groups.items() if k != "__ERR__"}
    if not valid_groups:
        # All errored — return first candidate as fallback
        return SelectorDecision(
            selected_sql=candidates[0], selected_idx=0,
            group_size=0, n_distinct_groups=0, fallback="all_errors",
        )

    # Pick group with most votes; tie-break by smallest minimum index
    best_key = max(
        valid_groups,
        key=lambda k: (len(valid_groups[k]), -min(i for i, _ in valid_groups[k])),
    )
    best_group = valid_groups[best_key]
    # Within the winning group, return the candidate with the lowest index
    best_idx, best_sql = min(best_group, key=lambda x: x[0])
    return SelectorDecision(
        selected_sql=best_sql,
        selected_idx=best_idx,
        group_size=len(best_group),
        n_distinct_groups=len(valid_groups),
        fallback="majority",
    )
