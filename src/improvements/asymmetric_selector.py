"""
Pattern 6.2: Asymmetric Veto Selector.

Default to trusting DeepEye's S7 tournament pick. Only override when S7's
chosen SQL fails a "physical health" probe:
  - Execution error
  - Empty result set ([])
  - Cartesian-explosion (dup_ratio >= 0.80)

When triggered, rescue by majority-vote over the 11 other revised candidates
(after filtering out any that also fail the err / empty health check).

Goal: collapse the 15 "breaks" that 6.0 introduced (S7 was right but we
overrode it) while keeping the 9 "fixes" (S7 picked broken output and a
healthy majority alternative exists).

No LLM calls.
"""
from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple


DUP_EXPLOSION_THRESHOLD = 0.80


@dataclass
class AsymmetricDecision:
    selected_sql: str
    selected_idx: int        # index in revised pool; -1 means "from S7 base"
    triggered: bool          # whether rescue was triggered (False = trust S7)
    trigger_reason: str      # "healthy" | "exec_error" | "empty_result" | "cartesian_explosion"
    rescue_group_size: int   # n candidates voting for the selected SQL when rescued
    n_safe_candidates: int   # how many of the 11 non-base candidates were safe


def execute(db_path: str, sql: str, timeout_sec: int = 5):
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


def dup_ratio(rows):
    if not rows:
        return 0.0
    return (len(rows) - len(set(rows))) / len(rows)


def probe_health(rows) -> str:
    """Return 'healthy' or a reason string."""
    if rows is None:
        return "exec_error"
    if len(rows) == 0:
        return "empty_result"
    if dup_ratio(rows) >= DUP_EXPLOSION_THRESHOLD:
        return "cartesian_explosion"
    return "healthy"


def select_asymmetric(
    base_sql: str,
    revised_candidates: List[str],
    db_path: str,
    timeout_sec: int = 5,
) -> AsymmetricDecision:
    """Trust S7 (base_sql) unless its physical probe fails.
    On failure, majority vote over the remaining 11 safe candidates."""
    base_rows = execute(db_path, base_sql, timeout_sec=timeout_sec)
    health = probe_health(base_rows)

    # 🟢 base is healthy → trust S7
    if health == "healthy":
        return AsymmetricDecision(
            selected_sql=base_sql, selected_idx=-1,
            triggered=False, trigger_reason="healthy",
            rescue_group_size=0, n_safe_candidates=0,
        )

    # 🔴 base failed probe → rescue mode
    # Filter the 11 other candidates: drop S7 base + drop err/empty
    other_candidates = [
        (idx, sql) for idx, sql in enumerate(revised_candidates)
        if sql != base_sql
    ]
    safe = []
    safe_groups = defaultdict(list)
    for idx, sql in other_candidates:
        rows = execute(db_path, sql, timeout_sec=timeout_sec)
        if rows is None or len(rows) == 0:
            continue
        safe.append((idx, sql, rows))
        safe_groups[frozenset(rows)].append((idx, sql))

    if not safe_groups:
        # All other candidates also unsafe — fall back to base
        return AsymmetricDecision(
            selected_sql=base_sql, selected_idx=-1,
            triggered=True, trigger_reason=health,
            rescue_group_size=0, n_safe_candidates=0,
        )

    # Majority vote among safe candidates
    best_key = max(
        safe_groups,
        key=lambda k: (len(safe_groups[k]), -min(i for i, _ in safe_groups[k])),
    )
    best_group = safe_groups[best_key]
    best_idx, best_sql = min(best_group, key=lambda x: x[0])
    return AsymmetricDecision(
        selected_sql=best_sql, selected_idx=best_idx,
        triggered=True, trigger_reason=health,
        rescue_group_size=len(best_group),
        n_safe_candidates=len(safe),
    )
