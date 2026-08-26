"""
Pattern 2 — minimal "slim" hint variant (post-regression iteration).

Lessons from the verbose hint pilot (which lost 4pp):
  - Top-k values were misused as SQL literals (Mode 3 failure)
  - Cardinality "%× dup" misleadingly triggered DISTINCT in non-JOIN contexts (Mode 1)
  - Range info added prompt bloat without clear benefit

The slim version keeps only three highly-discriminating signals that
target known failure patterns without leaking literal values into prompts:

  1. [Primary Key]           — disambiguate id vs label column choice
  2. [Common Prefix: 'TR']   — anchor ID-format expectations (anti-hallucination)
  3. [Contains NULLs]        — only when null_ratio > 0.05 (Mode 1 guard)

Anything else is dropped. Hint length: typically 10-30 chars.
"""

from __future__ import annotations

from typing import Any, Dict


NULL_THRESHOLD = 0.05
MIN_PREFIX_LEN = 2


def _pk_tag(col_profile: Dict[str, Any]) -> str | None:
    """Strong PK signal: declared PK or near-unique."""
    if col_profile.get("is_primary_key"):
        return "Primary Key"
    total = col_profile.get("total") or 0
    distinct = col_profile.get("n_distinct") or 0
    if total and distinct and distinct / total >= 0.99 and total >= 2:
        return "Primary Key"  # near-unique → treat as PK-like for column choice
    return None


def _prefix_tag(col_profile: Dict[str, Any]) -> str | None:
    """Anchor ID-format expectations (anti-hallucination)."""
    if col_profile.get("inferred_type") not in ("text", "mixed"):
        return None
    prefix = col_profile.get("common_prefix") or ""
    if len(prefix) >= MIN_PREFIX_LEN:
        return f"Common Prefix: '{prefix}'"
    return None


def _nulls_tag(col_profile: Dict[str, Any]) -> str | None:
    """Flag NULL presence only when meaningful (>5%)."""
    null_ratio = col_profile.get("null_ratio") or 0
    if null_ratio > NULL_THRESHOLD:
        return "Contains NULLs"
    return None


def generate_profile_hint(col_profile: Dict[str, Any]) -> str:
    """Render the [Profile: ...] hint for a single column (slim variant)."""
    parts = []
    for fn in (_pk_tag, _prefix_tag, _nulls_tag):
        tag = fn(col_profile)
        if tag:
            parts.append(tag)
    if not parts:
        return ""
    return "[Profile: " + ", ".join(parts) + "]"
