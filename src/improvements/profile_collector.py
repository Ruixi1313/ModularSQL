"""
Database Profile Collector (Pattern 2, Module 2.1)

For each column in a BIRD-Dev SQLite database, compute the structural
statistics that AT&T's profiling-based Text-to-SQL method uses as input
to its metadata summariser:

    - total record count
    - NULL count
    - distinct count
    - declared type (from PRAGMA)
    - inferred type ("numeric" / "text" / "mixed" / "date-like")
    - min / max
    - string length: min, max, average
    - character class signature (alpha / digit / punctuation)
    - common prefix (if all values share one)
    - top-k most frequent non-NULL values (with counts)

Pure SQLite queries. No LLM calls.

Output is a JSON dict per DB:
    {
        "db_id": "...",
        "tables": {
            "<table>": {
                "row_count": int,
                "columns": {
                    "<col>": {<stat fields>}
                }
            }
        }
    }
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


TOP_K_VALUES = 10
MAX_DISTINCT_FOR_FULL_TOP = 100_000
SAFE_LIKE = re.compile(r"[^\w\- ]")


def _qident(name: str) -> str:
    """SQLite identifier quoting."""
    return '"' + name.replace('"', '""') + '"'


def _classify_inferred_type(values: List[Any]) -> str:
    """Heuristic type from sample values."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "all_null"
    is_num = lambda v: isinstance(v, (int, float)) or (
        isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v) is not None
    )
    is_date = lambda v: isinstance(v, str) and re.fullmatch(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}.*", v
    ) is not None
    nums = sum(1 for v in non_null if is_num(v))
    dates = sum(1 for v in non_null if is_date(v))
    n = len(non_null)
    if dates / n > 0.8:
        return "date-like"
    if nums / n > 0.95:
        return "numeric"
    if nums == 0:
        return "text"
    return "mixed"


def _char_class_signature(values: List[Any]) -> str:
    """Compact alphabet hint for the column values."""
    chars = "".join(str(v) for v in values[:200] if v is not None)
    if not chars:
        return ""
    has_upper = any(c.isupper() for c in chars)
    has_lower = any(c.islower() for c in chars)
    has_digit = any(c.isdigit() for c in chars)
    has_space = " " in chars
    has_punct = any(not c.isalnum() and not c.isspace() for c in chars)
    parts = []
    if has_upper: parts.append("UPPER")
    if has_lower: parts.append("lower")
    if has_digit: parts.append("digit")
    if has_space: parts.append("space")
    if has_punct: parts.append("punct")
    return "+".join(parts)


def _common_prefix(values: List[str]) -> str:
    strs = [str(v) for v in values if v is not None][:200]
    if len(strs) < 2:
        return ""
    pref = strs[0]
    for s in strs[1:]:
        i = 0
        while i < min(len(pref), len(s)) and pref[i] == s[i]:
            i += 1
        pref = pref[:i]
        if not pref:
            break
    return pref


def _string_length_stats(values: List[Any]) -> Dict[str, Any]:
    lens = [len(str(v)) for v in values if v is not None]
    if not lens:
        return {"min": None, "max": None, "avg": None}
    return {"min": min(lens), "max": max(lens), "avg": round(sum(lens) / len(lens), 2)}


def profile_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    declared_type: str,
) -> Dict[str, Any]:
    cur = conn.cursor()
    tab_q = _qident(table)
    col_q = _qident(column)

    # Total + NULL counts
    cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {col_q} IS NULL THEN 1 ELSE 0 END) FROM {tab_q}")
    total, n_null = cur.fetchone()
    n_null = n_null or 0

    # Distinct count (could be expensive on huge tables; OK for BIRD scale)
    try:
        cur.execute(f"SELECT COUNT(DISTINCT {col_q}) FROM {tab_q}")
        n_distinct = cur.fetchone()[0]
    except sqlite3.Error:
        n_distinct = None

    # Min / Max
    try:
        cur.execute(f"SELECT MIN({col_q}), MAX({col_q}) FROM {tab_q} WHERE {col_q} IS NOT NULL")
        v_min, v_max = cur.fetchone()
    except sqlite3.Error:
        v_min = v_max = None

    # Top-k values + sample
    try:
        cur.execute(
            f"SELECT {col_q}, COUNT(*) c FROM {tab_q} "
            f"WHERE {col_q} IS NOT NULL GROUP BY {col_q} ORDER BY c DESC LIMIT {TOP_K_VALUES}"
        )
        top_k = [{"value": str(v), "count": int(c)} for v, c in cur.fetchall()]
    except sqlite3.Error:
        top_k = []

    # Sample for type / length / char class (re-read up to 200 non-null)
    try:
        cur.execute(f"SELECT {col_q} FROM {tab_q} WHERE {col_q} IS NOT NULL LIMIT 200")
        sample_vals = [row[0] for row in cur.fetchall()]
    except sqlite3.Error:
        sample_vals = []

    inferred = _classify_inferred_type(sample_vals)
    char_sig = _char_class_signature(sample_vals)
    length = _string_length_stats(sample_vals) if inferred not in ("numeric",) else {}
    prefix = _common_prefix(sample_vals) if inferred == "text" else ""

    return {
        "declared_type": (declared_type or "").upper() or None,
        "inferred_type": inferred,
        "total": total,
        "n_null": n_null,
        "null_ratio": round(n_null / total, 3) if total else 0,
        "n_distinct": n_distinct,
        "distinct_ratio": round(n_distinct / total, 3) if (total and n_distinct is not None) else None,
        "min": v_min if isinstance(v_min, (int, float, str)) or v_min is None else str(v_min),
        "max": v_max if isinstance(v_max, (int, float, str)) or v_max is None else str(v_max),
        "char_class": char_sig,
        "common_prefix": prefix,
        "string_length": length,
        "top_k_values": top_k,
    }


def profile_database(db_path: str, db_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cur.fetchall()]

    out_tables: Dict[str, Any] = {}
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {_qident(table)}")
        row_count = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info({_qident(table)})")
        cols_info = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
        col_profiles: Dict[str, Any] = {}
        for cid, name, declared_type, notnull, dflt, pk in cols_info:
            col_profiles[name] = profile_column(conn, table, name, declared_type)
            col_profiles[name]["is_primary_key"] = bool(pk)
            col_profiles[name]["not_null"] = bool(notnull)
        out_tables[table] = {
            "row_count": row_count,
            "columns": col_profiles,
        }
    conn.close()
    return {"db_id": db_id, "tables": out_tables}


def main():
    """Profile all 11 BIRD-Dev databases and cache results."""
    DB_ROOT = Path(__file__).resolve().parents[2] / "data/bird/dev/dev_databases"
    OUT_DIR = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/profile_cache"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    db_dirs = sorted([d for d in DB_ROOT.iterdir() if d.is_dir()])
    print(f"Profiling {len(db_dirs)} databases...")

    for db_dir in db_dirs:
        db_id = db_dir.name
        db_path = db_dir / f"{db_id}.sqlite"
        if not db_path.exists():
            print(f"  {db_id}: SKIP (no .sqlite file)")
            continue
        print(f"  {db_id} ...", end=" ", flush=True)
        profile = profile_database(str(db_path), db_id)
        out_file = OUT_DIR / f"{db_id}.json"
        out_file.write_text(json.dumps(profile, indent=2, default=str))
        n_cols = sum(len(t["columns"]) for t in profile["tables"].values())
        print(f"done ({len(profile['tables'])} tables, {n_cols} columns) → {out_file.name}")

    print(f"\nProfiles cached in {OUT_DIR}/")


if __name__ == "__main__":
    main()
