"""
LLM judge for the DISTINCT-aware Semantic Verifier (Pattern 1B).

When the rule-based pre-filter says "this SQL likely needs DISTINCT" (JOIN +
no aggregate + duplicates in result), this judge takes the final decision:
  - YES → add DISTINCT
  - NO  → leave as-is (duplicates are part of the intended answer)

Uses Qwen3-Coder-30B-A3B-Instruct via OpenRouter (same as main pipeline).
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Any
from pathlib import Path

from openai import OpenAI


PROMPT = """You are a SQL expert reviewing a generated query.

The query joins multiple tables and currently has no DISTINCT. Its execution result contains duplicate rows. We must decide whether to inject DISTINCT into the SELECT clause.

Question: {question}
Evidence: {evidence}

Generated SQL:
{sql}

Sample of executed result (first 5 rows):
{sample_rows}

Total rows: {n_rows}, distinct rows: {n_distinct}

Decision rule:
- Answer "YES" if the question asks for a list of unique entities (e.g., "list the names of races", "what are the categories of cards") and the duplicates in the current result are spurious (caused by JOIN multiplying rows).
- Answer "NO" if duplicates are part of the intended answer (e.g., "list all phone numbers" where each occurrence represents a separate record; "list every match score where teams played").

Respond with exactly one word: YES or NO. No explanation."""


def _load_env(env_path: Path) -> dict:
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _get_client() -> OpenAI:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    env = _load_env(env_path)
    api_key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base_url = env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env or env vars")
    return OpenAI(api_key=api_key, base_url=base_url)


def llm_should_add_distinct(
    question: str,
    evidence: str,
    sql: str,
    pred_rows: List[Tuple[Any, ...]],
    model: str = "qwen/qwen3-coder-30b-a3b-instruct",
    client: OpenAI | None = None,
) -> Tuple[bool, str]:
    """
    Ask the LLM whether DISTINCT should be added.

    Returns (decision: bool, raw_response: str).
    """
    client = client or _get_client()
    n_rows = len(pred_rows)
    n_distinct = len(set(pred_rows))
    sample = "\n".join(f"  {row}" for row in pred_rows[:5])

    user_prompt = PROMPT.format(
        question=question,
        evidence=evidence or "(none)",
        sql=sql,
        sample_rows=sample if sample else "  (empty)",
        n_rows=n_rows,
        n_distinct=n_distinct,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=8,
    )
    raw = (resp.choices[0].message.content or "").strip().upper()
    # Take the first YES/NO token in the response
    first = re.search(r"\b(YES|NO)\b", raw)
    decision = bool(first and first.group(1) == "YES")
    return decision, raw
