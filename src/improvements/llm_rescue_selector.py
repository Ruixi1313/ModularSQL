"""
Pattern 6.3: LLM-as-Judge Rescue Selector.

Extends v6.2 asymmetric veto. When S7's pick is physically unhealthy
(execution error / empty result / cartesian explosion), instead of
majority vote over the remaining safe candidates, call an LLM to pick
the best one.

LLM sees: question, schema hint, all candidates + their execution preview.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import ssl
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Any, Dict
import urllib.request

# Use certifi's CA bundle for HTTPS verification. macOS Python ships without
# bundled CA certs; certifi provides them portably across platforms.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # Fallback: use system defaults. Install certifi (`pip install certifi`)
    # if SSL verification errors occur on macOS.
    _SSL_CTX = ssl.create_default_context()


DUP_EXPLOSION_THRESHOLD = 0.80

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-coder-30b-a3b-instruct")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMJudgeDecision:
    selected_sql: str
    selected_idx: int
    triggered: bool
    trigger_reason: str
    llm_picked_idx: int      # 0..11 or -1 if parse failed
    llm_raw_response: str
    prompt_tokens: int
    completion_tokens: int


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
    if rows is None:
        return "exec_error"
    if len(rows) == 0:
        return "empty_result"
    if dup_ratio(rows) >= DUP_EXPLOSION_THRESHOLD:
        return "cartesian_explosion"
    return "healthy"


def _format_rows_preview(rows, max_rows=3) -> str:
    if rows is None:
        return "ERROR"
    if len(rows) == 0:
        return "EMPTY"
    preview = rows[:max_rows]
    suffix = f" ... ({len(rows)} rows total)" if len(rows) > max_rows else f" ({len(rows)} rows)"
    return "; ".join(str(r) for r in preview) + suffix


def build_prompt(question: str, hint: str, schema_summary: str,
                 candidates: List[str], previews: List[str],
                 base_sql: str, base_preview: str, trigger_reason: str) -> str:
    """Build a prompt asking LLM to pick the best candidate."""
    cand_text = "\n".join(
        f"[{i}] SQL: {c.strip()[:600]}\n    Result: {p}"
        for i, (c, p) in enumerate(zip(candidates, previews))
    )
    return f"""You are selecting the best SQL from candidate solutions for a BIRD benchmark question.

Question: {question}
Hint: {hint or '(none)'}

Schema (relevant tables/columns):
{schema_summary[:1500]}

The currently selected SQL has a physical issue ({trigger_reason}):
{base_sql.strip()[:400]}
Result: {base_preview}

Here are 12 alternative candidates with their execution results:

{cand_text}

Pick the candidate that best answers the question.
Respond with ONLY the candidate number (0-11), nothing else.

Your answer:"""


def call_openrouter(prompt: str, model: str = OPENROUTER_MODEL,
                    timeout: int = 30) -> Dict:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=body,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_llm_choice(text: str, n_candidates: int) -> int:
    """Parse LLM response to extract an integer in [0, n_candidates)."""
    m = re.search(r"\b(\d+)\b", text)
    if not m:
        return -1
    n = int(m.group(1))
    if 0 <= n < n_candidates:
        return n
    return -1


def select_with_llm_rescue(
    question: str,
    hint: str,
    schema_summary: str,
    base_sql: str,
    revised_candidates: List[str],
    db_path: str,
    timeout_sec: int = 5,
) -> LLMJudgeDecision:
    base_rows = execute(db_path, base_sql, timeout_sec=timeout_sec)
    health = probe_health(base_rows)
    if health == "healthy":
        return LLMJudgeDecision(
            selected_sql=base_sql, selected_idx=-1,
            triggered=False, trigger_reason="healthy",
            llm_picked_idx=-1, llm_raw_response="",
            prompt_tokens=0, completion_tokens=0,
        )

    previews = []
    for sql in revised_candidates:
        rows = execute(db_path, sql, timeout_sec=timeout_sec)
        previews.append(_format_rows_preview(rows))
    base_preview = _format_rows_preview(base_rows)

    prompt = build_prompt(question, hint, schema_summary,
                          revised_candidates, previews,
                          base_sql, base_preview, health)

    try:
        resp = call_openrouter(prompt)
        msg = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)
    except Exception as e:
        return LLMJudgeDecision(
            selected_sql=base_sql, selected_idx=-1,
            triggered=True, trigger_reason=f"{health}+api_error:{e}",
            llm_picked_idx=-1, llm_raw_response="",
            prompt_tokens=0, completion_tokens=0,
        )

    picked = parse_llm_choice(msg, len(revised_candidates))
    if picked < 0:
        # Parse failed → fall back to base
        return LLMJudgeDecision(
            selected_sql=base_sql, selected_idx=-1,
            triggered=True, trigger_reason=f"{health}+parse_fail",
            llm_picked_idx=-1, llm_raw_response=msg,
            prompt_tokens=prompt_tok, completion_tokens=completion_tok,
        )

    return LLMJudgeDecision(
        selected_sql=revised_candidates[picked], selected_idx=picked,
        triggered=True, trigger_reason=health,
        llm_picked_idx=picked, llm_raw_response=msg,
        prompt_tokens=prompt_tok, completion_tokens=completion_tok,
    )
