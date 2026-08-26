#!/usr/bin/env python3
"""Targeted robustness validation: does the multiplicity blind spot transfer
to GPT-4o predictions?

This is NOT an unbiased estimation of GPT-4o's blind-spot rate. It is a
*targeted robustness validation* — a stress test asking: "when GPT-4o is
presented with multiplicity-prone queries (oversampled from Qwen's failure
distribution), does the blind spot manifest?"

Sampling design (n=50):
  Failure-prone strata (40 cases, oversampled to maximize signal):
    missing_distinct          : 15  (Qwen missed DISTINCT)
    extra_distinct            : 10  (Qwen over-applied DISTINCT)
    extra_joins_+1            : 10  (Qwen had join amplification)
    structurally_identical    :  5  (right skeleton, wrong details)
  Control stratum (10 cases):
    random easy-pass          : 10  (random Qwen-easy-pass, no failure label)

By design this overweights cardinality-prone queries — biasing the test
TOWARD finding a blind spot. If GPT-4o is robust to multiplicity errors,
this design should still NOT find a gap; finding one is therefore strong
evidence the phenomenon is evaluator-level rather than Qwen-specific.

Cost: ~50 × $0.10-0.30 = $5-15
Time: ~10-15 min (sequential calls)
"""
import csv
import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.request
import ssl
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
S7 = ROOT / "external/DeepEye-SQL/workspace_full/sql_selection.topk3_backup/bird/dev.snapshot.data/items.jsonl"
CLUSTERS = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/failure_clusters_set.csv"
OUT = ROOT / "results/ModularSQL_Baseline_dev1534_20260514/gpt4o_qualitative.csv"

# Load OPENAI_API_KEY from .env
ENV = ROOT / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = "gpt-4o-2024-11-20"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def execute(db, sql, timeout=5):
    conn = sqlite3.connect(db, timeout=timeout)
    timer = threading.Timer(timeout, conn.interrupt)
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


def set_match(p, g):
    return p is not None and g is not None and set(p) == set(g)


def multiset_match(p, g):
    if p is None or g is None: return False
    return sorted(p, key=lambda r: str(r)) == sorted(g, key=lambda r: str(r))


def echo(m):
    print(m, flush=True)


def format_schema(item):
    tables = item["input"]["database_schema"].get("tables", {})
    if not isinstance(tables, dict):
        return ""
    parts = []
    for tbl_name, tbl in tables.items():
        cols = list(tbl.get("columns", {}).keys())[:20]
        parts.append(f"  {tbl_name}({', '.join(cols)})")
    return "\n".join(parts)


def build_prompt(item):
    return f"""You are an expert SQL writer. Given a question and database schema, output ONE SQL query.

Question: {item['input']['question']}
Hint: {item['input'].get('evidence', '(none)') or '(none)'}

Database schema:
{format_schema(item)}

Output ONLY the SQL query (no markdown, no commentary). The query must be valid SQLite syntax.
SQL:"""


def call_gpt4o(prompt, max_tokens=512):
    body = json.dumps({
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(OPENAI_URL, data=body, headers={
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_sql(text):
    """Strip markdown fences if present, return clean SQL."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def sample_stratified():
    """Pick n=50 stratified by failure cluster + safe baseline pass."""
    clusters = defaultdict(set)
    with CLUSTERS.open() as f:
        for r in csv.DictReader(f):
            for label in r["labels"].split(";"):
                clusters[label].add(int(r["qid"]))

    plan = [
        ("missing_distinct", 15),
        ("extra_distinct", 10),
        ("extra_joins_+1", 10),
        ("structurally_identical", 5),
    ]
    sampled = {}
    for label, n in plan:
        candidates = sorted(clusters.get(label, set()))
        if len(candidates) < n:
            echo(f"  WARN: only {len(candidates)} available for '{label}' (wanted {n})")
            n = len(candidates)
        chosen = random.sample(candidates, n)
        for qid in chosen:
            sampled[qid] = label

    # 10 random "safe" qids (not in any failure cluster = set-EX pass)
    all_fail_qids = set()
    with CLUSTERS.open() as f:
        for r in csv.DictReader(f):
            all_fail_qids.add(int(r["qid"]))
    all_qids = set(range(1534))
    safe_qids = sorted(all_qids - all_fail_qids - set(sampled.keys()))
    safe_chosen = random.sample(safe_qids, 10)
    for qid in safe_chosen:
        sampled[qid] = "safe_baseline"

    return sampled


def main():
    echo(f"Loading BIRD items + sampling stratified n=50...")
    items_by_qid = {it["input"]["question_id"]: it
                    for it in (json.loads(l) for l in S7.open())}
    sampled = sample_stratified()
    echo(f"  Sampled {len(sampled)} qids across {len(set(sampled.values()))} strata\n")

    rows = []
    counts = defaultdict(lambda: {"n": 0, "set": 0, "mset": 0, "blind": 0,
                                  "exec_err": 0})
    total_prompt_tok = total_completion_tok = 0

    for i, (qid, stratum) in enumerate(sorted(sampled.items()), 1):
        it = items_by_qid[qid]
        echo(f"[{i:>2}/{len(sampled)}] qid={qid:<5} stratum={stratum:<24} ...")
        db = it["input"]["database_schema"]["db_path"]
        gold = execute(db, it["input"]["gold_sql"])
        if gold is None:
            echo(f"      gold timed out → skipping")
            continue

        prompt = build_prompt(it)
        try:
            resp = call_gpt4o(prompt)
            sql_raw = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            total_prompt_tok += usage.get("prompt_tokens", 0)
            total_completion_tok += usage.get("completion_tokens", 0)
        except Exception as e:
            echo(f"      API error: {e}")
            continue
        sql = extract_sql(sql_raw)
        pred = execute(db, sql)
        s_pass = set_match(pred, gold)
        m_pass = multiset_match(pred, gold)
        blind = s_pass and not m_pass
        exec_err = pred is None

        counts[stratum]["n"] += 1
        counts[stratum]["set"] += int(s_pass)
        counts[stratum]["mset"] += int(m_pass)
        counts[stratum]["blind"] += int(blind)
        counts[stratum]["exec_err"] += int(exec_err)

        echo(f"      set={s_pass}  mset={m_pass}  blind={blind}  err={exec_err}")
        rows.append({
            "qid": qid, "stratum": stratum,
            "set_pass": s_pass, "mset_pass": m_pass, "blind": blind,
            "exec_err": exec_err, "gpt4o_sql": sql[:300],
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cost = total_prompt_tok / 1e6 * 2.5 + total_completion_tok / 1e6 * 10.0

    echo("\n" + "=" * 78)
    echo(f"GPT-4o ({OPENAI_MODEL}) on Stratified n={sum(c['n'] for c in counts.values())}")
    echo("=" * 78)
    echo(f"  {'stratum':<24}{'n':>4}{'Set-EX':>10}{'Multiset-EX':>14}{'Blind':>8}")
    total_n = total_set = total_mset = total_blind = 0
    for s, c in counts.items():
        n = c["n"]
        if n == 0:
            continue
        total_n += n
        total_set += c["set"]; total_mset += c["mset"]; total_blind += c["blind"]
        echo(f"  {s:<24}{n:>4}  {c['set']}/{n}={c['set']/n:.0%}"
             f"  {c['mset']}/{n}={c['mset']/n:.0%}"
             f"  {c['blind']}/{n}={c['blind']/n:.0%}")
    echo("  " + "-" * 50)
    echo(f"  {'OVERALL':<24}{total_n:>4}  "
         f"{total_set}/{total_n}={total_set/total_n:.0%}  "
         f"{total_mset}/{total_n}={total_mset/total_n:.0%}  "
         f"{total_blind}/{total_n}={total_blind/total_n:.0%}")
    echo(f"\n  Set-EX > Multiset-EX gap: {(total_set - total_mset)/total_n*100:+.2f}pp")
    echo(f"\nLLM cost: {total_prompt_tok} prompt + {total_completion_tok} completion "
         f"tokens → ${cost:.4f}")


if __name__ == "__main__":
    main()
