#!/usr/bin/env python3
"""
Resolve ${ENV_VAR} placeholders in a DeepEye-SQL TOML config using values
from .env, and write the substituted file to a runtime location.

Usage:
    python3 src/adapters/resolve_config.py \\
        --template experiments/configs/bird-modularsql.toml \\
        --out external/DeepEye-SQL/config/bird-modularsql.resolved.toml

The output TOML has every "${VAR}" replaced with the actual env value.
"""

import argparse
import os
import re
import sys
from pathlib import Path


ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def load_env(env_path: Path) -> dict:
    """Minimal .env parser (no python-dotenv dependency)."""
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


def substitute(text: str, env: dict) -> tuple[str, list[str]]:
    """Replace every ${VAR} with env[VAR]. Returns (text, missing_vars)."""
    missing = []

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name in env and env[name]:
            return env[name]
        if name in os.environ and os.environ[name]:
            return os.environ[name]
        missing.append(name)
        return match.group(0)

    return ENV_PATTERN.sub(repl, text), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="Input TOML with ${VAR} placeholders")
    parser.add_argument("--out", required=True, help="Output TOML with values substituted")
    parser.add_argument("--env", default=".env", help="Path to .env file (default: .env)")
    args = parser.parse_args()

    template_path = Path(args.template)
    out_path = Path(args.out)
    env_path = Path(args.env)

    if not template_path.exists():
        print(f"Template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    env = load_env(env_path)
    text = template_path.read_text(encoding="utf-8")
    resolved, missing = substitute(text, env)

    if missing:
        print(f"ERROR: missing env vars: {sorted(set(missing))}", file=sys.stderr)
        print(f"  add them to {env_path} or export in shell", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resolved, encoding="utf-8")
    # Restrict permissions: contains secrets
    out_path.chmod(0o600)
    print(f"Resolved {template_path} → {out_path} (permissions 600)")


if __name__ == "__main__":
    main()
