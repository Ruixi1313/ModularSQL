#!/bin/bash
# One-shot setup for ModularSQL: install uv, sync DeepEye-SQL deps, resolve config.
# DeepEye-SQL uses `uv` (not plain pip-editable) per their official README.

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "==> Step 1: install uv (if missing)"
if ! command -v uv >/dev/null 2>&1; then
    echo "   uv not found; installing via official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # add to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "   already installed ($(uv --version))"
fi

echo
echo "==> Step 2: sync DeepEye-SQL dependencies via uv (may take 5-10 minutes)"
cd external/DeepEye-SQL
uv sync
cd "$PROJECT_ROOT"

echo
echo "==> Step 3: resolve config (substitute env vars from .env)"
# Use system python3 for this small script (no DeepEye deps needed)
python3 src/adapters/resolve_config.py \
    --template experiments/configs/bird-modularsql.toml \
    --out external/DeepEye-SQL/config/bird-modularsql.resolved.toml \
    --env .env

echo
echo "==> Done. Next: run smoke test:"
echo "   cd external/DeepEye-SQL"
echo "   CONFIG_PATH=config/bird-modularsql.resolved.toml uv run bash script/run_pipeline.sh"
