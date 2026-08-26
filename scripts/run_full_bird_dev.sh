#!/bin/bash
# Safe launcher for the full BIRD-Dev (1534) pipeline run.
# - Verifies workspace_full/ state and lets you choose clean-start vs resume.
# - Resolves the config (env-substitution) from the .env file.
# - Wraps the actual pipeline in caffeinate so the machine cannot sleep mid-run.
#
# Usage:
#   bash scripts/run_full_bird_dev.sh              # default behavior
#   bash scripts/run_full_bird_dev.sh --fresh      # delete workspace_full first
#   bash scripts/run_full_bird_dev.sh --resume     # keep workspace_full, resume

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DEEPEYE="$PROJECT_ROOT/external/DeepEye-SQL"
WORKSPACE="$DEEPEYE/workspace_full"
CONFIG_TEMPLATE="experiments/configs/bird-modularsql-full.toml"
RESOLVED_CONFIG="$DEEPEYE/config/bird-modularsql-full.resolved.toml"

# Parse flags
MODE="auto"
LAUNCH=false
for arg in "$@"; do
    case "$arg" in
        --fresh)  MODE="fresh" ;;
        --resume) MODE="resume" ;;
        --go)     LAUNCH=true ;;       # required to actually start the pipeline
    esac
done

echo "===================================================================="
echo "Full BIRD-Dev (1534) launcher"
echo "===================================================================="

# 1. Workspace state
if [ -d "$WORKSPACE" ]; then
    item_count=$(find "$WORKSPACE" -name "items.jsonl" -path "*.snapshot.data/*" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  workspace_full/ exists, total snapshot entries: ${item_count:-0}"
    if [ "$MODE" = "auto" ]; then
        echo "  ⚠ Existing workspace detected. Re-run with --fresh OR --resume."
        echo "    --fresh:  delete workspace_full/, start from scratch"
        echo "    --resume: keep workspace_full/, continue from last checkpoint"
        exit 2
    fi
    if [ "$MODE" = "fresh" ]; then
        echo "  → Removing workspace_full/ (clean start)"
        rm -rf "$WORKSPACE"
    else
        echo "  → Resuming from workspace_full/"
    fi
else
    echo "  workspace_full/ does not exist → clean start"
fi

# 2. Resolve config
echo ""
echo "  Resolving config..."
python3 src/adapters/resolve_config.py \
    --template "$CONFIG_TEMPLATE" \
    --out "$RESOLVED_CONFIG" \
    --env .env

# 3. Sanity check the resolved config
echo ""
echo "  Resolved config preview (paths):"
grep -E "^max_samples|save_path|store_root_path" "$RESOLVED_CONFIG" | head -10

# 4. Disk space check
echo ""
echo "  Disk space:"
df -h "$PROJECT_ROOT" | tail -1

# 5. Confirm before running
echo ""
echo "  Estimated runtime: 3-5 hours"
echo "  Estimated cost:    \$13 (Qwen3-Coder-30B-A3B on OpenRouter)"
echo "  Wrapped in:        caffeinate -dims (system stays awake)"
echo ""
if [ "$LAUNCH" != "true" ]; then
    echo "  ⓘ Preflight checks complete. Re-run with --go to actually launch:"
    echo "      bash scripts/run_full_bird_dev.sh --fresh --go     # clean start"
    echo "      bash scripts/run_full_bird_dev.sh --resume --go    # resume existing"
    exit 0
fi

# 6. Launch
LOG_FILE="$DEEPEYE/logs/full_bird_dev_$(date +'%Y%m%d_%H%M%S').log"
mkdir -p "$DEEPEYE/logs"
echo ""
echo "===================================================================="
echo "  Launching with caffeinate. Log: $LOG_FILE"
echo "===================================================================="
cd "$DEEPEYE"
caffeinate -dims bash -c "CONFIG_PATH=config/bird-modularsql-full.resolved.toml bash script/run_pipeline.sh" 2>&1 | tee "$LOG_FILE"
