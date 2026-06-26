#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# PixelMind Pretrain Evaluation — quick run script
# ────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/eval_pretrain.sh              # quick mode
#   bash scripts/eval_pretrain.sh full         # full eval
#   bash scripts/eval_pretrain.sh perplexity   # only perplexity
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

MODE="${1:-quick}"
WEIGHT="${2:-pretrain}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"
echo "Mode: $MODE | Weight: $WEIGHT"
echo ""

# Check weight file
WEIGHT_FILE="./out/${WEIGHT}_768.pth"
if [[ ! -f "$WEIGHT_FILE" ]]; then
    echo "ERROR: Weight file not found: $WEIGHT_FILE"
    echo "Available .pth files in ./out/:"
    ls -la ./out/*.pth 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "Found weight: $WEIGHT_FILE ($(du -h "$WEIGHT_FILE" | cut -f1))"

python -m pixelmind.eval.eval_pretrain \
    --weight "$WEIGHT" \
    --mode "$MODE" \
    "$@"

echo ""
echo "Done."
