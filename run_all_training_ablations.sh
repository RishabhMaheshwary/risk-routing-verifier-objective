#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Launch ALL training-stage ablations sequentially on one 8-GPU node.
# Each combo runs no_dpo → no_bc → no_training, then moves to next.
#
# Usage:
#   nohup bash run_all_training_ablations.sh > logs/all_training_ablations.log 2>&1 &
#   bash run_all_training_ablations.sh --dry-run
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source .venv/bin/activate 2>/dev/null || true

NUM_GPUS=8
DRY_RUN_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN_FLAG="--dry-run" ;;
    esac
done

mkdir -p logs

FAILED=0
TOTAL=0

run_one() {
    local MODEL="$1" BENCH="$2"
    local ENV_ARGS=()

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  [${TOTAL}/4]  ${MODEL} / ${BENCH}  — training ablations"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # BC checkpoint override for uploaded models
    if [[ "${MODEL}" == "qwen14" && ( "${BENCH}" == "humaneval" || "${BENCH}" == "textworld" ) ]]; then
        export BC_CHECKPOINT_OVERRIDE="${SCRIPT_DIR}/qwen14_${BENCH}"
        export VLLM_MAX_MODEL_LEN=4096
        echo "[INFO] BC_CHECKPOINT_OVERRIDE=${BC_CHECKPOINT_OVERRIDE}"
        echo "[INFO] VLLM_MAX_MODEL_LEN=4096"
    else
        unset BC_CHECKPOINT_OVERRIDE 2>/dev/null || true
        unset VLLM_MAX_MODEL_LEN 2>/dev/null || true
    fi

    if bash run_training_ablations.sh \
        --model "${MODEL}" \
        --benchmark "${BENCH}" \
        --gpus "${NUM_GPUS}" \
        ${DRY_RUN_FLAG}; then
        echo "[OK] ${MODEL}/${BENCH} training ablations complete."
    else
        echo "[FAIL] ${MODEL}/${BENCH} training ablations failed (exit $?)."
        FAILED=$((FAILED + 1))
    fi
}

START_TIME=$(date +%s)

# ── 1. qwen7 / terminalbench ────────────────────────────────────
run_one qwen7  terminalbench

# ── 2. qwen14 / terminalbench ───────────────────────────────────
run_one qwen14 terminalbench

# ── 3. qwen14 / textworld ───────────────────────────────────────
run_one qwen14 textworld

# ── 4. qwen14 / humaneval ───────────────────────────────────────
run_one qwen14 humaneval

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ALL TRAINING ABLATIONS DONE"
echo "  Total: ${TOTAL}   Failed: ${FAILED}   Time: ${ELAPSED} min"
echo "══════════════════════════════════════════════════════════"

exit ${FAILED}
