#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Launch qwen7 training-stage ablations (no_dpo, no_bc, no_training)
# for humaneval and textworld on one 8-GPU node.
#
# Prerequisites (done manually before launch):
#   1. unzip qwen7_humaneval.zip && unzip qwen7_textworld.zip
#   2. Symlink pref data: pref_qwen7_* → pref_qwen14_*
#
# Usage:
#   nohup bash run_qwen7_training_ablations.sh > logs/qwen7_training_ablations.log 2>&1 &
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source .venv/bin/activate 2>/dev/null || true

NUM_GPUS=8
FAILED=0
START_TIME=$(date +%s)

BOLD='\033[1m' NC='\033[0m'
GREEN='\033[0;32m' CYAN='\033[0;36m' RED='\033[0;31m'
ok()   { echo -e "${GREEN}[OK]${NC}     $1"; }
info() { echo -e "${CYAN}[INFO]${NC}   $1"; }
err()  { echo -e "${RED}[ERR]${NC}    $1"; }

run_one() {
    local MODEL="$1" BENCH="$2"
    local BC_CKPT="${SCRIPT_DIR}/qwen7_${BENCH}"

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  ${MODEL} / ${BENCH} — training ablations${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    info "BC_CHECKPOINT_OVERRIDE=${BC_CKPT}"
    info "VLLM_MAX_MODEL_LEN=4096"

    export BC_CHECKPOINT_OVERRIDE="${BC_CKPT}"
    export VLLM_MAX_MODEL_LEN=4096

    if bash run_training_ablations.sh \
        --model "${MODEL}" \
        --benchmark "${BENCH}" \
        --gpus "${NUM_GPUS}"; then
        ok "${MODEL}/${BENCH} training ablations complete."
    else
        err "${MODEL}/${BENCH} training ablations failed (exit $?)."
        FAILED=$((FAILED + 1))
    fi

    unset BC_CHECKPOINT_OVERRIDE VLLM_MAX_MODEL_LEN
}

# ── 1. qwen7 / humaneval ────────────────────────────────────────
run_one qwen7 humaneval

# ── 2. qwen7 / textworld ────────────────────────────────────────
run_one qwen7 textworld

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  QWEN7 TRAINING ABLATIONS DONE"
echo "  Failed: ${FAILED}   Time: ${ELAPSED} min"
echo "══════════════════════════════════════════════════════════"

exit ${FAILED}
