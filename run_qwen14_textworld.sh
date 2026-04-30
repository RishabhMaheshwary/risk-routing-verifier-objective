#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# qwen14 / textworld: pref collection → consistency-λ ablation sweep
# Usage:
#   nohup bash run_qwen14_textworld.sh > logs/qwen14_textworld.log 2>&1 &
#   bash run_qwen14_textworld.sh --dry-run
#   bash run_qwen14_textworld.sh --skip-pref   # skip pref collection
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source .venv/bin/activate 2>/dev/null || true

MODEL=qwen14
BENCHMARK=textworld
NUM_GPUS=8
SKIP_PREF=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN_FLAG="--dry-run" ;;
        --skip-pref) SKIP_PREF=true ;;
    esac
done
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"

mkdir -p logs

echo "════════════════════════════════════════════════════════"
echo "  qwen14 / textworld — full pipeline"
echo "  GPUs: ${NUM_GPUS}"
echo "════════════════════════════════════════════════════════"

# ── Stage 2: Preference Collection ───────────────────────────
if [[ "${SKIP_PREF}" == false ]]; then
    echo ""
    echo "── Stage 2: Collecting preferences ──"
    echo ""

    # vLLM for qwen14 needs max_model_len capped (large default context)
    export VLLM_MAX_MODEL_LEN=4096

    bash run_pipeline_updated.sh \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --gpus "${NUM_GPUS}" \
        --only 2 \
        ${DRY_RUN_FLAG}

    echo ""
    echo "── Preference collection done ──"
    echo ""
fi

# ── Ablation sweep ───────────────────────────────────────────
echo ""
echo "── Starting consistency-λ ablation sweep ──"
echo ""

# BC checkpoint override — points to the extracted textworld model
export BC_CHECKPOINT_OVERRIDE="${SCRIPT_DIR}/qwen14_textworld"

# Data paths: updated_data/ is symlinked from data/ so default paths work.
# The ablation script defaults to updated_data/trajectories/${BENCH}_noisy/.

bash run_ablations_new.sh \
    --model "${MODEL}" \
    --benchmark "${BENCHMARK}" \
    --gpus "${NUM_GPUS}" \
    ${DRY_RUN_FLAG}

echo ""
echo "════════════════════════════════════════════════════════"
echo "  qwen14 / textworld — COMPLETE"
echo "════════════════════════════════════════════════════════"
