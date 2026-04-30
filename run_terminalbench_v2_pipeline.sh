#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Run full terminalbench v2 pipeline for multiple models.
#
# Stages: BC → Preferences → DPO → Router Features
# Data: data/trajectories/terminalbench_noisy (symlink to v2)
#
# Usage:
#   nohup bash run_terminalbench_v2_pipeline.sh > logs/terminalbench_v2_pipeline.log 2>&1 &
#
# Override models via env:
#   MODELS="llama gemma" bash run_terminalbench_v2_pipeline.sh
#   MODELS="llama" FROM_STAGE=2 bash run_terminalbench_v2_pipeline.sh
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Activate venv
source .venv/bin/activate 2>/dev/null || true

# Ensure C compiler is available (needed for Triton on some nodes)
export CC=/usr/bin/gcc

# ── Configuration ────────────────────────────────────────────────
MODELS="${MODELS:-llama gemma}"
BENCHMARK="terminalbench"
NUM_GPUS="${NUM_GPUS:-8}"
FROM_STAGE="${FROM_STAGE:-1}"
COPY_FEATURES="${COPY_FEATURES:-true}"    # copy router features to all_ablation_features_v2

FEATURES_DIR="${SCRIPT_DIR}/all_ablation_features_v2/terminalbench"

BOLD='\033[1m' NC='\033[0m'
GREEN='\033[0;32m' CYAN='\033[0;36m' RED='\033[0;31m' YELLOW='\033[0;33m'
ok()   { echo -e "${GREEN}[OK]${NC}     $1"; }
info() { echo -e "${CYAN}[INFO]${NC}   $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}   $1"; }
err()  { echo -e "${RED}[ERR]${NC}    $1"; }

FAILED=0
TOTAL_START=$(date +%s)

# ── Model tag mapping (for old BC checkpoint cleanup) ────────────
declare -A BC_DIR_TAG=(
    [qwen7]="qwen_coder_7b"
    [qwen14]="qwen_coder_14b"
    [llama]="llama_3_1_8b_instruct"
    [gemma]="gemma_2_9b_it"
    [deepseek]="deepseek_coder_6_7b"
)

cleanup_old_bc() {
    local model="$1"
    local tag="${BC_DIR_TAG[$model]}"
    local bc_dir="outputs/policy/${BENCHMARK}_noisy_bc_${tag}"
    if [[ -d "${bc_dir}" ]]; then
        warn "Removing old BC checkpoint: ${bc_dir}"
        rm -rf "${bc_dir}"
    fi
}

cleanup_old_candidates() {
    local model="$1"
    # Remove old shard files that could cause bad resumption
    local pattern="data/candidates/${BENCHMARK}_noisy_dpo_prefs_heuristic_${model}"
    local removed=0
    for f in ${pattern}.shard_*.jsonl ${pattern}_shard*.jsonl; do
        if [[ -f "$f" ]]; then
            rm -f "$f"
            removed=$((removed + 1))
        fi
    done
    # Remove old merged file too
    if [[ -f "${pattern}.jsonl" ]]; then
        rm -f "${pattern}.jsonl"
        removed=$((removed + 1))
    fi
    if [[ $removed -gt 0 ]]; then
        warn "Cleaned ${removed} old candidate files for ${model}"
    fi
}

cleanup_old_dpo() {
    local model="$1"
    local dpo_dir="outputs/policy/${BENCHMARK}_noisy_dpo_${model}"
    if [[ -d "${dpo_dir}" ]]; then
        warn "Removing old DPO checkpoint: ${dpo_dir}"
        rm -rf "${dpo_dir}"
    fi
}

cleanup_old_router_features() {
    local model="$1"
    local rf="data/router_features/${BENCHMARK}_noisy_router_features_heuristic_${model}.jsonl"
    if [[ -f "${rf}" ]]; then
        rm -f "${rf}"
        warn "Removed old router features: ${rf}"
    fi
    # Also remove shards
    for f in "${rf%.jsonl}"_shard*.jsonl; do
        [[ -f "$f" ]] && rm -f "$f"
    done
}

kill_stale_gpu_procs() {
    info "Cleaning up stale GPU processes..."
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [[ -n "${pids}" ]]; then
        echo "${pids}" | xargs -r kill -9 2>/dev/null || true
        sleep 5
    fi
    ok "GPU cleanup done"
}

run_model() {
    local MODEL="$1"
    local tag="${BC_DIR_TAG[$MODEL]}"
    local model_start=$(date +%s)

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  ${MODEL} / ${BENCHMARK} — full pipeline (stages ${FROM_STAGE}→4)${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Clean old artifacts for stages we're about to run
    if [[ ${FROM_STAGE} -le 1 ]]; then
        cleanup_old_bc "${MODEL}"
    fi
    if [[ ${FROM_STAGE} -le 2 ]]; then
        cleanup_old_candidates "${MODEL}"
    fi
    if [[ ${FROM_STAGE} -le 3 ]]; then
        cleanup_old_dpo "${MODEL}"
    fi
    if [[ ${FROM_STAGE} -le 4 ]]; then
        cleanup_old_router_features "${MODEL}"
    fi

    # Clean GPUs before each model
    kill_stale_gpu_procs

    # Run pipeline
    info "Launching run_pipeline_updated.sh --model ${MODEL} --benchmark ${BENCHMARK} --gpus ${NUM_GPUS} --from ${FROM_STAGE}"

    if bash run_pipeline_updated.sh \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --gpus "${NUM_GPUS}" \
        --from "${FROM_STAGE}"; then
        ok "${MODEL} pipeline complete"

        # Copy router features
        if [[ "${COPY_FEATURES}" == "true" ]]; then
            local src="data/router_features/${BENCHMARK}_noisy_router_features_heuristic_${MODEL}.jsonl"
            if [[ -f "${src}" ]]; then
                mkdir -p "${FEATURES_DIR}/${MODEL}"
                cp "${src}" "${FEATURES_DIR}/${MODEL}/baseline.jsonl"
                ok "Copied features → ${FEATURES_DIR}/${MODEL}/baseline.jsonl"
            else
                warn "Router features not found: ${src}"
            fi
        fi
    else
        err "${MODEL} pipeline failed (exit $?)"
        FAILED=$((FAILED + 1))
    fi

    local model_end=$(date +%s)
    local model_elapsed=$(( (model_end - model_start) / 60 ))
    info "${MODEL} elapsed: ${model_elapsed} min"
}

# ── Main ─────────────────────────────────────────────────────────
info "Models: ${MODELS}"
info "Benchmark: ${BENCHMARK}"
info "GPUs: ${NUM_GPUS}"
info "Starting from stage: ${FROM_STAGE}"
echo ""

for model in ${MODELS}; do
    run_model "${model}"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  TERMINALBENCH V2 PIPELINE COMPLETE"
echo "  Models: ${MODELS}"
echo "  Failed: ${FAILED}   Total time: ${TOTAL_ELAPSED} min"
echo "══════════════════════════════════════════════════════════"

exit ${FAILED}
