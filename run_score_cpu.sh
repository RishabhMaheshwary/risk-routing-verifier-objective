#!/usr/bin/env bash
# --------------------------------------------------------------
# CPU-only Router Feature Scoring (Phase 2: --score-only)
# --------------------------------------------------------------
#
# Runs the verifier scoring phase on CPU using cached candidates
# produced by the GPU generation phase (--generate-only) in
# run_pipeline_updated.sh.
#
# Delegates to scripts/run_router_features_humaneval.sh with
# SCORE_ONLY=true — reuses existing sharding, merging, and
# feature analysis logic.
#
# Use --no-resume after a bad/partial scoring run so each shard rewrites
# its humaneval_*_router_features_heuristic_<model>.shard_NNN.jsonl from scratch
# (candidate *.gen_cache.jsonl from generate-only is still reused).
#
# Usage:
#   bash run_score_cpu.sh --model qwen7 --benchmark humaneval
#   bash run_score_cpu.sh --model qwen7 --benchmark textworld
#   bash run_score_cpu.sh --model qwen7 --benchmark humaneval --no-resume   # full re-score (ignore partial shard *.jsonl)
#   bash run_score_cpu.sh --model llama --benchmark textworld --dry-run
#   (Cache shard count is auto-detected from *.gen_cache.jsonl — no --gpus.)
#
# Models (--model):
#   qwen7   Qwen/Qwen2.5-Coder-7B-Instruct
#   qwen14  Qwen/Qwen2.5-Coder-14B-Instruct
#   llama   meta-llama/Llama-3.1-8B-Instruct
#   gemma   google/gemma-2-9b-it
#
# --------------------------------------------------------------

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Defaults ─────────────────────────────────────────────────
MODEL_SHORT=""
BENCHMARK=""
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"
DRY_RUN=false
NO_RESUME=false

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL_SHORT="$2"; shift 2 ;;
        --benchmark)   BENCHMARK="$2"; shift 2 ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --batch-size)  ROUTER_BATCH_SIZE="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --no-resume)   NO_RESUME=true; shift ;;
        --help|-h)
            head -35 "$0" | tail -32
            exit 0
            ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_SHORT}" ]]; then
    echo "ERROR: --model is required.  Choose one of: qwen7  qwen14  llama  gemma"
    exit 1
fi

if [[ -z "${BENCHMARK}" ]]; then
    echo "ERROR: --benchmark is required. Choose one of: humaneval  textworld"
    exit 1
fi

case "${BENCHMARK}" in
    humaneval|textworld) ;;
    *)
        echo "ERROR: Unknown benchmark '${BENCHMARK}'. Choose one of: humaneval  textworld"
        exit 1
        ;;
esac

# ── Model registry (mirrors run_pipeline_updated.sh) ─────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
    [llama]="meta-llama/Llama-3.1-8B-Instruct"
    [gemma]="google/gemma-2-9b-it"
)

if [[ -z "${HF_ID[${MODEL_SHORT}]+_}" ]]; then
    echo "ERROR: Unknown model '${MODEL_SHORT}'.  Choose one of: qwen7  qwen14  llama  gemma"
    exit 1
fi

MODEL_HF="${HF_ID[${MODEL_SHORT}]}"

# ── Derived paths (benchmark-specific) ───────────────────────
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"
NOISY_TRAJECTORIES="data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl"
DPO_CHECKPOINT="outputs/policy/${BENCHMARK}_noisy_dpo_${MODEL_SHORT}/final"
# Must match run_pipeline_updated.sh ROUTER_FEATURES_FILE (not *_noisy_heuristic_*).
ROUTER_FEATURES_FILE="data/router_features/${BENCHMARK}_noisy_router_features_heuristic_${MODEL_SHORT}.jsonl"

# ── Banner ───────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  CPU Score-Only — ${MODEL_SHORT} (${MODEL_HF})"
echo "=========================================================="
echo ""
echo "  Benchmark:       ${BENCHMARK}"
echo "  Model:           ${MODEL_SHORT} (${MODEL_HF})"
echo "  DPO checkpoint:  ${DPO_CHECKPOINT}"
echo "  Trajectories:    ${NOISY_TRAJECTORIES}"
echo "  Output:          ${ROUTER_FEATURES_FILE}"
echo "  Cache shards:    (auto-detected from gen_cache files next to output)"
echo "  Batch size:      ${ROUTER_BATCH_SIZE}"
echo "  Dry run:         ${DRY_RUN}"
echo ""

# ── Delegate to run_router_features_humaneval.sh ─────────────
export POLICY_PATH="${DPO_CHECKPOINT}"
export TRAJECTORIES="${NOISY_TRAJECTORIES}"
export CONFIG="${CONFIG_NOISY}"
export OUTPUT="${ROUTER_FEATURES_FILE}"
export K="${ROUTER_K}"
export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
export SCORE_ONLY="true"
export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"
[[ "${NO_RESUME}" == true ]] && export ROUTER_NO_RESUME="true"

DRY_FLAG=""
[[ ${DRY_RUN} == true ]] && DRY_FLAG="--dry-run"

exec bash scripts/run_router_features_humaneval.sh ${DRY_FLAG}
