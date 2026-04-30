#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Cluster 2 Ablation Launcher (8× H100)
# ══════════════════════════════════════════════════════════════════
#
# Runs DPO ablations (β / consistency λ / margin) for two models
# in parallel, splitting 8 GPUs evenly: 4 per model → 4 concurrent
# DPO jobs each.
#
# All 4 terminalbench models have complete data on shared NFS:
#   deepseek · llama · qwen7 · qwen14
# Edit MODELS_A / MODELS_B below to change which two to run.
#
# Usage:
#   nohup bash scripts/ablation_job_cluster2.sh > logs/ablations_cluster2.log 2>&1 &
#   echo "PID: $!"
#
# Monitor:
#   tail -f logs/ablations_terminalbench/terminalbench_qwen14_*.log
#   tail -f logs/ablations_terminalbench/terminalbench_qwen7_*.log
#
# Results land in:
#   updated_data/router_features/terminalbench_noisy_*_abl_*.jsonl
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

# ── Models ────────────────────────────────────────────────────────
# Change these to run different pairs (deepseek / llama / qwen7 / qwen14)
MODEL_A="${MODEL_A:-qwen14}"   # GPUs 0-3
MODEL_B="${MODEL_B:-qwen7}"    # GPUs 4-7

# ── Paths ─────────────────────────────────────────────────────────
# Pref splits live in data/, not updated_data/ — export so the
# ablation script picks them up via its ${SPLIT_DIR:-...} default.
export SPLIT_DIR="data/trajectories/terminalbench_noisy"
export NOISY_TRAJECTORIES="${SPLIT_DIR}/trajectories.jsonl"

# Ablation outputs go here (created if needed)
mkdir -p updated_data/router_features
mkdir -p logs/ablations_terminalbench

# ── Shared knobs ──────────────────────────────────────────────────
GPUS_PER_MODEL=4        # 8 total / 2 models
ABLATION="${ABLATION:-all}"   # beta | lambda | margin | all
DRY_RUN="${DRY_RUN:-false}"

COMMON_ARGS=(
    --gpus          "${GPUS_PER_MODEL}"
    --gpus-per-job  1                   # 1 GPU per DPO job → 4 parallel per model
    --ablation      "${ABLATION}"
    --dpo-batch-size  1
    --dpo-grad-accum  32
)
[[ "${DRY_RUN}" == true ]] && COMMON_ARGS+=(--dry-run)

# ── Banner ────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Cluster 2 Ablation Job — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model A: ${MODEL_A}  (CUDA_VISIBLE_DEVICES=0,1,2,3)"
echo "  Model B: ${MODEL_B}  (CUDA_VISIBLE_DEVICES=4,5,6,7)"
echo "  Ablation: ${ABLATION}  |  GPUs/job: 1  |  Parallel/model: 4"
echo "  SPLIT_DIR: ${SPLIT_DIR}"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Launch model A on GPUs 0-3 ───────────────────────────────────
echo "[$(date +%H:%M:%S)] Launching ${MODEL_A} ablations on GPUs 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash run_ablations_terminalbench.sh \
        --model "${MODEL_A}" \
        "${COMMON_ARGS[@]}" \
    > "logs/ablations_terminalbench/run_${MODEL_A}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_A=$!
echo "  PID_A=${PID_A}  log: logs/ablations_terminalbench/run_${MODEL_A}_*.log"

# ── Launch model B on GPUs 4-7 ───────────────────────────────────
echo "[$(date +%H:%M:%S)] Launching ${MODEL_B} ablations on GPUs 4-7..."
CUDA_VISIBLE_DEVICES=4,5,6,7 \
    bash run_ablations_terminalbench.sh \
        --model "${MODEL_B}" \
        "${COMMON_ARGS[@]}" \
    > "logs/ablations_terminalbench/run_${MODEL_B}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_B=$!
echo "  PID_B=${PID_B}  log: logs/ablations_terminalbench/run_${MODEL_B}_*.log"

echo ""
echo "Both launched. Waiting for completion..."
echo "  Monitor A: tail -f logs/ablations_terminalbench/run_${MODEL_A}_*.log"
echo "  Monitor B: tail -f logs/ablations_terminalbench/run_${MODEL_B}_*.log"
echo "  Per-job:   tail -f logs/ablations_terminalbench/terminalbench_*.log"
echo ""

# ── Wait and report ───────────────────────────────────────────────
RC_A=0; RC_B=0
wait "${PID_A}" || RC_A=$?
echo "[$(date +%H:%M:%S)] ${MODEL_A} finished (exit ${RC_A})"

wait "${PID_B}" || RC_B=$?
echo "[$(date +%H:%M:%S)] ${MODEL_B} finished (exit ${RC_B})"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Ablation job complete — $(date '+%Y-%m-%d %H:%M:%S')"
if [[ ${RC_A} -eq 0 && ${RC_B} -eq 0 ]]; then
    echo "  Status: ALL OK"
else
    echo "  Status: FAILURES  (${MODEL_A}=${RC_A}, ${MODEL_B}=${RC_B})"
fi
echo ""
echo "  Outputs:"
ls updated_data/router_features/terminalbench_noisy_*_abl_*.jsonl 2>/dev/null \
    | while read -r f; do echo "    $f ($(wc -l < "$f") records)"; done \
    || echo "    (none yet)"
echo ""
echo "  Summary:"
cat logs/ablations_terminalbench/summary_*.txt 2>/dev/null | sort | uniq || echo "    (no summary file)"
echo "══════════════════════════════════════════════════════"

exit $(( RC_A | RC_B ))
