#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Ablation launcher — Cluster 1 (8× H100)
# Models: llama, deepseek, qwen7
#
# 8 GPUs, 1 per ablation job = 8 parallel jobs at a time.
# 3 models × 15 ablations (5 beta + 5 lambda + 5 margin) = 45 total.
#
# Usage:
#   nohup bash run_ablations_cluster1.sh > logs/ablations_cluster1.log 2>&1 &
#   bash run_ablations_cluster1.sh --dry-run
#   bash run_ablations_cluster1.sh --ablation beta      # single ablation type
#   bash run_ablations_cluster1.sh --skip-dpo           # features + scoring only
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate 2>/dev/null || true

exec bash run_ablations_terminalbench.sh \
    --models llama deepseek qwen7 \
    --gpus 8 \
    --gpus-per-job 1 \
    --split-dir data/trajectories/terminalbench_noisy \
    --trajectories updated_data/trajectories/terminalbench_noisy/trajectories.jsonl \
    "$@"
