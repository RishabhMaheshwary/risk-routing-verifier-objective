#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Ablation launcher (new) — Cluster 1 (8× H100)
# Models: qwen7, llama  |  Benchmark: terminalbench
# Runs 5 consistency-λ ablations × 2 models = 10 runs sequentially,
# each using all 8 GPUs for DPO.
#
# Usage:
#   nohup bash run_ablations_new_cluster1.sh > logs/ablations_new_cluster1.log 2>&1 &
#   bash run_ablations_new_cluster1.sh --dry-run
#   bash run_ablations_new_cluster1.sh --skip-dpo   # features + scoring only
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate 2>/dev/null || true

export PREF_SPLIT_DIR="data/trajectories/terminalbench_noisy"
export NOISY_TRAJECTORIES_OVERRIDE="updated_data/trajectories/terminalbench_noisy/trajectories.jsonl"

for model in qwen7 llama; do
    echo "════════════════════════════════════════════════════════"
    echo "  Launching ablations: ${model} / terminalbench"
    echo "════════════════════════════════════════════════════════"
    bash run_ablations_new.sh \
        --model "${model}" \
        --benchmark terminalbench \
        --gpus 8 \
        "$@"
done
