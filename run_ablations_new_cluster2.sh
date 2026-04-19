#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Ablation launcher (new) — Cluster 2 (8× H100)
# Models: qwen14, deepseek  |  Benchmark: terminalbench
# Runs 5 consistency-λ ablations × 2 models = 10 runs sequentially,
# each using all 8 GPUs for DPO.
#
# Usage:
#   nohup bash run_ablations_new_cluster2.sh > logs/ablations_new_cluster2.log 2>&1 &
#   bash run_ablations_new_cluster2.sh --dry-run
#   bash run_ablations_new_cluster2.sh --skip-dpo   # features + scoring only
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate 2>/dev/null || true

export PREF_SPLIT_DIR="data/trajectories/terminalbench_noisy"
export NOISY_TRAJECTORIES_OVERRIDE="updated_data/trajectories/terminalbench_noisy/trajectories.jsonl"

for model in qwen14 deepseek; do
    echo "════════════════════════════════════════════════════════"
    echo "  Launching ablations: ${model} / terminalbench"
    echo "════════════════════════════════════════════════════════"
    bash run_ablations_new.sh \
        --model "${model}" \
        --benchmark terminalbench \
        --gpus 8 \
        "$@"
done
