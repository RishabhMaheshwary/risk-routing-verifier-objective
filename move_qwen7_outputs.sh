#!/usr/bin/env bash
# Move completed qwen7 DPO checkpoints to core_llm_large to free space on queue4
set -euo pipefail

DST="/mnt/core_llm_large/rishabh/risk_output/policy"
SRC="/mnt/queue4/rishabh/risk-routing-verifier-objective/outputs/policy"

mkdir -p "${DST}"

for dir in \
    terminalbench_noisy_dpo_qwen7 \
    terminalbench_noisy_dpo_qwen7_abl_no_consistency \
    terminalbench_noisy_dpo_qwen7_abl_consistency_lambda_0.05 \
    terminalbench_noisy_dpo_qwen7_abl_consistency_lambda_0.2 \
    terminalbench_noisy_dpo_qwen7_abl_consistency_lambda_0.5 \
    terminalbench_noisy_dpo_qwen7_abl_consistency_lambda_1.0; do
    echo "Moving ${dir} ..."
    mv "${SRC}/${dir}" "${DST}/"
done

echo "Done. Freed space:"
df -h /mnt/queue4/rishabh/
