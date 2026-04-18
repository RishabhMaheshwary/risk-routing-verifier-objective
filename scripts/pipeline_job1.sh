#!/usr/bin/env bash
# Stages 2→4 (prefs + DPO + router features) for Job 1 models on 8 GPUs.
# Order: llama -> deepseek -> qwen7 (sequential, all 8 GPUs each).
#
# Launch on cluster:
#   nohup bash scripts/pipeline_job1.sh > logs/pipeline_job1.log 2>&1 &
#   echo "Pipeline Job1 PID: $!"

cd /mnt/queue4/rishabh/risk-routing-verifier-objective
source .venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

COMMON=(
    --benchmark terminalbench
    --gpus 8
    --from 2
)

echo "=========================================="
echo "[$(date)] Pipeline Job 1 stages 2-4: llama -> deepseek -> qwen7"
echo "=========================================="

echo "[$(date)] === llama: stages 2-4 ==="
bash run_pipeline_updated.sh --model llama "${COMMON[@]}"

echo "[$(date)] === deepseek: stages 2-4 ==="
bash run_pipeline_updated.sh --model deepseek "${COMMON[@]}"

echo "[$(date)] === qwen7: stages 2-4 ==="
bash run_pipeline_updated.sh --model qwen7 "${COMMON[@]}"

echo "[$(date)] === Pipeline Job 1 complete ==="
