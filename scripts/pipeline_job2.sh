#!/usr/bin/env bash
# Stages 2→4 (prefs + DPO + router features) for Job 2 models on 8 GPUs.
# Order: qwen14 -> gemma (sequential, all 8 GPUs each).
#
# Launch on cluster (after BC Job 2 finishes):
#   nohup bash scripts/pipeline_job2.sh > logs/pipeline_job2.log 2>&1 &
#   echo "Pipeline Job2 PID: $!"

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
echo "[$(date)] Pipeline Job 2 stages 2-4: qwen14 -> gemma"
echo "=========================================="

echo "[$(date)] === qwen14: stages 2-4 ==="
bash run_pipeline_updated.sh --model qwen14 "${COMMON[@]}"

echo "[$(date)] === gemma: stages 2-4 ==="
bash run_pipeline_updated.sh --model gemma "${COMMON[@]}"

echo "[$(date)] === Pipeline Job 2 complete ==="
