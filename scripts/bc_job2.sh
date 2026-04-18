#!/usr/bin/env bash
# BC training sequential on 8 GPUs — Job 2 (2 models)
# Order: qwen14 -> gemma (largest models get dedicated job)
# Launch on cluster with:
#   nohup bash scripts/bc_job2.sh > logs/bc_job2.log 2>&1 &
#   echo "Job2 PID: $!"

cd /mnt/queue4/rishabh/risk-routing-verifier-objective
source .venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

COMMON_ARGS=(
    --config configs/terminalbench/noisy.yaml
    --stage bc
    --train-data data/trajectories/terminalbench_noisy/bc_train.jsonl
    --val-data data/trajectories/terminalbench_noisy/bc_val.jsonl
)

COMMON_OVERRIDES=(
    policy.quantization.load_in_4bit=false
    training.bc.epochs=5
    policy.lora.r=32
    policy.lora.alpha=64
    logging.wandb_mode=disabled
)

echo "=========================================="
echo "[$(date)] Starting BC Job 2: qwen14 -> gemma"
echo "=========================================="

# 1. qwen14 (largest model first; retrain from scratch on terminalbench data)
echo "[$(date)] === qwen2.5-coder 14B ==="
accelerate launch --num_processes=8 --multi_gpu --main_process_port 29500 scripts/train_policy.py \
    "${COMMON_ARGS[@]}" \
    --output outputs/policy/terminalbench_noisy_bc_qwen_coder_14b \
    --overrides "${COMMON_OVERRIDES[@]}" policy.model_name=Qwen/Qwen2.5-Coder-14B-Instruct

# 2. gemma
echo "[$(date)] === gemma-2 9B ==="
accelerate launch --num_processes=8 --multi_gpu --main_process_port 29500 scripts/train_policy.py \
    "${COMMON_ARGS[@]}" \
    --output outputs/policy/terminalbench_noisy_bc_gemma_2_9b_it \
    --overrides "${COMMON_OVERRIDES[@]}" policy.model_name=google/gemma-2-9b-it

echo "[$(date)] === BC Job 2 complete ==="
