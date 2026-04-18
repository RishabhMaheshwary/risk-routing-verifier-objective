#!/usr/bin/env bash
# BC training sequential on 8 GPUs — Job 1 (3 models)
# Order: llama -> deepseek -> qwen7
# Launch on cluster with:
#   nohup bash scripts/bc_job1.sh > logs/bc_job1.log 2>&1 &
#   echo "Job1 PID: $!"

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
echo "[$(date)] Starting BC Job 1: llama -> deepseek -> qwen7"
echo "=========================================="

# 1. llama
echo "[$(date)] === llama 3.1 8B ==="
accelerate launch --num_processes=8 --multi_gpu --main_process_port 29500 scripts/train_policy.py \
    "${COMMON_ARGS[@]}" \
    --output outputs/policy/terminalbench_noisy_bc_llama_3_1_8b_instruct \
    --overrides "${COMMON_OVERRIDES[@]}" policy.model_name=meta-llama/Llama-3.1-8B-Instruct

# 2. deepseek
echo "[$(date)] === deepseek-coder 6.7B ==="
accelerate launch --num_processes=8 --multi_gpu --main_process_port 29500 scripts/train_policy.py \
    "${COMMON_ARGS[@]}" \
    --output outputs/policy/terminalbench_noisy_bc_deepseek_coder_6_7b \
    --overrides "${COMMON_OVERRIDES[@]}" policy.model_name=deepseek-ai/deepseek-coder-6.7b-instruct

# 3. qwen7
echo "[$(date)] === qwen2.5-coder 7B ==="
accelerate launch --num_processes=8 --multi_gpu --main_process_port 29500 scripts/train_policy.py \
    "${COMMON_ARGS[@]}" \
    --output outputs/policy/terminalbench_noisy_bc_qwen_coder_7b \
    --overrides "${COMMON_OVERRIDES[@]}" policy.model_name=Qwen/Qwen2.5-Coder-7B-Instruct

echo "[$(date)] === BC Job 1 complete ==="
