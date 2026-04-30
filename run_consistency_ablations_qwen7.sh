#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Consistency Lambda Ablation — Qwen7 Sequential (8 GPUs)
# ══════════════════════════════════════════════════════════════════
#
# Runs 7 consistency-λ ablations for qwen7 sequentially on 8 GPUs.
# Each job: DPO training → Router feature generation → copy to ablation dir
# No backgrounding — one job finishes before the next starts.
#
# Usage:
#   nohup bash run_consistency_ablations_qwen7.sh > logs/consistency_ablations_qwen7.log 2>&1 &
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate 2>/dev/null || true
export CC=/usr/bin/gcc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_IGNORE_CUDA_DETECTION=1

MODEL="qwen7"
TOTAL_GPUS=8
MODEL_HF="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_TAG="qwen_coder_7b"
BENCH="terminalbench"
CONFIG_NOISY="configs/terminalbench/noisy.yaml"
SPLIT_DIR="data/trajectories/terminalbench_noisy"
NOISY_TRAJECTORIES="${SPLIT_DIR}/trajectories.jsonl"
BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${MODEL_TAG}/final"
GPU_IDS="0,1,2,3,4,5,6,7"
DPO_EPOCHS="${DPO_EPOCHS:-3}"
DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-32}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"
QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"
FEATURES_DIR="${SCRIPT_DIR}/all_ablation_features_v2/terminalbench"
LOGDIR="${SCRIPT_DIR}/logs/consistency_ablations_${MODEL}"
mkdir -p "${LOGDIR}"

# ── Lambda ablation values ────────────────────────────────────────
LAMBDA_ABLATIONS=(
    "no_consistency|training.consistency.enabled=false|Consistency disabled"
    "consistency_lambda_0.05|training.consistency.lambda_cons=0.05 training.consistency.enabled=true|lambda=0.05"
    "consistency_lambda_0.2|training.consistency.lambda_cons=0.2 training.consistency.enabled=true|lambda=0.2"
    "consistency_lambda_0.5|training.consistency.lambda_cons=0.5 training.consistency.enabled=true|lambda=0.5"
    "consistency_lambda_1.0|training.consistency.lambda_cons=1.0 training.consistency.enabled=true|lambda=1.0"
    "consistency_lambda_2.0|training.consistency.lambda_cons=2.0 training.consistency.enabled=true|lambda=2.0"
    "consistency_lambda_5.0|training.consistency.lambda_cons=5.0 training.consistency.enabled=true|lambda=5.0"
)

echo "══════════════════════════════════════════════════════"
echo "  ${MODEL} Consistency Lambda Ablations — Sequential"
echo "  GPUs: ${TOTAL_GPUS} | Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

# ── Validate ──────────────────────────────────────────────────────
if [[ ! -d "${BC_CHECKPOINT}" ]]; then
  echo "ERROR: BC checkpoint not found: ${BC_CHECKPOINT}"; exit 1
fi
if [[ ! -f "${SPLIT_DIR}/pref_${MODEL}_train.jsonl" ]]; then
  echo "ERROR: Preference train split not found"; exit 1
fi

# Ensure val_half exists
val_half="${SPLIT_DIR}/pref_${MODEL}_val_half.jsonl"
val_rest="${SPLIT_DIR}/pref_${MODEL}_val_rest.jsonl"
val_data="${SPLIT_DIR}/pref_${MODEL}_val.jsonl"
if [[ ! -f "${val_half}" || ! -f "${val_rest}" ]]; then
  echo "Creating 50/50 val split..."
  python - "${val_data}" "${val_half}" "${val_rest}" <<'PY'
import random, sys
src, out_a, out_b = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    rows = [ln for ln in f if ln.strip()]
rng = random.Random(42)
idxs = list(range(len(rows)))
rng.shuffle(idxs)
cut = len(rows) // 2
keep = set(idxs[:cut])
with open(out_a, "w", encoding="utf-8") as fa, open(out_b, "w", encoding="utf-8") as fb:
    for i, row in enumerate(rows):
        (fa if i in keep else fb).write(row)
print(f"  val_total={len(rows)} val_half={cut} val_rest={len(rows)-cut}")
PY
fi

TOTAL=${#LAMBDA_ABLATIONS[@]}
RUN_IDX=0

for entry in "${LAMBDA_ABLATIONS[@]}"; do
    IFS='|' read -r ABL_KEY ABL_OVERRIDES ABL_DESC <<< "${entry}"
    RUN_IDX=$(( RUN_IDX + 1 ))

    DPO_OUTPUT="outputs/policy/${BENCH}_noisy_dpo_${MODEL}_abl_${ABL_KEY}"
    DPO_CHECKPOINT="${DPO_OUTPUT}/final"
    ROUTER_FEATURES_FILE="data/router_features/${BENCH}_noisy_router_features_heuristic_${MODEL}_abl_${ABL_KEY}.jsonl"
    LOGFILE="${LOGDIR}/${BENCH}_${MODEL}_${ABL_KEY}.log"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  [${RUN_IDX}/${TOTAL}] ${MODEL} / ${ABL_KEY}"
    echo "  ${ABL_DESC}"
    echo "════════════════════════════════════════════════════"

    # ── Phase 1: DPO Training ──────────────────────────────
    if [[ -d "${DPO_CHECKPOINT}" ]]; then
        echo "[$(date +%H:%M:%S)] DPO checkpoint exists — skipping: ${DPO_CHECKPOINT}"
    else
        echo "[$(date +%H:%M:%S)] Starting DPO on GPU(s) ${GPU_IDS}..."

        DPO_ARGS=(
            scripts/train_policy.py
            --config "${CONFIG_NOISY}"
            --output "${DPO_OUTPUT}"
            --stage preference
            --pref-train-data "${SPLIT_DIR}/pref_${MODEL}_train.jsonl"
            --pref-val-data "${val_half}"
            --overrides
                "${QUANT_OVERRIDE}"
                "${COMMON_OVERRIDES}"
                "policy.model_name=${MODEL_HF}"
                "policy.lora.r=${LORA_R}"
                "policy.lora.alpha=${LORA_ALPHA}"
                "training.preference.batch_size=${DPO_BATCH_SIZE}"
                "training.preference.gradient_accumulation_steps=${DPO_GRAD_ACCUM}"
                "training.preference.concat_pairs=false"
                "training.preference.gpu_keepalive_interval=0"
                "training.preference.epochs=${DPO_EPOCHS}"
        )
        for ov in ${ABL_OVERRIDES}; do DPO_ARGS+=("${ov}"); done
        DPO_ARGS+=(--resume "${BC_CHECKPOINT}")

        MAIN_PORT=$(( 30000 + RANDOM % 10000 ))
        T0=$(date +%s)

        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
            accelerate launch --num_processes="${TOTAL_GPUS}" --main_process_port="${MAIN_PORT}" --multi_gpu \
                "${DPO_ARGS[@]}" 2>&1 | tee "${LOGFILE}" || {
            echo "[ERROR] DPO FAILED for ${ABL_KEY}"; continue
        }

        echo "[$(date +%H:%M:%S)] DPO done in $(( ($(date +%s) - T0) / 60 ))m"
    fi

    # ── Phase 2: Router Feature Generation ────────────────
    echo "[$(date +%H:%M:%S)] Generating router features..."

    T0=$(date +%s)
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        POLICY_PATH="${DPO_CHECKPOINT}" \
        TRAJECTORIES="${NOISY_TRAJECTORIES}" \
        CONFIG="${CONFIG_NOISY}" \
        OUTPUT="${ROUTER_FEATURES_FILE}" \
        K="${ROUTER_K}" \
        BATCH_SIZE="${ROUTER_BATCH_SIZE}" \
        NUM_GPUS="${TOTAL_GPUS}" \
        GENERATE_ONLY="true" \
        EXTRA_OVERRIDES="policy.model_name=${MODEL_HF} verifier.mode=heuristic verifier.heuristic.run_code=true verifier.heuristic.benchmark=${BENCH}" \
        BENCHMARK="${BENCH}" \
        bash scripts/run_router_features_humaneval.sh 2>&1 | tee -a "${LOGFILE}" || {
            echo "[ERROR] Router features FAILED for ${ABL_KEY}"; continue
        }

    echo "[$(date +%H:%M:%S)] Router features done in $(( ($(date +%s) - T0) / 60 ))m"

    # ── Phase 3: Copy to ablation dir ─────────────────────
    if [[ -f "${ROUTER_FEATURES_FILE}" ]]; then
        DEST_DIR="${FEATURES_DIR}/${MODEL}"
        mkdir -p "${DEST_DIR}"
        cp "${ROUTER_FEATURES_FILE}" "${DEST_DIR}/abl_${ABL_KEY}.jsonl"
        echo "[$(date +%H:%M:%S)] Copied -> ${DEST_DIR}/abl_${ABL_KEY}.jsonl"
    fi
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ${MODEL} ablations complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"
