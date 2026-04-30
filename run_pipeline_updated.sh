#!/usr/bin/env bash
# --------------------------------------------------------------
# Unified Pipeline: BC → Preferences → DPO → Router Features
# --------------------------------------------------------------
#
# Single-model, multi-stage pipeline with heuristic verifier.
# All output files are tagged by model short name.
#
# Usage:
#   bash run_pipeline_updated.sh --model qwen7                   # full pipeline (humaneval)
#   bash run_pipeline_updated.sh --model qwen7 --benchmark textworld  # TextWorld
#   bash run_pipeline_updated.sh --model llama --from 2          # resume from stage 2
#   bash run_pipeline_updated.sh --model gemma --only 3          # run only stage 3
#   bash run_pipeline_updated.sh --model qwen14 --dry-run        # print commands
#   bash run_pipeline_updated.sh --model llama --bc-epochs 3 --lora-r 16
#
# Stages:
#   1  BC training          (behaviour cloning from teacher demos)
#   2  Collect preferences   (K candidates scored by heuristic verifier)
#   3  DPO training          (preference splitting + DPO fine-tuning)
#   4  Router features       (GENERATE_ONLY, batch_size=32)
#
# Models (--model):
#   qwen7   Qwen/Qwen2.5-Coder-7B-Instruct
#   qwen14  Qwen/Qwen2.5-Coder-14B-Instruct
#   llama   meta-llama/Llama-3.1-8B-Instruct
#   gemma   google/gemma-2-9b-it
#
# --------------------------------------------------------------

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Defaults (override via flags or env vars) ─────────────────
MODEL_SHORT="${MODEL_SHORT:-}"
BENCHMARK="humaneval"
NUM_GPUS="${NUM_GPUS:-2}"

BC_EPOCHS="${BC_EPOCHS:-5}"
BC_LR="${BC_LR:-}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-}"
BC_GRAD_ACCUM="${BC_GRAD_ACCUM:-}"

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

DPO_EPOCHS="${DPO_EPOCHS:-}"
DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-32}"
DPO_BETA="${DPO_BETA:-}"
DPO_GPU_KEEPALIVE_INTERVAL="${DPO_GPU_KEEPALIVE_INTERVAL:-0}"

PREF_K="${PREF_K:-5}"
PREF_BATCH_SIZE="${PREF_BATCH_SIZE:-16}"
PREF_MAX_NEW_TOKENS="${PREF_MAX_NEW_TOKENS:-256}"
PREF_CONSISTENCY_PAIRS="${PREF_CONSISTENCY_PAIRS:-true}"
PREF_GPU_KEEPALIVE_INTERVAL="${PREF_GPU_KEEPALIVE_INTERVAL:-0}"

ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"

FROM_STAGE=1
ONLY_STAGE=0
DRY_RUN=false

QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

# ── Argument parsing ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)             MODEL_SHORT="$2"; shift 2 ;;
        --from)              FROM_STAGE="$2"; shift 2 ;;
        --only)              ONLY_STAGE="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=true; shift ;;
        --gpus)              NUM_GPUS="$2"; shift 2 ;;
        --bc-epochs)         BC_EPOCHS="$2"; shift 2 ;;
        --bc-lr)             BC_LR="$2"; shift 2 ;;
        --bc-batch-size)     BC_BATCH_SIZE="$2"; shift 2 ;;
        --bc-grad-accum)     BC_GRAD_ACCUM="$2"; shift 2 ;;
        --lora-r)            LORA_R="$2"; shift 2 ;;
        --lora-alpha)        LORA_ALPHA="$2"; shift 2 ;;
        --dpo-epochs)        DPO_EPOCHS="$2"; shift 2 ;;
        --dpo-batch-size)    DPO_BATCH_SIZE="$2"; shift 2 ;;
        --dpo-grad-accum)    DPO_GRAD_ACCUM="$2"; shift 2 ;;
        --dpo-beta)          DPO_BETA="$2"; shift 2 ;;
        --dpo-gpu-keepalive-interval) DPO_GPU_KEEPALIVE_INTERVAL="$2"; shift 2 ;;
        --pref-K)            PREF_K="$2"; shift 2 ;;
        --pref-batch-size)   PREF_BATCH_SIZE="$2"; shift 2 ;;
        --pref-consistency-pairs) PREF_CONSISTENCY_PAIRS="$2"; shift 2 ;;
        --pref-gpu-keepalive-interval) PREF_GPU_KEEPALIVE_INTERVAL="$2"; shift 2 ;;
        --router-batch-size) ROUTER_BATCH_SIZE="$2"; shift 2 ;;
        --benchmark)         BENCHMARK="$2"; shift 2 ;;
        --help|-h)
            head -32 "$0" | tail -30
            exit 0
            ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_SHORT}" ]]; then
    echo "ERROR: --model is required.  Choose one of: qwen7  qwen14  llama  gemma"
    exit 1
fi

# ── Model registry ────────────────────────────────────────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
    [llama]="meta-llama/Llama-3.1-8B-Instruct"
    [gemma]="google/gemma-2-9b-it"
    [deepseek]="deepseek-ai/deepseek-coder-6.7b-instruct"
)

# Maps short name → directory tag used by run_bc_training.sh outputs
declare -A BC_DIR_TAG=(
    [qwen7]="qwen_coder_7b"
    [qwen14]="qwen_coder_14b"
    [llama]="llama_3_1_8b_instruct"
    [gemma]="gemma_2_9b_it"
    [deepseek]="deepseek_coder_6_7b"
)

if [[ -z "${HF_ID[${MODEL_SHORT}]+_}" ]]; then
    echo "ERROR: Unknown model '${MODEL_SHORT}'.  Choose one of: qwen7  qwen14  llama  gemma  deepseek"
    exit 1
fi

MODEL_HF="${HF_ID[${MODEL_SHORT}]}"
MODEL_TAG="${BC_DIR_TAG[${MODEL_SHORT}]}"

# On some HPC schedulers, low GPU util can trigger job termination during
# CPU-heavy preference scoring. Use a conservative keepalive default for qwen14
# unless explicitly overridden by env/flag.
if [[ "${MODEL_SHORT}" == "qwen14" && "${PREF_GPU_KEEPALIVE_INTERVAL}" == "0" ]]; then
    PREF_GPU_KEEPALIVE_INTERVAL="0.2"
fi
if [[ "${MODEL_SHORT}" == "qwen14" && "${DPO_GPU_KEEPALIVE_INTERVAL}" == "0" ]]; then
    DPO_GPU_KEEPALIVE_INTERVAL="0.2"
fi

# ── Derived paths (all tagged by model short name) ────────────
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"
SPLIT_DIR="data/trajectories/${BENCHMARK}_noisy"
NOISY_TRAJECTORIES="${SPLIT_DIR}/trajectories.jsonl"
BC_TRAIN_DATA="${SPLIT_DIR}/bc_train.jsonl"
BC_VAL_DATA="${SPLIT_DIR}/bc_val.jsonl"

BC_OUTPUT="outputs/policy/${BENCHMARK}_noisy_bc_${MODEL_TAG}"
# Prefer the BC "best/" checkpoint (lowest val loss).
#
# Note: `scripts/train_policy.py` writes BC checkpoints under:
#   ${BC_OUTPUT}/bc_<model_type_tag>/{best,final}
# Some older runs saved directly under:
#   ${BC_OUTPUT}/{best,final}
resolve_bc_checkpoint() {
    BC_CHECKPOINT="${BC_OUTPUT}/best"
    BC_CHECKPOINT_FALLBACK="${BC_OUTPUT}/final"

    shopt -s nullglob
    local nested_best=( "${BC_OUTPUT}"/bc_*/best )
    local nested_final=( "${BC_OUTPUT}"/bc_*/final )
    shopt -u nullglob

    if [[ ! -d "${BC_CHECKPOINT}" ]]; then
        # Use nested best if it exists.
        if (( ${#nested_best[@]} > 0 )); then
            BC_CHECKPOINT="${nested_best[0]}"
        fi
        # If neither direct best nor nested best exists, keep/try final.
        if [[ ! -d "${BC_CHECKPOINT}" && ${#nested_final[@]} -gt 0 ]]; then
            BC_CHECKPOINT_FALLBACK="${nested_final[0]}"
        fi
    fi

    if [[ ! -d "${BC_CHECKPOINT}" && -d "${BC_CHECKPOINT_FALLBACK}" ]]; then
        echo "  ⚠ BC best checkpoint not found; using final: ${BC_CHECKPOINT_FALLBACK}"
        BC_CHECKPOINT="${BC_CHECKPOINT_FALLBACK}"
    fi
}

resolve_bc_checkpoint

CANDIDATES_FILE="data/candidates/${BENCHMARK}_noisy_dpo_prefs_heuristic_${MODEL_SHORT}.jsonl"

PREF_PREFIX="pref_${MODEL_SHORT}"
PREF_TRAIN_DATA="${SPLIT_DIR}/${PREF_PREFIX}_train.jsonl"
PREF_VAL_DATA="${SPLIT_DIR}/${PREF_PREFIX}_val.jsonl"
PREF_TEST_DATA="${SPLIT_DIR}/${PREF_PREFIX}_test.jsonl"
PREF_VAL_HALF_DATA="${SPLIT_DIR}/${PREF_PREFIX}_val_half.jsonl"
PREF_VAL_REST_DATA="${SPLIT_DIR}/${PREF_PREFIX}_val_rest.jsonl"

DPO_OUTPUT="outputs/policy/${BENCHMARK}_noisy_dpo_${MODEL_SHORT}"
DPO_CHECKPOINT="${DPO_OUTPUT}/final"

ROUTER_FEATURES_FILE="data/router_features/${BENCHMARK}_noisy_router_features_heuristic_${MODEL_SHORT}.jsonl"

LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGDIR}"

# ── Helpers ───────────────────────────────────────────────────
should_run() {
    local stage=$1
    if [[ ${ONLY_STAGE} -ne 0 ]]; then
        [[ ${stage} -eq ${ONLY_STAGE} ]]
    else
        [[ ${stage} -ge ${FROM_STAGE} ]]
    fi
}

run_cmd() {
    local label="$1"; shift
    echo ""
    echo "========================================================"
    echo "  ${label}"
    echo "========================================================"
    echo "$ $*"
    echo ""
    if [[ ${DRY_RUN} == false ]]; then
        "$@"
    else
        echo "  [DRY RUN] Skipping execution"
    fi
}

ensure_bc_splits() {
    if [[ -f "${BC_TRAIN_DATA}" && -f "${BC_VAL_DATA}" ]]; then
        echo "  ✓ BC splits already exist"
        return
    fi
    echo "  Generating BC splits..."
    if [[ ! -f "${NOISY_TRAJECTORIES}" ]]; then
        echo "  ERROR: Trajectories missing: ${NOISY_TRAJECTORIES}"
        exit 1
    fi
    local cmd=(
        python scripts/save_static_splits.py
        --trajectories "${NOISY_TRAJECTORIES}"
        --output-dir "${SPLIT_DIR}"
        --prefix bc
        --success-only
        --max-perturbations-per-task 2
        --data-fraction 0.4
        --seed 42
    )
    echo "  $ ${cmd[*]}"
    if [[ ${DRY_RUN} == false ]]; then
        "${cmd[@]}"
    fi
    echo "  ✓ BC splits generated"
}

ensure_pref_splits() {
    if [[ -f "${PREF_TRAIN_DATA}" && -f "${PREF_VAL_DATA}" && -f "${PREF_TEST_DATA}" ]]; then
        echo "  ✓ Preference splits already exist (${MODEL_SHORT})"
        return
    fi
    echo "  Generating preference splits for ${MODEL_SHORT}..."
    python scripts/save_preference_splits.py \
        --trajectories "${NOISY_TRAJECTORIES}" \
        --preference-data "${CANDIDATES_FILE}" \
        --output-dir "${SPLIT_DIR}" \
        --prefix "${PREF_PREFIX}" \
        --seed 42
    echo "  ✓ Preference splits saved"
}

ensure_pref_val_half_split() {
    if [[ -f "${PREF_VAL_HALF_DATA}" && -f "${PREF_VAL_REST_DATA}" ]]; then
        echo "  ✓ Preference val 50/50 split already exists (${MODEL_SHORT})"
        return
    fi
    if [[ ! -f "${PREF_VAL_DATA}" ]]; then
        echo "  ERROR: Preference val data missing: ${PREF_VAL_DATA}"
        exit 1
    fi
    echo "  Creating deterministic 50/50 split from preference val data..."
    python - "${PREF_VAL_DATA}" "${PREF_VAL_HALF_DATA}" "${PREF_VAL_REST_DATA}" <<'PY'
import random
import sys

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
        if i in keep:
            fa.write(row)
        else:
            fb.write(row)

print(f"split_val_total={len(rows)} val_half={cut} val_rest={len(rows)-cut}")
PY
    echo "  ✓ Wrote val-half: ${PREF_VAL_HALF_DATA}"
    echo "  ✓ Wrote val-rest: ${PREF_VAL_REST_DATA}"
}

# ── Banner ────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  Unified Pipeline — ${MODEL_SHORT} (${MODEL_HF})"
echo "=========================================================="
echo ""
echo "  Benchmark:       ${BENCHMARK}"
echo "  Model short:     ${MODEL_SHORT}"
echo "  HF model:        ${MODEL_HF}"
echo "  BC dir tag:      ${MODEL_TAG}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Dry run:         ${DRY_RUN}"
echo ""
echo "  ── Training params ──"
echo "  BC epochs:       ${BC_EPOCHS}"
echo "  LoRA r/alpha:    ${LORA_R} / ${LORA_ALPHA}"
echo "  DPO batch/accum: ${DPO_BATCH_SIZE} / ${DPO_GRAD_ACCUM}"
echo "  DPO GPU keepalive: ${DPO_GPU_KEEPALIVE_INTERVAL}s"
echo "  Pref K:          ${PREF_K}"
echo "  Pref consistency pairs: ${PREF_CONSISTENCY_PAIRS}"
echo "  Pref GPU keepalive: ${PREF_GPU_KEEPALIVE_INTERVAL}s"
echo "  Router batch:    ${ROUTER_BATCH_SIZE}"
echo ""
echo "  ── Key paths ──"
echo "  BC checkpoint:   ${BC_CHECKPOINT}"
echo "  Candidates:      ${CANDIDATES_FILE}"
echo "  DPO checkpoint:  ${DPO_CHECKPOINT}"
echo "  Router features: ${ROUTER_FEATURES_FILE}"
echo ""
if [[ ${ONLY_STAGE} -ne 0 ]]; then
    echo "  Running ONLY stage ${ONLY_STAGE}"
else
    echo "  Running stages ${FROM_STAGE}..4"
fi
echo ""

# --------------------------------------------------------------
# Stage 1: BC Training
# --------------------------------------------------------------
if should_run 1; then
    echo "──────────────────────────────────────────────────────────"
    echo "  Stage 1: Behavior Cloning — ${MODEL_SHORT}"
    echo "──────────────────────────────────────────────────────────"

    ensure_bc_splits

    LOGFILE="${LOGDIR}/bc_${BENCHMARK}_${MODEL_SHORT}.log"

    BC_OVERRIDES=(
        "${QUANT_OVERRIDE}"
        "policy.model_name=${MODEL_HF}"
        "training.bc.epochs=${BC_EPOCHS}"
        "policy.lora.r=${LORA_R}"
        "policy.lora.alpha=${LORA_ALPHA}"
        "${COMMON_OVERRIDES}"
    )
    [[ -n "${BC_LR}" ]]         && BC_OVERRIDES+=("training.bc.learning_rate=${BC_LR}")
    [[ -n "${BC_BATCH_SIZE}" ]]  && BC_OVERRIDES+=("training.bc.batch_size=${BC_BATCH_SIZE}")
    [[ -n "${BC_GRAD_ACCUM}" ]]  && BC_OVERRIDES+=("training.bc.gradient_accumulation_steps=${BC_GRAD_ACCUM}")

    if [[ ${NUM_GPUS} -gt 1 ]]; then
        CMD=(
            accelerate launch
            --num_processes="${NUM_GPUS}" --multi_gpu
            scripts/train_policy.py
            --config "${CONFIG_NOISY}"
            --output "${BC_OUTPUT}"
            --stage bc
            --train-data "${BC_TRAIN_DATA}"
            --val-data "${BC_VAL_DATA}"
            --overrides "${BC_OVERRIDES[@]}"
        )
    else
        CMD=(
            python scripts/train_policy.py
            --config "${CONFIG_NOISY}"
            --output "${BC_OUTPUT}"
            --stage bc
            --train-data "${BC_TRAIN_DATA}"
            --val-data "${BC_VAL_DATA}"
            --overrides "${BC_OVERRIDES[@]}"
        )
    fi

    START_T=$(date +%s)
    BC_RC=0
    run_cmd "BC Training: ${MODEL_SHORT} (${MODEL_HF})" \
        "${CMD[@]}" 2>&1 | tee -a "${LOGFILE}" || BC_RC=$?
    ELAPSED=$(( $(date +%s) - START_T ))

    if [[ ${BC_RC} -ne 0 ]]; then
        if [[ -d "${BC_CHECKPOINT}" ]]; then
            echo "  ⚠ BC process exited with code ${BC_RC}, but checkpoint exists — continuing"
        else
            echo "  ERROR: BC failed (exit ${BC_RC}) and no checkpoint at ${BC_CHECKPOINT}"
            exit 1
        fi
    fi
    # Re-resolve after training because BC checkpoints may be created in a
    # nested bc_*/{best,final} directory during Stage 1.
    resolve_bc_checkpoint
    echo "  ✓ BC done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
    echo "  checkpoint: ${BC_CHECKPOINT}"
fi

# Validate BC checkpoint exists before proceeding to later stages
if [[ ${DRY_RUN} == false ]] && should_run 2 && [[ ! -d "${BC_CHECKPOINT}" ]]; then
    if [[ ${ONLY_STAGE} -ne 0 && ${ONLY_STAGE} -lt 2 ]]; then
        : # not needed
    else
        echo "ERROR: BC checkpoint not found: ${BC_CHECKPOINT}"
        echo "  Run stage 1 first, or verify the path."
        exit 1
    fi
fi

# --------------------------------------------------------------
# Stage 2: Collect Preference Data (Heuristic Verifier)
# --------------------------------------------------------------
if should_run 2; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Stage 2: Collect Preferences — ${MODEL_SHORT}"
    echo "──────────────────────────────────────────────────────────"

    LOGFILE="${LOGDIR}/preferences_${BENCHMARK}_${MODEL_SHORT}.log"
    mkdir -p "$(dirname "${CANDIDATES_FILE}")"

    PREF_ARGS=(
        --config "${CONFIG_NOISY}"
        --policy-path "${BC_CHECKPOINT}"
        --trajectories "${NOISY_TRAJECTORIES}"
        --output "${CANDIDATES_FILE}"
        --K "${PREF_K}"
        --batch-size "${PREF_BATCH_SIZE}"
        --max-new-tokens "${PREF_MAX_NEW_TOKENS}"
        --gpu-keepalive-interval "${PREF_GPU_KEEPALIVE_INTERVAL}"
        --overrides
            "${QUANT_OVERRIDE}"
            "${COMMON_OVERRIDES}"
            "verifier.mode=heuristic"
            "verifier.heuristic.run_code=true"
            "verifier.heuristic.benchmark=${BENCHMARK}"
            "policy.model_name=${MODEL_HF}"
    )
    if [[ "${PREF_CONSISTENCY_PAIRS}" == "false" ]]; then
        PREF_ARGS+=(--no-consistency-pairs)
    fi

    # FlashInfer (used by vLLM for some models, e.g. gemma-2) JIT-compiles
    # attention kernels and needs CUDA_HOME with nvcc + headers. On clusters
    # without /usr/local/cuda, point to a stub we built from conda packages.
    if [[ -z "${CUDA_HOME:-}" ]]; then
        _cuda_stub="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}/cuda_home"
        _cuda_stub="$(cd "${_cuda_stub}" 2>/dev/null && pwd || true)"
        if [[ -x "${_cuda_stub}/bin/nvcc" ]]; then
            export CUDA_HOME="${_cuda_stub}"
            export PATH="${_cuda_stub}/bin:${PATH}"
        fi
    fi

    # Triton JIT-compiles CUDA kernels via a C compiler; ensure CC is set to
    # a full absolute path so worker subprocesses find it regardless of PATH.
    # Priority: existing CC > /usr/bin/gcc (full path, bypasses PATH lookup)
    if [[ -z "${CC:-}" ]]; then
        if [[ -x "/usr/bin/gcc" ]]; then
            export CC="/usr/bin/gcc"
        else
            _gcc=$(command -v gcc 2>/dev/null || true)
            export CC="${_gcc:-gcc}"
        fi
    fi

    START_T=$(date +%s)
    PREF_RC=0
    run_cmd "Collect preferences: ${MODEL_SHORT} (heuristic verifier, K=${PREF_K})" \
        bash scripts/launch_candidates.sh "${NUM_GPUS}" \
            "${PREF_ARGS[@]}" \
        2>&1 | tee -a "${LOGFILE}" || PREF_RC=$?
    ELAPSED=$(( $(date +%s) - START_T ))

    if [[ ${PREF_RC} -ne 0 ]]; then
        if [[ -f "${CANDIDATES_FILE}" ]] && [[ $(wc -l < "${CANDIDATES_FILE}") -gt 0 ]]; then
            echo "  ⚠ Preference collection exited with code ${PREF_RC}, but candidates file exists — continuing"
        else
            echo "  ERROR: Preference collection failed (exit ${PREF_RC}) and no output at ${CANDIDATES_FILE}"
            exit 1
        fi
    fi
    echo "  ✓ Preference collection done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
    echo "  candidates: ${CANDIDATES_FILE}"
fi

# --------------------------------------------------------------
# Stage 3: Preference Splitting + DPO Training
# --------------------------------------------------------------
if should_run 3; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Stage 3: DPO Training — ${MODEL_SHORT}"
    echo "──────────────────────────────────────────────────────────"

    # 3a. Ensure preference splits exist
    if [[ ${DRY_RUN} == false ]]; then
        ensure_pref_splits
        ensure_pref_val_half_split
    fi

    LOGFILE="${LOGDIR}/dpo_${BENCHMARK}_${MODEL_SHORT}.log"

    # qwen14 (14B) needs 4-bit loading for DPO: the policy + frozen reference
    # model together exceed 80 GB HBM in bf16. Other models are fine in bf16.
    DPO_QUANT_OVERRIDE="${QUANT_OVERRIDE}"
    if [[ "${MODEL_SHORT}" == "qwen14" ]]; then
        DPO_QUANT_OVERRIDE="policy.quantization.load_in_4bit=true"
    fi

    # Large-vocab models accumulate huge logits tensors during DPO.
    # Consistency JSD creates 6× (batch, seq, vocab) tensors simultaneously:
    #   qwen14: seq=4096, vocab=151936 → ~12 GB; plus deepcopy of 4-bit ref
    #           may dequantize to bf16 (~29 GB) → 71 GiB OOM observed.
    #   gemma:  seq=4096, vocab=256000 → ~20 GB for consistency alone.
    # Fix: disable consistency for these two models.  qwen14 also drops the
    # reference model to avoid the bf16-deepcopy memory spike.
    DPO_EXTRA_OVERRIDES=()
    if [[ "${MODEL_SHORT}" == "qwen14" ]]; then
        DPO_EXTRA_OVERRIDES+=(
            "training.consistency.enabled=false"
            "training.preference.use_reference_model=false"
        )
    elif [[ "${MODEL_SHORT}" == "gemma" ]]; then
        DPO_EXTRA_OVERRIDES+=("training.consistency.enabled=false")
    fi

    DPO_ARGS=(
        scripts/train_policy.py
        --config "${CONFIG_NOISY}"
        --output "${DPO_OUTPUT}"
        --stage preference
        --resume "${BC_CHECKPOINT}"
        --pref-train-data "${PREF_TRAIN_DATA}"
        --pref-val-data "${PREF_VAL_HALF_DATA}"
        --overrides
            "${DPO_QUANT_OVERRIDE}"
            "${COMMON_OVERRIDES}"
            "policy.model_name=${MODEL_HF}"
            "policy.lora.r=${LORA_R}"
            "policy.lora.alpha=${LORA_ALPHA}"
            "training.preference.batch_size=${DPO_BATCH_SIZE}"
            "training.preference.gradient_accumulation_steps=${DPO_GRAD_ACCUM}"
            "training.preference.concat_pairs=false"
            "training.preference.gpu_keepalive_interval=${DPO_GPU_KEEPALIVE_INTERVAL}"
    )
    [[ -n "${DPO_EPOCHS}" ]] && DPO_ARGS+=("training.preference.epochs=${DPO_EPOCHS}")
    [[ -n "${DPO_BETA}" ]]   && DPO_ARGS+=("training.preference.beta=${DPO_BETA}")
    [[ ${#DPO_EXTRA_OVERRIDES[@]} -gt 0 ]] && DPO_ARGS+=("${DPO_EXTRA_OVERRIDES[@]}")

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # DeepSpeed checks CUDA_HOME at import time; skip the check when nvcc is
    # absent (runtime-only CUDA install) — torch.cuda still works fine.
    export DS_IGNORE_CUDA_DETECTION=1

    START_T=$(date +%s)
    DPO_RC=0
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        run_cmd "DPO: ${MODEL_SHORT} (${NUM_GPUS} GPUs, resume from BC)" \
            accelerate launch --num_processes="${NUM_GPUS}" --multi_gpu \
                "${DPO_ARGS[@]}" \
            2>&1 | tee -a "${LOGFILE}" || DPO_RC=$?
    else
        run_cmd "DPO: ${MODEL_SHORT} (single GPU, resume from BC)" \
            python "${DPO_ARGS[@]}" \
            2>&1 | tee -a "${LOGFILE}" || DPO_RC=$?
    fi
    ELAPSED=$(( $(date +%s) - START_T ))

    if [[ ${DPO_RC} -ne 0 ]]; then
        if [[ -d "${DPO_CHECKPOINT}" ]]; then
            echo "  ⚠ accelerate exited with code ${DPO_RC}, but DPO checkpoint exists — continuing"
        else
            echo "  ERROR: DPO failed (exit ${DPO_RC}) and no checkpoint at ${DPO_CHECKPOINT}"
            exit 1
        fi
    fi
    echo "  ✓ DPO done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
    echo "  checkpoint: ${DPO_CHECKPOINT}"
fi

# --------------------------------------------------------------
# Stage 4: Generate Router Features (GENERATE_ONLY)
# --------------------------------------------------------------
if should_run 4; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Stage 4: Router Features — ${MODEL_SHORT}"
    echo "──────────────────────────────────────────────────────────"

    LOGFILE="${LOGDIR}/router_features_${BENCHMARK}_${MODEL_SHORT}.log"
    mkdir -p "$(dirname "${ROUTER_FEATURES_FILE}")"

    VERIFIER_OVERRIDE="verifier.mode=heuristic verifier.heuristic.run_code=true verifier.heuristic.benchmark=${BENCHMARK}"

    export VERIFIER_OVERRIDE
    export POLICY_PATH="${DPO_CHECKPOINT}"
    export TRAJECTORIES="${NOISY_TRAJECTORIES}"
    export CONFIG="${CONFIG_NOISY}"
    export OUTPUT="${ROUTER_FEATURES_FILE}"
    export K="${ROUTER_K}"
    export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
    export NUM_GPUS="${NUM_GPUS}"
    export GENERATE_ONLY="true"
    export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"

    START_T=$(date +%s)
    run_cmd "Router features: ${MODEL_SHORT} (GENERATE_ONLY, batch=${ROUTER_BATCH_SIZE})" \
        bash scripts/run_router_features_humaneval.sh \
        2>&1 | tee -a "${LOGFILE}"
    ELAPSED=$(( $(date +%s) - START_T ))
    echo "  ✓ Router features done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
    echo "  features: ${ROUTER_FEATURES_FILE}"
fi

# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------
echo ""
echo "=========================================================="
echo "  Pipeline Complete — ${MODEL_SHORT} (${MODEL_HF})"
echo "=========================================================="
echo ""
echo "  Artifacts:"
echo "    BC checkpoint:     ${BC_CHECKPOINT}"
echo "    Candidates:        ${CANDIDATES_FILE}"
echo "    Pref splits:       ${PREF_TRAIN_DATA}"
echo "                       ${PREF_VAL_DATA}"
echo "    DPO checkpoint:    ${DPO_CHECKPOINT}"
echo "    Router features:   ${ROUTER_FEATURES_FILE}"
echo ""
echo "=========================================================="
