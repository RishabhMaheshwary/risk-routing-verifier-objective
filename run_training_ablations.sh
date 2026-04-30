#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Training Stage Ablations: no_dpo, no_bc, no_training
# ══════════════════════════════════════════════════════════════════
#
# Three ablation conditions that test which training stages matter:
#   no_dpo       — BC checkpoint used directly as policy (skip DPO)
#   no_bc        — Base HF model → DPO (skip BC, train LoRA from scratch)
#   no_training  — Base HF model used directly as policy (skip both)
#
# Pipeline per condition:
#   no_dpo:       BC ckpt → Router Features (GPU) → CPU Scoring
#   no_bc:        Base HF → DPO → Router Features (GPU) → CPU Scoring
#   no_training:  Base HF → Router Features (GPU) → CPU Scoring
#
# Usage:
#   bash run_training_ablations.sh --model qwen7 --benchmark terminalbench --gpus 8
#   bash run_training_ablations.sh --model qwen14 --benchmark textworld --gpus 8
#   bash run_training_ablations.sh --model qwen14 --benchmark humaneval --gpus 8
#   bash run_training_ablations.sh --dry-run --model qwen7 --benchmark terminalbench
#   bash run_training_ablations.sh --only no_dpo --model qwen7 --benchmark terminalbench --gpus 8
#
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source .venv/bin/activate 2>/dev/null || true

# ── Defaults ─────────────────────────────────────────────────
FILTER_MODEL=""
FILTER_BENCHMARK=""
NUM_GPUS="${NUM_GPUS:-8}"
DRY_RUN=false
ONLY_CONDITION=""   # run all 3 by default
ROUTER_K=5
ROUTER_BATCH_SIZE=32

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       FILTER_MODEL="$2"; shift 2 ;;
        --benchmark)   FILTER_BENCHMARK="$2"; shift 2 ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --only)        ONLY_CONDITION="$2"; shift 2 ;;
        --help|-h)     head -25 "$0" | tail -22; exit 0 ;;
        *)             echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${FILTER_MODEL}" ]]; then
    echo "ERROR: --model is required (qwen7, qwen14)"; exit 1
fi
if [[ -z "${FILTER_BENCHMARK}" ]]; then
    echo "ERROR: --benchmark is required (terminalbench, textworld, humaneval)"; exit 1
fi

# ── Model registry ───────────────────────────────────────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
)

declare -A BC_DIR_TAG=(
    [qwen7]="qwen_coder_7b"
    [qwen14]="qwen_coder_14b"
)

if [[ -z "${HF_ID[${FILTER_MODEL}]+_}" ]]; then
    echo "ERROR: Unknown model '${FILTER_MODEL}'. Choose: qwen7, qwen14"; exit 1
fi

MODEL_SHORT="${FILTER_MODEL}"
MODEL_HF="${HF_ID[${MODEL_SHORT}]}"
BENCH="${FILTER_BENCHMARK}"

# ── Formatting helpers ───────────────────────────────────────
BOLD='\033[1m' NC='\033[0m'
RED='\033[0;31m' GREEN='\033[0;32m' CYAN='\033[0;36m' YELLOW='\033[1;33m'
header() { echo -e "\n${BOLD}══════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $1${NC}"; echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"; }
ok()     { echo -e "${GREEN}[OK]${NC}     $1"; }
info()   { echo -e "${CYAN}[INFO]${NC}   $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}   $1"; }
err()    { echo -e "${RED}[ERR]${NC}    $1"; }
run_cmd() { local desc="$1"; shift; echo ""; echo -e "${BOLD}────────────────────────────────────────────────────${NC}"; echo -e "  ${desc}"; echo -e "${BOLD}────────────────────────────────────────────────────${NC}"; echo "$ $*"; echo ""; if [[ ${DRY_RUN} == true ]]; then echo "  [DRY RUN] Skipping execution"; return 0; fi; "$@"; }

# ── Paths ────────────────────────────────────────────────────
CONFIG_NOISY="configs/${BENCH}/noisy.yaml"

# Trajectories: updated_data/ for humaneval/textworld, data/ for terminalbench
if [[ "${BENCH}" == "terminalbench" ]]; then
    NOISY_TRAJECTORIES="${NOISY_TRAJECTORIES_OVERRIDE:-updated_data/trajectories/${BENCH}_noisy/trajectories.jsonl}"
    SPLIT_DIR="${PREF_SPLIT_DIR:-data/trajectories/${BENCH}_noisy}"
else
    NOISY_TRAJECTORIES="updated_data/trajectories/${BENCH}_noisy/trajectories.jsonl"
    SPLIT_DIR="updated_data/trajectories/${BENCH}_noisy"
fi

# BC checkpoint: respect BC_CHECKPOINT_OVERRIDE, else look in standard locations
if [[ -n "${BC_CHECKPOINT_OVERRIDE:-}" && -d "${BC_CHECKPOINT_OVERRIDE}" ]]; then
    BC_CHECKPOINT="${BC_CHECKPOINT_OVERRIDE}"
else
    BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${BC_DIR_TAG[${MODEL_SHORT}]}/best"
    if [[ ! -d "${BC_CHECKPOINT}" ]]; then
        BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${BC_DIR_TAG[${MODEL_SHORT}]}/final"
    fi
fi

PREF_PREFIX="pref_${MODEL_SHORT}"
PREF_TRAIN_DATA="${SPLIT_DIR}/${PREF_PREFIX}_train.jsonl"
PREF_VAL_HALF_DATA="${SPLIT_DIR}/${PREF_PREFIX}_val_half.jsonl"

LOGDIR="logs/training_ablations"
mkdir -p "${LOGDIR}"

# DPO settings
LORA_R=32
LORA_ALPHA=64
DPO_BATCH_SIZE=1
DPO_GRAD_ACCUM=32
QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

# GPU keepalive for qwen14
GPU_KEEPALIVE="0"
[[ "${MODEL_SHORT}" == "qwen14" ]] && GPU_KEEPALIVE="0.2"

# vLLM max model len (qwen14 has 128k default context, needs capping)
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

# DPO checkpoint resolver (same as run_ablations_new.sh)
resolve_dpo_checkpoint() {
    local base="$1"
    for sub in preference/best preference/final final best; do
        if [[ -d "${base}/${sub}" ]]; then echo "${base}/${sub}"; return 0; fi
    done
    [[ -d "${base}" ]] && echo "${base}" && return 0
    return 1
}

# ── Ablation conditions ──────────────────────────────────────
# Each entry: KEY|POLICY_SOURCE|RUN_DPO|DESCRIPTION
#   POLICY_SOURCE: "bc" = BC checkpoint, "base" = base HF model, "dpo" = DPO output
ABLATION_CONDITIONS=(
    "no_dpo|bc|false|Skip DPO: use BC checkpoint directly as policy"
    "no_bc|dpo|true|Skip BC: DPO from base HF model (no BC init)"
    "no_training|base|false|Skip both: use base HF model directly as policy"
)

# Filter if --only specified
if [[ -n "${ONLY_CONDITION}" ]]; then
    FILTERED=()
    for entry in "${ABLATION_CONDITIONS[@]}"; do
        IFS='|' read -r KEY _ _ _ <<< "${entry}"
        [[ "${KEY}" == "${ONLY_CONDITION}" ]] && FILTERED+=("${entry}")
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "ERROR: Unknown condition '${ONLY_CONDITION}'. Choose: no_dpo, no_bc, no_training"; exit 1
    fi
    ABLATION_CONDITIONS=("${FILTERED[@]}")
fi

TOTAL_RUNS=${#ABLATION_CONDITIONS[@]}
SUCCEEDED_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()

# ── Banner ───────────────────────────────────────────────────
header "Training Stage Ablations"
echo "  Model:          ${MODEL_SHORT} (${MODEL_HF})"
echo "  Benchmark:      ${BENCH}"
echo "  GPUs:           ${NUM_GPUS}"
echo "  BC checkpoint:  ${BC_CHECKPOINT}"
echo "  Trajectories:   ${NOISY_TRAJECTORIES}"
echo "  Conditions:     ${TOTAL_RUNS} (${ABLATION_CONDITIONS[*]%%|*})"
echo "  Dry run:        ${DRY_RUN}"
echo ""

# ── Validate prerequisites ───────────────────────────────────
if [[ ! -f "${CONFIG_NOISY}" ]]; then
    err "Config not found: ${CONFIG_NOISY}"; exit 1
fi
if [[ ! -f "${NOISY_TRAJECTORIES}" ]]; then
    err "Trajectories not found: ${NOISY_TRAJECTORIES}"; exit 1
fi

RUN_IDX=0

for entry in "${ABLATION_CONDITIONS[@]}"; do
    IFS='|' read -r ABL_KEY POLICY_SOURCE RUN_DPO ABL_DESC <<< "${entry}"
    RUN_IDX=$(( RUN_IDX + 1 ))

    DPO_OUTPUT="outputs/policy/${BENCH}_noisy_dpo_${MODEL_SHORT}_abl_${ABL_KEY}"
    ROUTER_FEATURES_FILE="updated_data/router_features/${BENCH}_noisy_router_features_heuristic_${MODEL_SHORT}_abl_${ABL_KEY}.jsonl"
    LOGFILE="${LOGDIR}/${BENCH}_${MODEL_SHORT}_${ABL_KEY}.log"

    header "[${RUN_IDX}/${TOTAL_RUNS}] ${MODEL_SHORT} / ${BENCH} / ${ABL_KEY}"
    echo "  ${ABL_DESC}"
    echo "  Policy source:   ${POLICY_SOURCE}"
    echo "  Run DPO:         ${RUN_DPO}"
    echo "  DPO output:      ${DPO_OUTPUT}"
    echo "  Router features: ${ROUTER_FEATURES_FILE}"
    echo "  Log:             ${LOGFILE}"
    echo ""

    # ──────────────────────────────────────────────────
    # Determine the effective policy path for features
    # ──────────────────────────────────────────────────
    EFFECTIVE_POLICY=""

    # ──────────────────────────────────────────────────
    # Phase 1: DPO Training (only for no_bc)
    # ──────────────────────────────────────────────────
    if [[ "${RUN_DPO}" == "true" ]]; then
        # no_bc: DPO from base model (no --resume flag)
        if [[ ! -f "${PREF_TRAIN_DATA}" ]]; then
            err "Preference train data missing: ${PREF_TRAIN_DATA}"
            FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [no pref data]")
            continue
        fi

        DPO_CHECKPOINT_PATH="$(resolve_dpo_checkpoint "${DPO_OUTPUT}" || echo "")"
        if [[ -n "${DPO_CHECKPOINT_PATH}" && -d "${DPO_CHECKPOINT_PATH}" ]]; then
            info "DPO checkpoint already exists: ${DPO_CHECKPOINT_PATH} — skipping DPO"
        else
            # Note: NO --resume flag — starts LoRA from scratch on base model
            DPO_ARGS=(
                scripts/train_policy.py
                --config "${CONFIG_NOISY}"
                --output "${DPO_OUTPUT}"
                --stage preference
                --pref-train-data "${PREF_TRAIN_DATA}"
                --pref-val-data "${PREF_VAL_HALF_DATA}"
                --overrides
                    "${QUANT_OVERRIDE}"
                    "${COMMON_OVERRIDES}"
                    "policy.model_name=${MODEL_HF}"
                    "policy.lora.r=${LORA_R}"
                    "policy.lora.alpha=${LORA_ALPHA}"
                    "training.preference.batch_size=${DPO_BATCH_SIZE}"
                    "training.preference.gradient_accumulation_steps=${DPO_GRAD_ACCUM}"
                    "training.preference.concat_pairs=false"
                    "training.preference.gpu_keepalive_interval=${GPU_KEEPALIVE}"
                    "training.consistency.enabled=false"
            )

            # qwen14 OOM guard
            if [[ "${MODEL_SHORT}" == "qwen14" ]]; then
                DPO_ARGS+=("training.preference.use_reference_model=false")
                warn "qwen14 OOM guard: use_reference_model=false applied"
            fi

            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            START_T=$(date +%s)
            DPO_RC=0

            if [[ ${NUM_GPUS} -gt 1 ]]; then
                run_cmd "DPO [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH} (${NUM_GPUS} GPUs, no BC init)" \
                    accelerate launch --num_processes="${NUM_GPUS}" --multi_gpu \
                    "${DPO_ARGS[@]}" \
                    2>&1 | tee -a "${LOGFILE}" || DPO_RC=$?
            else
                run_cmd "DPO [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH} (single GPU, no BC init)" \
                    python "${DPO_ARGS[@]}" \
                    2>&1 | tee -a "${LOGFILE}" || DPO_RC=$?
            fi

            if [[ ${DRY_RUN} == false ]]; then
                ELAPSED=$(( $(date +%s) - START_T ))
                DPO_CHECKPOINT_PATH="$(resolve_dpo_checkpoint "${DPO_OUTPUT}" || echo "")"
                if [[ -n "${DPO_CHECKPOINT_PATH}" && -d "${DPO_CHECKPOINT_PATH}" ]]; then
                    ok "DPO done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s: ${DPO_CHECKPOINT_PATH}"
                elif [[ ${DPO_RC} -ne 0 ]]; then
                    err "DPO FAILED (exit ${DPO_RC}) for ${ABL_KEY} — no checkpoint"
                    FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [DPO failed]")
                    continue
                fi
            fi
        fi

        EFFECTIVE_POLICY="$(resolve_dpo_checkpoint "${DPO_OUTPUT}" || echo "${DPO_OUTPUT}/preference/best")"

    elif [[ "${POLICY_SOURCE}" == "bc" ]]; then
        # no_dpo: use BC checkpoint directly
        if [[ ! -d "${BC_CHECKPOINT}" ]]; then
            err "BC checkpoint not found: ${BC_CHECKPOINT}"
            FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [no BC checkpoint]")
            continue
        fi
        EFFECTIVE_POLICY="${BC_CHECKPOINT}"
        info "Using BC checkpoint as policy: ${EFFECTIVE_POLICY}"

    elif [[ "${POLICY_SOURCE}" == "base" ]]; then
        # no_training: use base HF model directly
        EFFECTIVE_POLICY="${MODEL_HF}"
        info "Using base HF model as policy: ${EFFECTIVE_POLICY}"
    fi

    # ──────────────────────────────────────────────────
    # Phase 2: Router Feature Generation (GPU)
    # ──────────────────────────────────────────────────
    if [[ -f "${ROUTER_FEATURES_FILE}" ]]; then
        EXISTING_LINES=$(wc -l < "${ROUTER_FEATURES_FILE}")
        info "Router features already exist (${EXISTING_LINES} lines): ${ROUTER_FEATURES_FILE}"
        info "Will RESUME (append missing records)"
    fi

    export POLICY_PATH="${EFFECTIVE_POLICY}"
    export TRAJECTORIES="${NOISY_TRAJECTORIES}"
    export CONFIG="${CONFIG_NOISY}"
    export OUTPUT="${ROUTER_FEATURES_FILE}"
    export K="${ROUTER_K}"
    export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
    export NUM_GPUS="${NUM_GPUS}"
    export GENERATE_ONLY="true"
    export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"
    export BENCHMARK="${BENCH}"

    START_T=$(date +%s)
    FEAT_RC=0

    run_cmd "Router features (generate-only) [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH}" \
        bash scripts/run_router_features_humaneval.sh \
        2>&1 | tee -a "${LOGFILE}" || FEAT_RC=$?

    if [[ ${DRY_RUN} == false ]]; then
        ELAPSED=$(( $(date +%s) - START_T ))
        if [[ ${FEAT_RC} -ne 0 ]]; then
            err "Router feature generation FAILED for ${ABL_KEY}"
            FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [router features failed]")
            unset POLICY_PATH TRAJECTORIES CONFIG OUTPUT K BATCH_SIZE GENERATE_ONLY EXTRA_OVERRIDES BENCHMARK
            continue
        fi
        ok "Router features done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
    fi

    unset POLICY_PATH TRAJECTORIES CONFIG OUTPUT K BATCH_SIZE GENERATE_ONLY EXTRA_OVERRIDES BENCHMARK

    # ──────────────────────────────────────────────────
    # Phase 3: CPU Scoring
    # ──────────────────────────────────────────────────
    export POLICY_PATH="${EFFECTIVE_POLICY}"
    export TRAJECTORIES="${NOISY_TRAJECTORIES}"
    export CONFIG="${CONFIG_NOISY}"
    export OUTPUT="${ROUTER_FEATURES_FILE}"
    export K="${ROUTER_K}"
    export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
    export SCORE_ONLY="true"
    export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"
    export BENCHMARK="${BENCH}"
    # Terminalbench has no heuristic verifier; fall back to humaneval verifier
    if [[ "${BENCH}" == "terminalbench" ]]; then
        export VERIFIER_OVERRIDE="verifier.mode=heuristic verifier.heuristic.run_code=false verifier.heuristic.benchmark=humaneval"
    fi

    START_T=$(date +%s)
    SCORE_RC=0

    run_cmd "CPU Scoring [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH}" \
        bash scripts/run_router_features_humaneval.sh \
        2>&1 | tee -a "${LOGFILE}" || SCORE_RC=$?

    if [[ ${DRY_RUN} == false ]]; then
        ELAPSED=$(( $(date +%s) - START_T ))
        if [[ ${SCORE_RC} -ne 0 ]]; then
            err "CPU scoring FAILED for ${ABL_KEY}"
            FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [scoring failed]")
        else
            ok "Scoring done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        fi
    fi

    unset POLICY_PATH TRAJECTORIES CONFIG OUTPUT K BATCH_SIZE SCORE_ONLY EXTRA_OVERRIDES BENCHMARK VERIFIER_OVERRIDE

    SUCCEEDED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY}")

done  # ablation conditions

# ── Summary ──────────────────────────────────────────────────
header "Training Stage Ablation Sweep Complete"
echo ""
echo "  Succeeded: ${#SUCCEEDED_RUNS[@]}"
echo "  Failed:    ${#FAILED_RUNS[@]}"
echo "  Skipped:   ${#SKIPPED_RUNS[@]}"
echo ""

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "  Failed runs:"
    for r in "${FAILED_RUNS[@]}"; do echo "    ✗ ${r}"; done
    echo ""
fi

if [[ ${#SUCCEEDED_RUNS[@]} -gt 0 ]]; then
    echo "  Feature files:"
    for r in "${SUCCEEDED_RUNS[@]}"; do
        KEY="${r##*/}"
        f="updated_data/router_features/${BENCH}_noisy_router_features_heuristic_${MODEL_SHORT}_abl_${KEY}.jsonl"
        LINES="$(wc -l < "${f}" 2>/dev/null || echo '?')"
        echo "    ${KEY}: ${f} (${LINES} records)"
    done
fi

echo ""
echo "  Done."
