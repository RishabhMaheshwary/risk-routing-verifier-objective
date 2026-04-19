#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Parallel Ablation Runner — Terminal-Bench (4× H100)
# ══════════════════════════════════════════════════════════════════
#
# Runs DPO β / consistency λ / margin ablations for terminalbench,
# dispatching independent jobs across N GPUs in parallel.
#
# Each job owns GPUS_PER_JOB GPUs for its full pipeline:
#   DPO training → Router feature generation → CPU scoring
#
# Usage:
#   bash run_ablations_terminalbench.sh                    # 4 GPUs, 1 per job → 4 parallel
#   bash run_ablations_terminalbench.sh --gpus 4           # total GPUs available
#   bash run_ablations_terminalbench.sh --gpus-per-job 2   # 2 GPUs per DPO job → 2 parallel
#   bash run_ablations_terminalbench.sh --model qwen14     # larger model
#   bash run_ablations_terminalbench.sh --ablation beta    # single ablation type
#   bash run_ablations_terminalbench.sh --dry-run          # preview without running
#   bash run_ablations_terminalbench.sh --skip-dpo         # features + scoring only
#   bash run_ablations_terminalbench.sh --only-score       # CPU scoring only
#   bash run_ablations_terminalbench.sh --skip-bc          # DPO from base model (no BC init)
#
# Ablation types:
#   beta     DPO β sweep
#   lambda   Consistency λ sweep
#   margin   DPO min_score_gap sweep
#   all      All three (default)
#
# Prerequisites:
#   - updated_data/trajectories/terminalbench_noisy/trajectories.jsonl
#   - configs/terminalbench/noisy.yaml
#   - BC checkpoint: outputs/policy/terminalbench_noisy_bc_<model_tag>/best (or /final)
#   - Preference splits: updated_data/trajectories/terminalbench_noisy/pref_<model>_{train,val}.jsonl
#
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Defaults ─────────────────────────────────────────────────────
TOTAL_GPUS="${TOTAL_GPUS:-4}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"          # GPUs each DPO job uses
FILTER_MODEL=""
FILTER_MODELS=()                            # multi-model override (--models m1 m2 ...)
FILTER_ABLATION="all"
DRY_RUN=false
SKIP_DPO=false
SKIP_BC=false
SKIP_ROUTER_FEATURES=false
SKIP_SCORING=false
ONLY_SCORE=false

DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-32}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"

BENCH="terminalbench"
CONFIG_NOISY="configs/terminalbench/noisy.yaml"
SPLIT_DIR="${SPLIT_DIR:-updated_data/trajectories/terminalbench_noisy}"
NOISY_TRAJECTORIES="${NOISY_TRAJECTORIES:-${SPLIT_DIR}/trajectories.jsonl}"

QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

# ── Argument parsing ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)             TOTAL_GPUS="$2";    shift 2 ;;
        --gpus-per-job)     GPUS_PER_JOB="$2";  shift 2 ;;
        --model)            FILTER_MODEL="$2";   shift 2 ;;
        --models)           shift; while [[ $# -gt 0 && "$1" != --* ]]; do FILTER_MODELS+=("$1"); shift; done ;;
        --split-dir)        SPLIT_DIR="$2";      shift 2 ;;
        --trajectories)     NOISY_TRAJECTORIES="$2"; shift 2 ;;
        --ablation)         FILTER_ABLATION="$2"; shift 2 ;;
        --dpo-batch-size)   DPO_BATCH_SIZE="$2"; shift 2 ;;
        --dpo-grad-accum)   DPO_GRAD_ACCUM="$2"; shift 2 ;;
        --lora-r)           LORA_R="$2";         shift 2 ;;
        --lora-alpha)       LORA_ALPHA="$2";     shift 2 ;;
        --router-batch-size) ROUTER_BATCH_SIZE="$2"; shift 2 ;;
        --skip-dpo)         SKIP_DPO=true;       shift ;;
        --skip-bc)          SKIP_BC=true;        shift ;;
        --skip-router-features) SKIP_ROUTER_FEATURES=true; shift ;;
        --skip-scoring)     SKIP_SCORING=true;   shift ;;
        --only-score)       ONLY_SCORE=true;     shift ;;
        --dry-run)          DRY_RUN=true;        shift ;;
        --help|-h) head -35 "$0" | tail -33; exit 0 ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

MAX_PARALLEL=$(( TOTAL_GPUS / GPUS_PER_JOB ))
[[ ${MAX_PARALLEL} -lt 1 ]] && MAX_PARALLEL=1

# ── Model registry ────────────────────────────────────────────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
    [llama]="meta-llama/Llama-3.1-8B-Instruct"
    [gemma]="google/gemma-2-9b-it"
    [deepseek]="deepseek-ai/deepseek-coder-6.7b-instruct"
)
declare -A BC_DIR_TAG=(
    [qwen7]="qwen_coder_7b"
    [qwen14]="qwen_coder_14b"
    [llama]="llama_3_1_8b_instruct"
    [gemma]="gemma_2_9b_it"
    [deepseek]="deepseek_coder_6_7b"
)

MODELS=(qwen7)
if [[ ${#FILTER_MODELS[@]} -gt 0 ]]; then
    for _m in "${FILTER_MODELS[@]}"; do
        [[ -z "${HF_ID[${_m}]+_}" ]] && { echo "ERROR: Unknown model '${_m}'"; exit 1; }
    done
    MODELS=("${FILTER_MODELS[@]}")
elif [[ -n "${FILTER_MODEL}" ]]; then
    [[ -z "${HF_ID[${FILTER_MODEL}]+_}" ]] && { echo "ERROR: Unknown model '${FILTER_MODEL}'"; exit 1; }
    MODELS=("${FILTER_MODEL}")
fi

# ── Ablation definitions ──────────────────────────────────────────
BETA_ABLATIONS=(
    "dpo_beta_0.01|training.preference.beta=0.01|DPO beta=0.01"
    "dpo_beta_0.05|training.preference.beta=0.05|DPO beta=0.05"
    "dpo_beta_0.2|training.preference.beta=0.2|DPO beta=0.2"
    "dpo_beta_0.5|training.preference.beta=0.5|DPO beta=0.5"
    "dpo_beta_1.0|training.preference.beta=1.0|DPO beta=1.0"
)
LAMBDA_ABLATIONS=(
    "no_consistency|training.consistency.enabled=false|Consistency disabled"
    "consistency_lambda_0.05|training.consistency.lambda_cons=0.05 training.consistency.enabled=true|lambda=0.05"
    "consistency_lambda_0.2|training.consistency.lambda_cons=0.2 training.consistency.enabled=true|lambda=0.2"
    "consistency_lambda_0.5|training.consistency.lambda_cons=0.5 training.consistency.enabled=true|lambda=0.5"
    "consistency_lambda_1.0|training.consistency.lambda_cons=1.0 training.consistency.enabled=true|lambda=1.0"
)
MARGIN_ABLATIONS=(
    "dpo_margin_0.0|training.preference.min_score_gap=0.0|margin=0.0"
    "dpo_margin_0.2|training.preference.min_score_gap=0.2|margin=0.2"
    "dpo_margin_0.4|training.preference.min_score_gap=0.4|margin=0.4"
    "dpo_margin_0.6|training.preference.min_score_gap=0.6|margin=0.6"
    "dpo_margin_0.8|training.preference.min_score_gap=0.8|margin=0.8"
)

declare -a ABLATIONS_TO_RUN=()
case "${FILTER_ABLATION}" in
    beta)   ABLATIONS_TO_RUN=("${BETA_ABLATIONS[@]}") ;;
    lambda) ABLATIONS_TO_RUN=("${LAMBDA_ABLATIONS[@]}") ;;
    margin) ABLATIONS_TO_RUN=("${MARGIN_ABLATIONS[@]}") ;;
    all)    ABLATIONS_TO_RUN=("${BETA_ABLATIONS[@]}" "${LAMBDA_ABLATIONS[@]}" "${MARGIN_ABLATIONS[@]}") ;;
    *) echo "ERROR: Unknown --ablation '${FILTER_ABLATION}'"; exit 1 ;;
esac

# ── Colour helpers / helpers ──────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()   { echo -e "${CYAN}[INFO]${NC}   $*"; }
ok()     { echo -e "${GREEN}[OK]${NC}     $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}   $*"; }
err()    { echo -e "${RED}[ERR]${NC}    $*"; }
header() { echo -e "\n${BOLD}══════════════════════════════════════════════════════${NC}"; \
           echo -e "${BOLD}  $*${NC}"; \
           echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"; }

# ── Helpers ───────────────────────────────────────────────────────

ensure_pref_val_half() {
    local model_short="$1"
    local val_half="${SPLIT_DIR}/pref_${model_short}_val_half.jsonl"
    local val_rest="${SPLIT_DIR}/pref_${model_short}_val_rest.jsonl"
    local val_data="${SPLIT_DIR}/pref_${model_short}_val.jsonl"

    if [[ -f "${val_half}" && -f "${val_rest}" ]]; then return 0; fi
    if [[ ! -f "${val_data}" ]]; then
        err "Preference val data missing: ${val_data}"
        return 1
    fi
    info "Creating 50/50 val split for ${model_short}..."
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
}

# ── GPU slot pool ─────────────────────────────────────────────────
# Each GPU gets a lock file. acquire_gpus grabs N files atomically.
GPU_SLOT_DIR=$(mktemp -d)
trap 'rm -rf "${GPU_SLOT_DIR}"' EXIT

for (( g=0; g<TOTAL_GPUS; g++ )); do
    touch "${GPU_SLOT_DIR}/gpu_${g}.free"
done

acquire_gpus() {
    # Acquire GPUS_PER_JOB slots. Blocks until available.
    # Prints comma-separated GPU IDs to stdout.
    local n="$1"
    local acquired=()
    while true; do
        # Try to atomically claim n free slots
        local candidates=()
        for f in "${GPU_SLOT_DIR}"/gpu_*.free; do
            [[ -f "${f}" ]] && candidates+=("${f}")
        done
        if [[ ${#candidates[@]} -ge ${n} ]]; then
            local ok_count=0
            for f in "${candidates[@]}"; do
                if mv "${f}" "${f%.free}.busy" 2>/dev/null; then
                    acquired+=("${f}")
                    ok_count=$(( ok_count + 1 ))
                    [[ ${ok_count} -ge ${n} ]] && break
                fi
            done
            if [[ ${#acquired[@]} -eq ${n} ]]; then
                # Extract GPU IDs from filenames
                local ids=()
                for f in "${acquired[@]}"; do
                    local base
                    base=$(basename "${f}" .free)
                    ids+=("${base#gpu_}")
                done
                echo "$(IFS=,; echo "${ids[*]}")"
                return 0
            else
                # Couldn't get enough, release what we grabbed
                for f in "${acquired[@]}"; do
                    mv "${f%.free}.busy" "${f}" 2>/dev/null || true
                done
                acquired=()
            fi
        fi
        sleep 2
    done
}

release_gpus() {
    # Release GPU IDs back to the pool.
    local ids_csv="$1"
    IFS=',' read -ra ids <<< "${ids_csv}"
    for id in "${ids[@]}"; do
        mv "${GPU_SLOT_DIR}/gpu_${id}.busy" "${GPU_SLOT_DIR}/gpu_${id}.free" 2>/dev/null || true
    done
}

# ── Banner ────────────────────────────────────────────────────────
header "Parallel Ablation Runner — terminalbench"
echo ""
echo "  Benchmark:    ${BENCH}"
echo "  Models:       ${MODELS[*]}"
echo "  Ablations:    ${FILTER_ABLATION} (${#ABLATIONS_TO_RUN[@]} configs)"
echo "  Total GPUs:   ${TOTAL_GPUS}"
echo "  GPUs/job:     ${GPUS_PER_JOB}"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "  DPO batch:    ${DPO_BATCH_SIZE} × ${DPO_GRAD_ACCUM} accum"
echo "  LoRA:         r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  Dry run:      ${DRY_RUN}"
echo "  Skip BC:      ${SKIP_BC}"
echo ""

# ── Validate shared prerequisites ────────────────────────────────
if [[ ! -f "${CONFIG_NOISY}" ]]; then
    err "Config not found: ${CONFIG_NOISY}"
    err "  Copy it from feature/terminalbench-collection: configs/terminalbench/noisy.yaml"
    exit 1
fi
if [[ ! -f "${NOISY_TRAJECTORIES}" ]]; then
    err "Trajectories not found: ${NOISY_TRAJECTORIES}"
    err "  Run: python scripts/convert_hf_trajectories.py --output ${NOISY_TRAJECTORIES}"
    exit 1
fi
ok "Config:       ${CONFIG_NOISY}"
ok "Trajectories: ${NOISY_TRAJECTORIES}"

# ── Logging ───────────────────────────────────────────────────────
LOGDIR="${SCRIPT_DIR}/logs/ablations_terminalbench"
mkdir -p "${LOGDIR}"
SUMMARY_FILE="${LOGDIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

# ── Worker function (runs as background job) ──────────────────────
run_ablation_job() {
    local model_short="$1"
    local abl_key="$2"
    local abl_overrides="$3"
    local abl_desc="$4"
    local gpu_ids="$5"    # comma-separated, e.g. "0,1"
    local run_idx="$6"
    local total="$7"

    local model_hf="${HF_ID[${model_short}]}"
    local model_tag="${BC_DIR_TAG[${model_short}]}"
    # Prefer best/ checkpoint; fall back to final/
    local bc_checkpoint="outputs/policy/${BENCH}_noisy_bc_${model_tag}/best"
    if [[ ! -d "${bc_checkpoint}" ]]; then
        bc_checkpoint="outputs/policy/${BENCH}_noisy_bc_${model_tag}/final"
    fi
    local dpo_prefix
    if [[ "${SKIP_BC}" == true ]]; then
        dpo_prefix="${BENCH}_noisy_dpo_${model_short}_skipBC_abl_${abl_key}"
    else
        dpo_prefix="${BENCH}_noisy_dpo_${model_short}_abl_${abl_key}"
    fi
    local dpo_output="outputs/policy/${dpo_prefix}"
    local dpo_checkpoint="${dpo_output}/final"
    local router_features_file="updated_data/router_features/${BENCH}_noisy_router_features_heuristic_${model_short}_abl_${abl_key}.jsonl"
    if [[ "${SKIP_BC}" == true ]]; then
        router_features_file="updated_data/router_features/${BENCH}_noisy_router_features_heuristic_${model_short}_skipBC_abl_${abl_key}.jsonl"
    fi
    local pref_train="${SPLIT_DIR}/pref_${model_short}_train.jsonl"
    local pref_val_half="${SPLIT_DIR}/pref_${model_short}_val_half.jsonl"
    local logfile="${LOGDIR}/${BENCH}_${model_short}_${abl_key}.log"
    local num_gpus_job
    num_gpus_job=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l | tr -d ' ')

    export CUDA_VISIBLE_DEVICES="${gpu_ids}"

    {
    echo "════════════════════════════════════════════════════"
    echo "  [${run_idx}/${total}] ${model_short} / ${abl_key}"
    echo "  ${abl_desc}"
    echo "  GPUs: ${gpu_ids} | Override: ${abl_overrides}"
    echo "  DPO output:  ${dpo_output}"
    echo "  Features:    ${router_features_file}"
    echo "════════════════════════════════════════════════════"

    local overall_rc=0

    # ── Phase 1: DPO Training ──────────────────────────────
    if [[ "${SKIP_DPO}" == false && "${ONLY_SCORE}" == false ]]; then
        if [[ -d "${dpo_checkpoint}" ]]; then
            echo "[$(date +%H:%M:%S)] DPO checkpoint exists — skipping: ${dpo_checkpoint}"
        else
            echo "[$(date +%H:%M:%S)] Starting DPO training on GPU(s) ${gpu_ids}..."

            # qwen14 needs keepalive to avoid GPU idle timeouts during long runs
            local gpu_keepalive="0"
            [[ "${model_short}" == "qwen14" || "${model_short}" == "gemma" ]] && gpu_keepalive="0.2"

            DPO_ARGS=(
                scripts/train_policy.py
                --config "${CONFIG_NOISY}"
                --output "${dpo_output}"
                --stage preference
                --pref-train-data "${pref_train}"
                --pref-val-data "${pref_val_half}"
                --overrides
                    "${QUANT_OVERRIDE}"
                    "${COMMON_OVERRIDES}"
                    "policy.model_name=${model_hf}"
                    "policy.lora.r=${LORA_R}"
                    "policy.lora.alpha=${LORA_ALPHA}"
                    "training.preference.batch_size=${DPO_BATCH_SIZE}"
                    "training.preference.gradient_accumulation_steps=${DPO_GRAD_ACCUM}"
                    "training.preference.concat_pairs=false"
                    "training.preference.gpu_keepalive_interval=${gpu_keepalive}"
            )
            # Ablation overrides and model-specific guards must all go BEFORE
            # --resume so they remain part of --overrides [nargs=*] for argparse.
            for ov in ${abl_overrides}; do DPO_ARGS+=("${ov}"); done

            # qwen14: large vocab (151k) OOM guards.
            # 1. Always disable reference model (copy.deepcopy dequantizes ~29 GB).
            # 2. If this ablation turns consistency ON, override it off (6× vocab tensors OOM).
            if [[ "${model_short}" == "qwen14" ]]; then
                DPO_ARGS+=("training.preference.use_reference_model=false")
                if [[ "${abl_overrides}" == *"consistency.enabled=true"* ]]; then
                    DPO_ARGS+=("training.consistency.enabled=false")
                    echo "[WARN] qwen14 lambda ablation: overriding consistency.enabled=false (vocab OOM guard)"
                fi
            fi

            # --resume must come after all --overrides values
            if [[ "${SKIP_BC}" == false ]]; then
                DPO_ARGS+=(--resume "${bc_checkpoint}")
            fi

            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            local dpo_rc=0
            local t0; t0=$(date +%s)

            if [[ "${DRY_RUN}" == true ]]; then
                echo "[DRY RUN] Would run DPO: ${DPO_ARGS[*]}"
            elif [[ ${num_gpus_job} -gt 1 ]]; then
                accelerate launch --num_processes="${num_gpus_job}" --multi_gpu \
                    "${DPO_ARGS[@]}" || dpo_rc=$?
            else
                python "${DPO_ARGS[@]}" || dpo_rc=$?
            fi

            local elapsed=$(( $(date +%s) - t0 ))
            if [[ ${dpo_rc} -ne 0 && ! -d "${dpo_checkpoint}" ]]; then
                echo "[ERROR] DPO FAILED (exit ${dpo_rc}) — no checkpoint produced"
                echo "FAILED|${model_short}|${abl_key}|DPO|exit=${dpo_rc}|${elapsed}s" >> "${SUMMARY_FILE}"
                overall_rc=1
            else
                echo "[OK] DPO done in $(( elapsed/60 ))m $(( elapsed%60 ))s"
                echo "DPO_OK|${model_short}|${abl_key}|${elapsed}s" >> "${SUMMARY_FILE}"
            fi
        fi
    fi

    [[ ${overall_rc} -ne 0 ]] && return ${overall_rc}

    # ── Phase 2: Router Feature Generation ────────────────
    if [[ "${SKIP_ROUTER_FEATURES}" == false && "${ONLY_SCORE}" == false ]]; then
        echo "[$(date +%H:%M:%S)] Generating router features on GPU(s) ${gpu_ids}..."

        local effective_dpo="${dpo_checkpoint}"
        if [[ ! -d "${effective_dpo}" ]]; then
            effective_dpo="outputs/policy/${BENCH}_noisy_dpo_${model_short}/final"
        fi

        local t0; t0=$(date +%s)
        local feat_rc=0

        if [[ "${DRY_RUN}" == true ]]; then
            echo "[DRY RUN] Would generate router features: POLICY_PATH=${effective_dpo}"
        else
            POLICY_PATH="${effective_dpo}" \
            TRAJECTORIES="${NOISY_TRAJECTORIES}" \
            CONFIG="${CONFIG_NOISY}" \
            OUTPUT="${router_features_file}" \
            K="${ROUTER_K}" \
            BATCH_SIZE="${ROUTER_BATCH_SIZE}" \
            NUM_GPUS="${num_gpus_job}" \
            GENERATE_ONLY="true" \
            EXTRA_OVERRIDES="policy.model_name=${model_hf} verifier.mode=heuristic verifier.heuristic.run_code=true verifier.heuristic.benchmark=${BENCH}" \
            BENCHMARK="${BENCH}" \
            bash scripts/run_router_features_humaneval.sh || feat_rc=$?
        fi

        local elapsed=$(( $(date +%s) - t0 ))
        if [[ ${feat_rc} -ne 0 ]]; then
            echo "[ERROR] Router features FAILED (exit ${feat_rc})"
            echo "FAILED|${model_short}|${abl_key}|ROUTER_FEAT|exit=${feat_rc}|${elapsed}s" >> "${SUMMARY_FILE}"
            overall_rc=1
        else
            echo "[OK] Router features done in $(( elapsed/60 ))m $(( elapsed%60 ))s"
            echo "FEAT_OK|${model_short}|${abl_key}|${elapsed}s" >> "${SUMMARY_FILE}"
        fi
    fi

    [[ ${overall_rc} -ne 0 ]] && return ${overall_rc}

    # ── Phase 3: CPU Scoring ───────────────────────────────
    if [[ "${SKIP_SCORING}" == false ]]; then
        echo "[$(date +%H:%M:%S)] CPU scoring..."

        local effective_dpo="${dpo_checkpoint}"
        if [[ ! -d "${effective_dpo}" ]]; then
            effective_dpo="outputs/policy/${BENCH}_noisy_dpo_${model_short}/final"
        fi

        local t0; t0=$(date +%s)
        local score_rc=0

        if [[ "${DRY_RUN}" == true ]]; then
            echo "[DRY RUN] Would score: OUTPUT=${router_features_file}"
        else
            POLICY_PATH="${effective_dpo}" \
            TRAJECTORIES="${NOISY_TRAJECTORIES}" \
            CONFIG="${CONFIG_NOISY}" \
            OUTPUT="${router_features_file}" \
            K="${ROUTER_K}" \
            BATCH_SIZE="${ROUTER_BATCH_SIZE}" \
            SCORE_ONLY="true" \
            EXTRA_OVERRIDES="policy.model_name=${model_hf}" \
            BENCHMARK="${BENCH}" \
            bash scripts/run_router_features_humaneval.sh || score_rc=$?
        fi

        local elapsed=$(( $(date +%s) - t0 ))
        if [[ ${score_rc} -ne 0 ]]; then
            echo "[ERROR] Scoring FAILED (exit ${score_rc})"
            echo "FAILED|${model_short}|${abl_key}|SCORE|exit=${score_rc}|${elapsed}s" >> "${SUMMARY_FILE}"
        else
            echo "[OK] Scoring done in $(( elapsed/60 ))m $(( elapsed%60 ))s"
            echo "SCORE_OK|${model_short}|${abl_key}|${elapsed}s" >> "${SUMMARY_FILE}"
        fi
    fi

    } 2>&1 | tee "${logfile}"
    local pipe_rc=${PIPESTATUS[0]}

    return ${pipe_rc}
}

# ── Build job list ────────────────────────────────────────────────
declare -a JOB_MODEL=()
declare -a JOB_KEY=()
declare -a JOB_OVERRIDES=()
declare -a JOB_DESC=()

for MODEL_SHORT in "${MODELS[@]}"; do
    # Prefer best/ checkpoint; fall back to final/
    BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${BC_DIR_TAG[${MODEL_SHORT}]}/best"
    if [[ ! -d "${BC_CHECKPOINT}" ]]; then
        BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${BC_DIR_TAG[${MODEL_SHORT}]}/final"
    fi
    PREF_TRAIN="${SPLIT_DIR}/pref_${MODEL_SHORT}_train.jsonl"
    PREF_VAL="${SPLIT_DIR}/pref_${MODEL_SHORT}_val.jsonl"

    # Validate prerequisites per model
    if [[ "${SKIP_DPO}" == false && "${SKIP_BC}" == false && "${ONLY_SCORE}" == false && ! -d "${BC_CHECKPOINT}" ]]; then
        warn "BC checkpoint not found: ${BC_CHECKPOINT} — skipping model ${MODEL_SHORT}"
        warn "  (use --skip-bc to run DPO from base model instead)"
        continue
    fi
    if [[ "${DRY_RUN}" == false && "${ONLY_SCORE}" == false && "${SKIP_DPO}" == false ]]; then
        if [[ ! -f "${PREF_TRAIN}" ]]; then
            warn "Preference splits missing: ${PREF_TRAIN} — skipping model ${MODEL_SHORT}"
            warn "  Run: bash run_pipeline_updated.sh --model ${MODEL_SHORT} --from 2 --only 2"
            continue
        fi
        ensure_pref_val_half "${MODEL_SHORT}" || continue
    fi
    ok "Model ${MODEL_SHORT}: prerequisites OK"

    for entry in "${ABLATIONS_TO_RUN[@]}"; do
        IFS='|' read -r key overrides desc <<< "${entry}"
        JOB_MODEL+=("${MODEL_SHORT}")
        JOB_KEY+=("${key}")
        JOB_OVERRIDES+=("${overrides}")
        JOB_DESC+=("${desc}")
    done
done

TOTAL_JOBS=${#JOB_MODEL[@]}
if [[ ${TOTAL_JOBS} -eq 0 ]]; then
    warn "No jobs to run. Check prerequisites above."
    exit 0
fi

echo ""
info "Jobs to run: ${TOTAL_JOBS} ablations × ${MAX_PARALLEL} parallel workers"
echo ""

# ── Dispatch jobs with GPU pool ───────────────────────────────────
declare -a PIDS=()
declare -a PID_LABELS=()
declare -a PID_GPUS=()

for (( i=0; i<TOTAL_JOBS; i++ )); do
    model="${JOB_MODEL[$i]}"
    key="${JOB_KEY[$i]}"
    overrides="${JOB_OVERRIDES[$i]}"
    desc="${JOB_DESC[$i]}"
    run_num=$(( i + 1 ))

    info "Waiting for ${GPUS_PER_JOB} GPU slot(s) for [${run_num}/${TOTAL_JOBS}] ${model}/${key}..."
    gpu_ids=$(acquire_gpus "${GPUS_PER_JOB}")
    info "Acquired GPU(s) ${gpu_ids} → launching ${model}/${key}"

    if [[ "${DRY_RUN}" == true ]]; then
        run_ablation_job "${model}" "${key}" "${overrides}" "${desc}" \
            "${gpu_ids}" "${run_num}" "${TOTAL_JOBS}"
        release_gpus "${gpu_ids}"
    else
        run_ablation_job "${model}" "${key}" "${overrides}" "${desc}" \
            "${gpu_ids}" "${run_num}" "${TOTAL_JOBS}" &
        pid=$!
        PIDS+=("${pid}")
        PID_LABELS+=("${model}/${key}")
        PID_GPUS+=("${gpu_ids}")

        # Register a subshell to release GPUs when this job finishes
        ( wait "${pid}" 2>/dev/null || true; release_gpus "${gpu_ids}" ) &
    fi
done

# ── Wait for all jobs ─────────────────────────────────────────────
if [[ "${DRY_RUN}" == false ]]; then
    echo ""
    info "All jobs dispatched. Waiting for completion..."
    failed=0
    for (( i=0; i<${#PIDS[@]}; i++ )); do
        pid="${PIDS[$i]}"
        label="${PID_LABELS[$i]}"
        rc=0
        wait "${pid}" || rc=$?
        if [[ ${rc} -ne 0 ]]; then
            err "FAILED: ${label} (exit ${rc})"
            failed=$(( failed + 1 ))
        else
            ok "Done: ${label}"
        fi
    done
fi

# ── Summary ───────────────────────────────────────────────────────
header "Ablation Sweep Complete"
echo ""
if [[ -f "${SUMMARY_FILE}" ]]; then
    dpo_ok=$(grep -c "^DPO_OK"   "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    feat_ok=$(grep -c "^FEAT_OK" "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    score_ok=$(grep -c "^SCORE_OK" "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    n_failed=$(grep -c "^FAILED"  "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    echo "  DPO trained:      ${dpo_ok}"
    echo "  Features gen:     ${feat_ok}"
    echo "  Scored:           ${score_ok}"
    echo "  Failed stages:    ${n_failed}"
    echo "  Summary log:      ${SUMMARY_FILE}"
    if [[ ${n_failed} -gt 0 ]]; then
        echo ""
        echo "  Failed:"
        grep "^FAILED" "${SUMMARY_FILE}" | while IFS='|' read -r _ m k stage rest; do
            echo "    ✗ ${m}/${k} [${stage}] ${rest}"
        done
    fi
fi
echo ""
echo "  Artifacts:"
echo "    DPO checkpoints: outputs/policy/terminalbench_noisy_dpo_*_abl_*/"
echo "    Router features: updated_data/router_features/terminalbench_noisy_*_abl_*.jsonl"
echo ""
echo "  Next: train routers"
echo "    for f in updated_data/router_features/terminalbench_noisy_*_abl_*.jsonl; do"
echo "      python scripts/train_router.py \\"
echo "        --config configs/terminalbench/noisy.yaml \\"
echo "        --features \"\$f\" \\"
echo "        --output outputs/router/ablations/\$(basename \"\$f\" .jsonl)"
echo "    done"
echo "══════════════════════════════════════════════════════"
