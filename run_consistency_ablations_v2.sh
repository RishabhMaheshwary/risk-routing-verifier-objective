#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Consistency Lambda Ablation Runner — TerminalBench v2
# ══════════════════════════════════════════════════════════════════
#
# Runs DPO with various consistency λ values for qwen7 and qwen14,
# then generates router features for each.
#
# Each job: DPO training → Router feature generation → copy to ablation dir
#
# Usage:
#   nohup bash run_consistency_ablations_v2.sh > logs/consistency_ablations_v2.log 2>&1 &
#   bash run_consistency_ablations_v2.sh --gpus 4          # total GPUs (default 8)
#   bash run_consistency_ablations_v2.sh --gpus-per-job 2  # GPUs per DPO job
#   bash run_consistency_ablations_v2.sh --model qwen7     # single model
#   bash run_consistency_ablations_v2.sh --dry-run         # preview
#
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate 2>/dev/null || true
export CC=/usr/bin/gcc

# ── Defaults ─────────────────────────────────────────────────────
TOTAL_GPUS="${TOTAL_GPUS:-8}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
FILTER_MODEL=""
DRY_RUN=false

DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-32}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"
DPO_EPOCHS="${DPO_EPOCHS:-3}"

BENCH="terminalbench"
CONFIG_NOISY="configs/terminalbench/noisy.yaml"
SPLIT_DIR="data/trajectories/terminalbench_noisy"
NOISY_TRAJECTORIES="${SPLIT_DIR}/trajectories.jsonl"

QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

FEATURES_DIR="${SCRIPT_DIR}/all_ablation_features_v2/terminalbench"

# ── Argument parsing ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)             TOTAL_GPUS="$2";    shift 2 ;;
        --gpus-per-job)     GPUS_PER_JOB="$2";  shift 2 ;;
        --model)            FILTER_MODEL="$2";   shift 2 ;;
        --dpo-batch-size)   DPO_BATCH_SIZE="$2"; shift 2 ;;
        --dpo-grad-accum)   DPO_GRAD_ACCUM="$2"; shift 2 ;;
        --dpo-epochs)       DPO_EPOCHS="$2";     shift 2 ;;
        --lora-r)           LORA_R="$2";         shift 2 ;;
        --lora-alpha)       LORA_ALPHA="$2";     shift 2 ;;
        --router-batch-size) ROUTER_BATCH_SIZE="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=true;        shift ;;
        --help|-h) head -18 "$0" | tail -16; exit 0 ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

MAX_PARALLEL=$(( TOTAL_GPUS / GPUS_PER_JOB ))
[[ ${MAX_PARALLEL} -lt 1 ]] && MAX_PARALLEL=1

# ── Model registry ────────────────────────────────────────────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
)
declare -A BC_DIR_TAG=(
    [qwen7]="qwen_coder_7b"
    [qwen14]="qwen_coder_14b"
)

if [[ -n "${FILTER_MODEL}" ]]; then
    [[ -z "${HF_ID[${FILTER_MODEL}]+_}" ]] && { echo "ERROR: Unknown model '${FILTER_MODEL}'"; exit 1; }
    MODELS=("${FILTER_MODEL}")
else
    MODELS=(qwen7 qwen14)
fi

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

# ── Colour helpers ────────────────────────────────────────────────
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
GPU_SLOT_DIR=$(mktemp -d)
trap 'rm -rf "${GPU_SLOT_DIR}"' EXIT

for (( g=0; g<TOTAL_GPUS; g++ )); do
    touch "${GPU_SLOT_DIR}/gpu_${g}.free"
done

acquire_gpus() {
    local n="$1"
    local acquired=()
    while true; do
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
                local ids=()
                for f in "${acquired[@]}"; do
                    local base
                    base=$(basename "${f}" .free)
                    ids+=("${base#gpu_}")
                done
                echo "$(IFS=,; echo "${ids[*]}")"
                return 0
            else
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
    local ids_csv="$1"
    IFS=',' read -ra ids <<< "${ids_csv}"
    for id in "${ids[@]}"; do
        mv "${GPU_SLOT_DIR}/gpu_${id}.busy" "${GPU_SLOT_DIR}/gpu_${id}.free" 2>/dev/null || true
    done
}

# ── Banner ────────────────────────────────────────────────────────
header "Consistency Lambda Ablation — TerminalBench v2"
echo ""
echo "  Benchmark:    ${BENCH}"
echo "  Models:       ${MODELS[*]}"
echo "  Ablations:    ${#LAMBDA_ABLATIONS[@]} lambda values"
echo "  Total GPUs:   ${TOTAL_GPUS}"
echo "  GPUs/job:     ${GPUS_PER_JOB}"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "  DPO batch:    ${DPO_BATCH_SIZE} x ${DPO_GRAD_ACCUM} accum, ${DPO_EPOCHS} epochs"
echo "  LoRA:         r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  Dry run:      ${DRY_RUN}"
echo ""

# ── Validate prerequisites ────────────────────────────────────────
if [[ ! -f "${CONFIG_NOISY}" ]]; then
    err "Config not found: ${CONFIG_NOISY}"; exit 1
fi
if [[ ! -f "${NOISY_TRAJECTORIES}" ]]; then
    err "Trajectories not found: ${NOISY_TRAJECTORIES}"; exit 1
fi
ok "Config:       ${CONFIG_NOISY}"
ok "Trajectories: ${NOISY_TRAJECTORIES}"

# ── Logging ───────────────────────────────────────────────────────
LOGDIR="${SCRIPT_DIR}/logs/consistency_ablations_v2"
mkdir -p "${LOGDIR}"
SUMMARY_FILE="${LOGDIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

# ── Worker function ───────────────────────────────────────────────
run_ablation_job() {
    local model_short="$1"
    local abl_key="$2"
    local abl_overrides="$3"
    local abl_desc="$4"
    local gpu_ids="$5"
    local run_idx="$6"
    local total="$7"

    local model_hf="${HF_ID[${model_short}]}"
    local model_tag="${BC_DIR_TAG[${model_short}]}"
    local bc_checkpoint="outputs/policy/${BENCH}_noisy_bc_${model_tag}/best"
    if [[ ! -d "${bc_checkpoint}" ]]; then
        bc_checkpoint="outputs/policy/${BENCH}_noisy_bc_${model_tag}/final"
    fi
    local dpo_output="outputs/policy/${BENCH}_noisy_dpo_${model_short}_abl_${abl_key}"
    local dpo_checkpoint="${dpo_output}/final"
    local router_features_file="data/router_features/${BENCH}_noisy_router_features_heuristic_${model_short}_abl_${abl_key}.jsonl"
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
    if [[ -d "${dpo_checkpoint}" ]]; then
        echo "[$(date +%H:%M:%S)] DPO checkpoint exists — skipping: ${dpo_checkpoint}"
    else
        echo "[$(date +%H:%M:%S)] Starting DPO on GPU(s) ${gpu_ids}..."

        local gpu_keepalive="0"
        [[ "${model_short}" == "qwen14" ]] && gpu_keepalive="0.2"

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
                "training.preference.epochs=${DPO_EPOCHS}"
        )
        # Add ablation overrides
        for ov in ${abl_overrides}; do DPO_ARGS+=("${ov}"); done

        # qwen14: disable reference model to save memory
        if [[ "${model_short}" == "qwen14" ]]; then
            DPO_ARGS+=("training.preference.use_reference_model=false")
        fi

        DPO_ARGS+=(--resume "${bc_checkpoint}")

        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export DS_IGNORE_CUDA_DETECTION=1
        local dpo_rc=0
        local t0; t0=$(date +%s)

        # Each parallel job needs a unique TCP rendezvous port.
        # Use a random high port to avoid collisions when multiple jobs
        # acquire GPUs simultaneously.
        local main_port=$(( 30000 + RANDOM % 10000 ))

        if [[ "${DRY_RUN}" == true ]]; then
            echo "[DRY RUN] Would run DPO: accelerate launch --num_processes=${num_gpus_job} --main_process_port=${main_port} --multi_gpu ${DPO_ARGS[*]}"
        elif [[ ${num_gpus_job} -gt 1 ]]; then
            accelerate launch --num_processes="${num_gpus_job}" --main_process_port="${main_port}" --multi_gpu \
                "${DPO_ARGS[@]}" || dpo_rc=$?
        else
            python "${DPO_ARGS[@]}" || dpo_rc=$?
        fi

        local elapsed=$(( $(date +%s) - t0 ))
        if [[ ${dpo_rc} -ne 0 && ! -d "${dpo_checkpoint}" ]]; then
            echo "[ERROR] DPO FAILED (exit ${dpo_rc})"
            echo "FAILED|${model_short}|${abl_key}|DPO|exit=${dpo_rc}|${elapsed}s" >> "${SUMMARY_FILE}"
            overall_rc=1
        else
            echo "[OK] DPO done in $(( elapsed/60 ))m $(( elapsed%60 ))s"
            echo "DPO_OK|${model_short}|${abl_key}|${elapsed}s" >> "${SUMMARY_FILE}"
        fi
    fi

    [[ ${overall_rc} -ne 0 ]] && return ${overall_rc}

    # ── Phase 2: Router Feature Generation ────────────────
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

    [[ ${overall_rc} -ne 0 ]] && return ${overall_rc}

    # ── Phase 3: Merge shards & copy to ablation dir ──────
    echo "[$(date +%H:%M:%S)] Merging router feature shards..."

    if [[ "${DRY_RUN}" == false && ! -f "${router_features_file}" ]]; then
        python scripts/generate_router_features.py --merge --output "${router_features_file}" || true
    fi

    # Copy to ablation features directory
    if [[ "${DRY_RUN}" == false && -f "${router_features_file}" ]]; then
        local dest_dir="${FEATURES_DIR}/${model_short}"
        mkdir -p "${dest_dir}"
        cp "${router_features_file}" "${dest_dir}/abl_${abl_key}.jsonl"
        echo "[OK] Copied features -> ${dest_dir}/abl_${abl_key}.jsonl"
    fi

    } 2>&1 | tee "${logfile}"
    local pipe_rc=${PIPESTATUS[0]}
    return ${pipe_rc}
}

# ── Build job list & validate ─────────────────────────────────────
declare -a JOB_MODEL=()
declare -a JOB_KEY=()
declare -a JOB_OVERRIDES=()
declare -a JOB_DESC=()

for MODEL_SHORT in "${MODELS[@]}"; do
    BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${BC_DIR_TAG[${MODEL_SHORT}]}/best"
    if [[ ! -d "${BC_CHECKPOINT}" ]]; then
        BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${BC_DIR_TAG[${MODEL_SHORT}]}/final"
    fi
    PREF_TRAIN="${SPLIT_DIR}/pref_${MODEL_SHORT}_train.jsonl"

    if [[ ! -d "${BC_CHECKPOINT}" ]]; then
        warn "BC checkpoint not found for ${MODEL_SHORT} — skipping"
        continue
    fi
    if [[ ! -f "${PREF_TRAIN}" ]]; then
        warn "Preference splits missing for ${MODEL_SHORT} — skipping"
        continue
    fi
    ensure_pref_val_half "${MODEL_SHORT}" || continue
    ok "Model ${MODEL_SHORT}: prerequisites OK"

    for entry in "${LAMBDA_ABLATIONS[@]}"; do
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
info "Jobs to run: ${TOTAL_JOBS} (${#MODELS[@]} models x ${#LAMBDA_ABLATIONS[@]} lambda values), ${MAX_PARALLEL} parallel"
echo ""

# ── Dispatch jobs ─────────────────────────────────────────────────
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
    info "Acquired GPU(s) ${gpu_ids} -> launching ${model}/${key}"

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
header "Consistency Lambda Ablation Complete"
echo ""
if [[ -f "${SUMMARY_FILE}" ]]; then
    dpo_ok=$(grep -c "^DPO_OK"   "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    feat_ok=$(grep -c "^FEAT_OK" "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    n_failed=$(grep -c "^FAILED"  "${SUMMARY_FILE}" 2>/dev/null || echo 0)
    echo "  DPO trained:      ${dpo_ok}"
    echo "  Features gen:     ${feat_ok}"
    echo "  Failed stages:    ${n_failed}"
    echo "  Summary log:      ${SUMMARY_FILE}"
    if [[ ${n_failed} -gt 0 ]]; then
        echo ""
        echo "  Failed:"
        grep "^FAILED" "${SUMMARY_FILE}" | while IFS='|' read -r _ m k stage rest; do
            echo "    x ${m}/${k} [${stage}] ${rest}"
        done
    fi
fi
echo ""
echo "  Artifacts:"
echo "    DPO checkpoints: outputs/policy/terminalbench_noisy_dpo_*_abl_*/"
echo "    Router features: all_ablation_features_v2/terminalbench/{model}/abl_*.jsonl"
echo "══════════════════════════════════════════════════════"
