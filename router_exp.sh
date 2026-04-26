#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Required by user preference: always use the existing ~/.venv.
if [[ ! -f "${HOME}/.venv/bin/activate" ]]; then
  echo "ERROR: Missing ${HOME}/.venv/bin/activate"
  exit 1
fi
source "${HOME}/.venv/bin/activate"

PARQUET="${PARQUET:-data/router_dataset/unified_router_features.parquet}"
CONFIG="${CONFIG:-configs/base.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/router/hparam_ablations_full}"
LOG_ROOT="${LOG_ROOT:-logs/run_evaluation/router_hparam_ablations}"
RESULTS_ROOT="${RESULTS_ROOT:-results/router_hparam_ablations}"
SEED="${SEED:-42}"
RUN_EVAL="${RUN_EVAL:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

# Optional filters. Leave empty to train/eval on all rows in the selected split.
BENCHMARK_FILTER="${BENCHMARK_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"

# Restrict to the main heuristic slice for clean one-factor router ablations.
CATEGORY_FILTER="${CATEGORY_FILTER:-main}"
VARIANT_FILTER="${VARIANT_FILTER:-heuristic}"

if ! [[ "${PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL_JOBS}" -lt 1 ]]; then
  echo "ERROR: PARALLEL_JOBS must be an integer >= 1 (got: ${PARALLEL_JOBS})"
  exit 1
fi

ROWS_DIR="${RESULTS_ROOT}/_rows"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${RESULTS_ROOT}" "${ROWS_DIR}"

SUMMARY_CSV="${RESULTS_ROOT}/router_sweep_summary.csv"

COMMON_ARGS=(
  --config "${CONFIG}"
  --features "${PARQUET}"
  --train-split train
  --val-split val
  --category-filter "${CATEGORY_FILTER}"
  --variant-filter "${VARIANT_FILTER}"
)

if [[ -n "${BENCHMARK_FILTER}" ]]; then
  COMMON_ARGS+=(--benchmark-filter "${BENCHMARK_FILTER}")
fi
if [[ -n "${MODEL_FILTER}" ]]; then
  COMMON_ARGS+=(--model-filter "${MODEL_FILTER}")
fi

THRESHOLD_SWEEP="${THRESHOLD_SWEEP:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"

COMMON_OVERRIDES=(
  "project.seed=${SEED}"
  "logging.wandb_mode=disabled"
  "training.router.eval_every_epochs=1"
  "training.router.checkpoint_every_epochs=1"
)

# run_threshold_sweep_eval — eval-only, no training.
# Sweeps the routing threshold on an already-trained router checkpoint.
# Called after the "baseline" run so the Pareto SR-vs-LLM% curve can be plotted.
run_threshold_sweep_eval() {
  local router_path="$1"
  local eval_output_dir="${RESULTS_ROOT}/threshold_sweep"
  local log_file="${LOG_ROOT}/threshold_sweep.eval.log"

  if [[ "${RUN_EVAL}" != "true" ]]; then
    return 0
  fi

  if [[ ! -f "${router_path}" ]]; then
    echo "[SKIP] threshold sweep: router checkpoint missing: ${router_path}"
    return 0
  fi

  echo
  echo "================================================================"
  echo "Threshold sweep eval (eval-only, no training)"
  echo "Router: ${router_path}"
  echo "Thresholds: ${THRESHOLD_SWEEP}"
  echo "================================================================"

  # shellcheck disable=SC2068
  read -r -a threshold_arr <<< "${THRESHOLD_SWEEP}"

  local -a eval_args=(
    --config "${CONFIG}"
    --features "${PARQUET}"
    --split test
    --feature-transform none
    --category-filter "${CATEGORY_FILTER}"
    --variant-filter "${VARIANT_FILTER}"
    --router-path "${router_path}"
    --output "${eval_output_dir}"
    --methods r2v slm_only llm_only entropy_router oracle_router
    --router-threshold-sweep "${threshold_arr[@]}"
    --seeds 1 2 3
  )
  if [[ -n "${BENCHMARK_FILTER}" ]]; then
    eval_args+=(--benchmark-filter "${BENCHMARK_FILTER}")
  fi
  if [[ -n "${MODEL_FILTER}" ]]; then
    eval_args+=(--model-filter "${MODEL_FILTER}")
  fi

  python scripts/evaluate.py "${eval_args[@]}" > "${log_file}" 2>&1
}

declare -a RUN_SPECS=()

add_run() {
  local name="$1"
  local group="$2"
  local hparam="$3"
  local value="$4"
  local feature_transform="$5"
  local overrides="$6"
  RUN_SPECS+=("${name}"$'\t'"${group}"$'\t'"${hparam}"$'\t'"${value}"$'\t'"${feature_transform}"$'\t'"${overrides}")
}

run_train_and_eval() {
  local name="$1"
  local group="$2"
  local hparam="$3"
  local value="$4"
  local feature_transform="$5"
  local overrides_raw="$6"

  local output_dir="${OUT_ROOT}/${name}"
  local log_file="${LOG_ROOT}/${name}.log"
  local eval_log_file="${LOG_ROOT}/${name}.eval.log"
  local eval_output_dir="${RESULTS_ROOT}/${name}"
  local row_json="${ROWS_DIR}/${name}.json"
  local -a overrides=("${COMMON_OVERRIDES[@]}")

  if [[ -n "${overrides_raw}" ]]; then
    local ov
    for ov in ${overrides_raw}; do
      overrides+=("${ov}")
    done
  fi

  echo
  echo "================================================================"
  echo "Router sweep run: ${name}"
  echo "Group/HParam: ${group} / ${hparam}=${value}"
  echo "Feature transform: ${feature_transform}"
  echo "Output: ${output_dir}"
  echo "Train log: ${log_file}"
  echo "Eval log: ${eval_log_file}"
  echo "Row summary: ${row_json}"
  echo "Overrides: ${overrides[*]}"
  echo "================================================================"

  if [[ "${SKIP_EXISTING}" == "true" && -f "${output_dir}/router_final.pt" ]]; then
    echo "[SKIP] Found existing checkpoint: ${output_dir}/router_final.pt"
  else
    python scripts/train_router.py \
      "${COMMON_ARGS[@]}" \
      --feature-transform "${feature_transform}" \
      --output "${output_dir}" \
      --overrides "${overrides[@]}" \
      > "${log_file}" 2>&1
  fi

  if [[ "${RUN_EVAL}" != "true" ]]; then
    return 0
  fi

  local -a eval_args=(
    --config "${CONFIG}"
    --features "${PARQUET}"
    --split test
    --feature-transform "${feature_transform}"
    --category-filter "${CATEGORY_FILTER}"
    --variant-filter "${VARIANT_FILTER}"
    --router-path "${output_dir}/router_final.pt"
    --output "${eval_output_dir}"
    --methods r2v
    --seeds 1 2 3
  )
  if [[ -n "${BENCHMARK_FILTER}" ]]; then
    eval_args+=(--benchmark-filter "${BENCHMARK_FILTER}")
  fi
  if [[ -n "${MODEL_FILTER}" ]]; then
    eval_args+=(--model-filter "${MODEL_FILTER}")
  fi

  python scripts/evaluate.py "${eval_args[@]}" > "${eval_log_file}" 2>&1

  python - "${row_json}" "${name}" "${group}" "${hparam}" "${value}" \
    "${feature_transform}" "${output_dir}" "${eval_output_dir}" <<'PY'
import json
import sys
from pathlib import Path

(
    row_json,
    run_name,
    group,
    hparam,
    value,
    feature_transform,
    output_dir,
    eval_output_dir,
) = sys.argv[1:]

train_summary_path = Path(output_dir) / "training_summary.json"
best_epoch = None
best_eval_brier = None
if train_summary_path.exists():
    with open(train_summary_path, "r", encoding="utf-8") as f:
        ts = json.load(f)
    best_epoch = ts.get("best_epoch")
    best_eval_brier = ts.get("best_eval_brier")

eval_dir = Path(eval_output_dir) / "structured_results"
eval_jsons = sorted(eval_dir.glob("eval_*.json"))
if not eval_jsons:
    raise SystemExit(f"No eval bundle found in {eval_dir}")

eval_json_path = eval_jsons[-1]
with open(eval_json_path, "r", encoding="utf-8") as f:
    bundle = json.load(f)

r2v_rows = [r for r in bundle.get("eval_results", []) if r.get("method") == "r2v"]
if not r2v_rows:
    raise SystemExit(f"No r2v row found in {eval_json_path}")

row = next((r for r in r2v_rows if r.get("cvar_failure") is not None), r2v_rows[0])

out_row = {
    "run_name": run_name,
    "group": group,
    "hparam": hparam,
    "value": value,
    "feature_transform": feature_transform,
    "best_epoch": best_epoch,
    "best_eval_brier": best_eval_brier,
    "success_rate": row.get("success_rate"),
    "worst_seed_sr": row.get("worst_seed_sr"),
    "cvar_failure": row.get("cvar_failure"),
    "avg_cost": row.get("avg_cost"),
    "llm_call_rate": row.get("llm_call_rate"),
    "ece": row.get("ece"),
    "brier": row.get("brier"),
    "router_path": str(Path(output_dir) / "router_final.pt"),
    "eval_json": str(eval_json_path),
}

with open(row_json, "w", encoding="utf-8") as f:
    json.dump(out_row, f, indent=2)
PY
}

# Baseline
add_run "baseline" "baseline" "none" "baseline" "none" ""

# Feature-transform ablations from existing smoke evidence.
add_run "feature_no_entropy" "feature_transform" "feature_transform" "no_entropy" "no_entropy" ""
add_run "feature_verifier_pseudo_entropy" "feature_transform" "feature_transform" "verifier_pseudo_entropy" "verifier_pseudo_entropy" ""

# Robust objective family.
add_run "objective_worst_case" "objective" "robust_objective" "worst_case" "none" "training.router.robust_objective=worst_case"
add_run "objective_expected" "objective" "robust_objective" "expected" "none" "training.router.robust_objective=expected"

# CVaR alpha sweep (denser for plotting).
for alpha in 0.05 0.10 0.15 0.30 0.40 0.50; do
  name="cvar_alpha_${alpha//./_}"
  add_run "${name}" "cvar_alpha" "training.router.cvar_alpha" "${alpha}" "none" "training.router.cvar_alpha=${alpha}"
done

# CVaR epsilon sweep (denser for plotting).
for eps in 0.05 0.10 0.20 0.40 0.50 0.60; do
  name="cvar_epsilon_${eps//./_}"
  add_run "${name}" "cvar_epsilon" "training.router.cvar_epsilon" "${eps}" "none" "training.router.cvar_epsilon=${eps}"
done

# Calibration sweep.
for w in 0.0 0.25 0.50 2.0 4.0; do
  name="brier_weight_${w//./_}"
  add_run "${name}" "calibration" "training.router.brier_weight" "${w}" "none" "training.router.brier_weight=${w}"
done
add_run "no_temp_scaling" "calibration" "training.router.temperature_scaling" "false" "none" "training.router.temperature_scaling=false"

# Cost sweep (extends plotting range).
for cost in 10 25 35 75 100 150; do
  name="cost_llm_${cost}"
  add_run "${name}" "cost" "training.router.cost_llm" "${cost}" "none" "training.router.cost_llm=${cost}.0"
done

# Optimization sensitivity.
for llr in 0.005 0.020; do
  name="lagrangian_lr_${llr//./_}"
  add_run "${name}" "optimization" "training.router.lagrangian_lr" "${llr}" "none" "training.router.lagrangian_lr=${llr}"
done
for lr in 0.0005 0.0020; do
  name="router_lr_${lr//./_}"
  add_run "${name}" "optimization" "training.router.learning_rate" "${lr}" "none" "training.router.learning_rate=${lr}"
done

# Architecture sensitivity.
add_run "router_hidden_shallow" "architecture" "router.hidden_dims" "[128]" "none" "router.hidden_dims=[128]"
add_run "router_hidden_deep" "architecture" "router.hidden_dims" "[256,128,64]" "none" "router.hidden_dims=[256,128,64]"
add_run "router_dropout_0_1" "architecture" "router.dropout" "0.1" "none" "router.dropout=0.1"
add_run "router_dropout_0_3" "architecture" "router.dropout" "0.3" "none" "router.dropout=0.3"

echo "Prepared ${#RUN_SPECS[@]} router sweep runs"
echo "Parallel jobs: ${PARALLEL_JOBS}"

for spec in "${RUN_SPECS[@]}"; do
  IFS=$'\t' read -r name group hparam value feature_transform overrides <<< "${spec}"
  run_train_and_eval "${name}" "${group}" "${hparam}" "${value}" "${feature_transform}" "${overrides}" &
  while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do
    wait -n
  done
done

while (( $(jobs -rp | wc -l) > 0 )); do
  wait -n
done

# Threshold sweep eval on the baseline checkpoint (eval-only, no training needed).
baseline_router="${OUT_ROOT}/baseline/router_final.pt"
run_threshold_sweep_eval "${baseline_router}"

if [[ "${RUN_EVAL}" == "true" ]]; then
  python - "${ROWS_DIR}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

rows_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])

rows = []
for p in sorted(rows_dir.glob("*.json")):
    with open(p, "r", encoding="utf-8") as f:
        rows.append(json.load(f))

fieldnames = [
    "run_name",
    "group",
    "hparam",
    "value",
    "feature_transform",
    "best_epoch",
    "best_eval_brier",
    "success_rate",
    "worst_seed_sr",
    "cvar_failure",
    "avg_cost",
    "llm_call_rate",
    "ece",
    "brier",
    "router_path",
    "eval_json",
]

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote {len(rows)} rows to {summary_csv}")
PY
fi

echo
echo "All router hyperparameter sweep runs finished."
echo "Outputs: ${OUT_ROOT}"
echo "Logs: ${LOG_ROOT}"
if [[ "${RUN_EVAL}" == "true" ]]; then
  echo "Summary CSV: ${SUMMARY_CSV}"
fi