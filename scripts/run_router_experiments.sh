#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Reuse the existing local venv only; do not create a new environment.
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
elif [[ -f "${HOME}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.venv/bin/activate"
fi

FEATURES_ROOT="router_features_data"
DATASET_PATH="data/router_dataset/unified_router_features.parquet"
ROUTER_OUTPUT="outputs/router/unified_single_router"
RESULTS_ROOT="results/router_experiments_single"
TRAIN_CONFIG="configs/base.yaml"

SEED=42
MAX_PERTURBATIONS_PER_TASK=2
THRESHOLDS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

CLOSED_SOURCE_MODELS=(deepseek)

MANIFEST_PATH="${RESULTS_ROOT}/experiment_manifest.jsonl"
METRICS_PATH="${RESULTS_ROOT}/metrics_long.csv"

DRY_RUN=false
SKIP_BUILD=false
SKIP_TRAIN=false
SKIP_EVAL=false
FORCE_RETRAIN=false
EXTRA_OVERRIDES=""
TRAIN_EXTRA_ROUTERS=true

print_help() {
    cat <<'EOF'
Usage: bash scripts/run_router_experiments.sh [options]

Dataset-first pipeline:
1) Build one unified Parquet dataset from router_features_data
2) Train one router on train split, validate on val split
3) Evaluate paper slices on test split (main, feature ablations, main ablations, figure)
4) Generate plots/tables via scripts/plot_router_experiments.py

Options:
  --features-root PATH            Feature JSONL root (default: router_features_data)
  --dataset-path PATH             Unified parquet output/input path
  --router-output PATH            Single router output dir
  --results-root PATH             Evaluation outputs root
  --train-config PATH             Config for single router training (default: configs/base.yaml)
  --seed INT                      Split seed (default: 42)
  --max-perturbations-per-task N  Split reconstruction setting (default: 2)
  --thresholds "t1 t2 ..."        Threshold sweep for figure
  --extra-overrides "k=v ..."     Extra OmegaConf overrides for train/eval
  --skip-build                    Skip dataset build
  --skip-train                    Skip single router training
  --skip-eval                     Skip evaluations and plotting
  --force-retrain                 Retrain router even if checkpoint exists
    --no-extra-routers              Train only the main router (skip ablation routers)
  --dry-run                       Print commands only
  -h, --help                      Show help

Example:
  bash scripts/run_router_experiments.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --features-root)
            FEATURES_ROOT="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --router-output)
            ROUTER_OUTPUT="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            MANIFEST_PATH="${RESULTS_ROOT}/experiment_manifest.jsonl"
            METRICS_PATH="${RESULTS_ROOT}/metrics_long.csv"
            shift 2
            ;;
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max-perturbations-per-task)
            MAX_PERTURBATIONS_PER_TASK="$2"
            shift 2
            ;;
        --thresholds)
            IFS=' ' read -r -a THRESHOLDS <<< "$2"
            shift 2
            ;;
        --extra-overrides)
            EXTRA_OVERRIDES="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        --no-extra-routers)
            TRAIN_EXTRA_ROUTERS=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${RESULTS_ROOT}"

config_for_benchmark() {
    local benchmark="$1"
    case "${benchmark}" in
        humaneval)
            echo "configs/humaneval/noisy.yaml"
            ;;
        textworld)
            echo "configs/textworld/noisy.yaml"
            ;;
        terminalbench)
            # Use textworld defaults if a dedicated terminalbench config is unavailable.
            echo "configs/textworld/noisy.yaml"
            ;;
        *)
            echo "${TRAIN_CONFIG}"
            ;;
    esac
}

is_closed_source_model() {
    local model="$1"
    local m
    for m in "${CLOSED_SOURCE_MODELS[@]}"; do
        if [[ "${m}" == "${model}" ]]; then
            return 0
        fi
    done
    return 1
}

run_cmd() {
    local label="$1"
    shift
    echo
    echo "============================================================"
    echo "${label}"
    echo "============================================================"
    printf '$ %q ' "$@"
    echo

    if [[ "${DRY_RUN}" == true ]]; then
        echo "[DRY RUN]"
        return 0
    fi

    "$@"
}

latest_bundle_in_eval_dir() {
    local eval_dir="$1"
    local bundle
    bundle="$(ls -1t "${eval_dir}"/structured_results/*.json 2>/dev/null | head -n 1 || true)"
    echo "${bundle}"
}

record_bundle_metrics() {
    local bundle_path="$1"
    local category="$2"
    local variant="$3"
    local benchmark="$4"
    local model="$5"
    local split="$6"
    local router_path="$7"

    if [[ "${DRY_RUN}" == true ]]; then
        return 0
    fi

    if [[ -z "${bundle_path}" || ! -f "${bundle_path}" ]]; then
        echo "WARN: Missing bundle for metrics append: ${bundle_path}"
        return 0
    fi

    python - "${bundle_path}" "${MANIFEST_PATH}" "${METRICS_PATH}" \
        "${category}" "${variant}" "${benchmark}" "${model}" "${split}" "${router_path}" <<'PY'
import csv
import json
import os
import sys
from datetime import datetime

(
    bundle_path,
    manifest_path,
    metrics_path,
    category,
    variant,
    benchmark,
    model,
    split,
    router_path,
) = sys.argv[1:]

with open(bundle_path, "r", encoding="utf-8") as f:
    bundle = json.load(f)

manifest_row = {
    "timestamp": datetime.utcnow().isoformat(),
    "bundle_path": bundle_path,
    "category": category,
    "variant": variant,
    "benchmark": benchmark,
    "model": model,
    "split": split,
    "router_path": router_path,
}

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(manifest_row) + "\n")

rows = []
for er in bundle.get("eval_results", []):
    if er.get("worst_seed_sr") is None and er.get("cvar_failure") is None:
        continue
    rows.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "variant": variant,
            "benchmark": benchmark,
            "model": model,
            "split": split,
            "method": er.get("method"),
            "seed": er.get("seed"),
            "success_rate": er.get("success_rate"),
            "success_rate_ci_low": (er.get("success_rate_ci") or [None, None])[0],
            "success_rate_ci_high": (er.get("success_rate_ci") or [None, None])[1],
            "worst_seed_sr": er.get("worst_seed_sr"),
            "cvar_failure": er.get("cvar_failure"),
            "avg_cost": er.get("avg_cost"),
            "llm_call_rate": er.get("llm_call_rate"),
            "ece": er.get("ece"),
            "brier": er.get("brier"),
            "alpha": None,
            "epsilon": None,
            "bundle_path": bundle_path,
        }
    )

if rows:
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
PY
}

build_overrides_array() {
    local benchmark="$1"
    local -n out_arr="$2"

    out_arr=("project.seed=${SEED}")
    if [[ "${benchmark}" == "terminalbench" ]]; then
        out_arr+=("benchmark=terminalbench" "data.benchmark=terminalbench")
    fi
    if [[ -n "${EXTRA_OVERRIDES}" ]]; then
        local extra
        for extra in ${EXTRA_OVERRIDES}; do
            out_arr+=("${extra}")
        done
    fi
}

run_eval_and_record() {
    local benchmark="$1"
    local model="$2"
    local category="$3"
    local record_variant="$4"
    local dataset_variant="$5"
    local eval_dir="$6"
    shift 6
    local methods=("$@")

    local config
    config="$(config_for_benchmark "${benchmark}")"

    local overrides=()
    build_overrides_array "${benchmark}" overrides

    local cmd=(
        python scripts/evaluate.py
        --config "${config}"
        --features "${DATASET_PATH}"
        --router-path "${ROUTER_OUTPUT}/router_final.pt"
        --output "${eval_dir}"
        --split test
        --benchmark-filter "${benchmark}"
        --model-filter "${model}"
        --variant-filter "${dataset_variant}"
        --methods "${methods[@]}"
        --overrides "${overrides[@]}"
    )

    run_cmd "Evaluate ${category}/${record_variant} [${benchmark}/${model}]" "${cmd[@]}"

    local bundle
    bundle="$(latest_bundle_in_eval_dir "${eval_dir}")"
    record_bundle_metrics "${bundle}" "${category}" "${record_variant}" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"
}

echo
echo "Single-router dataset-first experiment pipeline"
echo "  features_root: ${FEATURES_ROOT}"
echo "  dataset_path:  ${DATASET_PATH}"
echo "  router_output: ${ROUTER_OUTPUT}"
echo "  results_root:  ${RESULTS_ROOT}"
echo "  dry_run:       ${DRY_RUN}"
echo "  train_extra_routers: ${TRAIN_EXTRA_ROUTERS}"

if [[ "${SKIP_BUILD}" != true ]]; then
    run_cmd "Build unified router dataset (Parquet)" \
        python scripts/build_router_dataset.py \
        --features-root "${FEATURES_ROOT}" \
        --output "${DATASET_PATH}" \
        --seed "${SEED}" \
        --max-perturbations-per-task "${MAX_PERTURBATIONS_PER_TASK}"
fi

if [[ "${SKIP_TRAIN}" != true ]]; then
    if [[ -f "${ROUTER_OUTPUT}/router_final.pt" && "${FORCE_RETRAIN}" != true ]]; then
        echo "INFO: Reusing existing single router: ${ROUTER_OUTPUT}/router_final.pt"
    else
        mkdir -p "${ROUTER_OUTPUT}"
        train_overrides=("project.seed=${SEED}")
        if [[ -n "${EXTRA_OVERRIDES}" ]]; then
            for extra in ${EXTRA_OVERRIDES}; do
                train_overrides+=("${extra}")
            done
        fi
        run_cmd "Train single router on unified dataset" \
            python scripts/train_router.py \
            --config "${TRAIN_CONFIG}" \
            --features "${DATASET_PATH}" \
            --output "${ROUTER_OUTPUT}" \
            --train-split train \
            --val-split val \
            --feature-transform none \
            --overrides "${train_overrides[@]}"

        if [[ "${TRAIN_EXTRA_ROUTERS}" == true ]]; then
            run_cmd "Train no-entropy router on unified dataset" \
                python scripts/train_router.py \
                --config "${TRAIN_CONFIG}" \
                --features "${DATASET_PATH}" \
                --output "${ROUTER_OUTPUT}_no_entropy" \
                --train-split train \
                --val-split val \
                --feature-transform no_entropy \
                --overrides "${train_overrides[@]}"

            run_cmd "Train verifier-pseudo-entropy router on unified dataset" \
                python scripts/train_router.py \
                --config "${TRAIN_CONFIG}" \
                --features "${DATASET_PATH}" \
                --output "${ROUTER_OUTPUT}_verifier_pseudo_entropy" \
                --train-split train \
                --val-split val \
                --feature-transform verifier_pseudo_entropy \
                --overrides "${train_overrides[@]}"
        fi
    fi
fi

if [[ "${SKIP_EVAL}" != true ]]; then
    if [[ "${DRY_RUN}" == true && ! -e "${DATASET_PATH}" ]]; then
        mapfile -t combos < <(
            python - "${FEATURES_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(r"^(?P<b>.+?)_noisy_router_features_(?P<v>.+)_(?P<m>[^_]+)\.jsonl$")
rows = set()
for p in sorted(root.glob("*_noisy_router_features_*.jsonl")):
    m = pat.match(p.name)
    if not m:
        continue
    b = m.group("b")
    v = m.group("v")
    model = m.group("m")
    category = "main" if v == "heuristic" else "ablation"
    rows.add((b, model, v, category))
for b, model, v, category in sorted(rows):
    print(f"{b}\t{model}\t{v}\t{category}")
PY
        )
    else
        mapfile -t combos < <(
            python - "${DATASET_PATH}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path, columns=["benchmark", "model", "variant", "category", "split"])
if "split" in df.columns:
    df = df[df["split"] == "test"]
uniq = (
    df[["benchmark", "model", "variant", "category"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["benchmark", "model", "variant"])
)
for row in uniq.itertuples(index=False):
    print(f"{row.benchmark}\t{row.model}\t{row.variant}\t{row.category}")
PY
        )
    fi

    for line in "${combos[@]}"; do
        IFS=$'\t' read -r benchmark model variant category <<< "${line}"

        run_root="${RESULTS_ROOT}/runs/${benchmark}/${model}"
        mkdir -p "${run_root}"

        if [[ "${variant}" == "heuristic" ]]; then
            # Main results.
            run_eval_and_record "${benchmark}" "${model}" "main" "main_results" "heuristic" \
                "${run_root}/eval/main_results" \
                r2v slm_only llm_only entropy_router

            # Feature ablations from the same single trained router.
            feature_base_dir="${run_root}/eval/feature_ablations"
            mkdir -p "${feature_base_dir}"
            overrides=()
            build_overrides_array "${benchmark}" overrides

            run_cmd "Feature ablation all_features [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "$(config_for_benchmark "${benchmark}")" \
                --features "${DATASET_PATH}" \
                --router-path "${ROUTER_OUTPUT}/router_final.pt" \
                --output "${feature_base_dir}/all_features" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "heuristic" \
                --methods r2v \
                --overrides "${overrides[@]}"
            bundle="$(latest_bundle_in_eval_dir "${feature_base_dir}/all_features")"
            record_bundle_metrics "${bundle}" "feature_ablation" "all_features" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"

            run_cmd "Feature ablation entropy_only [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "$(config_for_benchmark "${benchmark}")" \
                --features "${DATASET_PATH}" \
                --router-path "${ROUTER_OUTPUT}/router_final.pt" \
                --output "${feature_base_dir}/entropy_only" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "heuristic" \
                --methods r2v \
                --feature-mask 0 \
                --overrides "${overrides[@]}"
            bundle="$(latest_bundle_in_eval_dir "${feature_base_dir}/entropy_only")"
            record_bundle_metrics "${bundle}" "feature_ablation" "entropy_only" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"

            run_cmd "Feature ablation no_verifier [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "$(config_for_benchmark "${benchmark}")" \
                --features "${DATASET_PATH}" \
                --router-path "${ROUTER_OUTPUT}/router_final.pt" \
                --output "${feature_base_dir}/no_verifier" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "heuristic" \
                --methods r2v \
                --feature-mask 0 6 7 8 9 10 11 12 13 14 \
                --overrides "${overrides[@]}"
            bundle="$(latest_bundle_in_eval_dir "${feature_base_dir}/no_verifier")"
            record_bundle_metrics "${bundle}" "feature_ablation" "no_verifier" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"

            run_cmd "Feature ablation verifier_plus_entropy [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "$(config_for_benchmark "${benchmark}")" \
                --features "${DATASET_PATH}" \
                --router-path "${ROUTER_OUTPUT}/router_final.pt" \
                --output "${feature_base_dir}/verifier_plus_entropy" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "heuristic" \
                --methods r2v \
                --feature-mask 0 1 2 3 4 5 \
                --overrides "${overrides[@]}"
            bundle="$(latest_bundle_in_eval_dir "${feature_base_dir}/verifier_plus_entropy")"
            record_bundle_metrics "${bundle}" "feature_ablation" "verifier_plus_entropy" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"

            run_cmd "Feature ablation no_entropy_all_features [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "$(config_for_benchmark "${benchmark}")" \
                --features "${DATASET_PATH}" \
                --router-path "${ROUTER_OUTPUT}/router_final.pt" \
                --output "${feature_base_dir}/no_entropy_all_features" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "heuristic" \
                --methods r2v \
                --feature-mask 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
                --overrides "${overrides[@]}"
            bundle="$(latest_bundle_in_eval_dir "${feature_base_dir}/no_entropy_all_features")"
            record_bundle_metrics "${bundle}" "feature_ablation" "no_entropy_all_features" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"

            if is_closed_source_model "${model}"; then
                run_eval_and_record "${benchmark}" "${model}" "feature_ablation" "verifier_driven_pseudo_entropy" "heuristic" \
                    "${feature_base_dir}/verifier_driven_pseudo_entropy" \
                    verifier_router
            fi

            # Figure run for SR vs LLM% plot.
            fig_dir="${run_root}/eval/figure_method_sweep"
            run_cmd "Figure sweep [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "$(config_for_benchmark "${benchmark}")" \
                --features "${DATASET_PATH}" \
                --router-path "${ROUTER_OUTPUT}/router_final.pt" \
                --output "${fig_dir}" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "heuristic" \
                --methods r2v slm_only llm_only entropy_router \
                --router-threshold-sweep "${THRESHOLDS[@]}" \
                --overrides "${overrides[@]}"
            bundle="$(latest_bundle_in_eval_dir "${fig_dir}")"
            record_bundle_metrics "${bundle}" "figure" "method_and_threshold_sweep" "${benchmark}" "${model}" "test" "${ROUTER_OUTPUT}/router_final.pt"
        fi

        # Main ablations from all non-heuristic dataset variants.
        if [[ "${variant}" != "heuristic" ]]; then
            run_eval_and_record "${benchmark}" "${model}" "main_ablation" "${variant}" "${variant}" \
                "${run_root}/eval/main_ablation_${variant}" \
                r2v slm_only llm_only entropy_router
        fi
    done

    run_cmd "Generate plots and LaTeX/CSV tables" \
        python scripts/plot_router_experiments.py --results-root "${RESULTS_ROOT}"
fi

echo
echo "Pipeline complete"
echo "  Unified dataset: ${DATASET_PATH}"
echo "  Single router:    ${ROUTER_OUTPUT}/router_final.pt"
if [[ "${TRAIN_EXTRA_ROUTERS}" == true ]]; then
    echo "  No-entropy router: ${ROUTER_OUTPUT}_no_entropy/router_final.pt"
    echo "  Pseudo-entropy router: ${ROUTER_OUTPUT}_verifier_pseudo_entropy/router_final.pt"
fi
echo "  Manifest:         ${MANIFEST_PATH}"
echo "  Metrics:          ${METRICS_PATH}"
