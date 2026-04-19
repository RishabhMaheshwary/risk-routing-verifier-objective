#!/bin/bash
# Launch multi-GPU router-feature generation with episode sharding.
#
# Each GPU gets its own process handling a non-overlapping slice of episodes.
# Both the policy and verifier are loaded independently on each GPU.
#
# Usage:
#   bash scripts/launch_router_features.sh <NUM_GPUS> [generate_router_features.py args...]
#
# Example:
#   bash scripts/launch_router_features.sh 4 \
#       --config configs/gaia/noisy.yaml \
#       --policy-path outputs/policy/gaia_noisy/final \
#       --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
#       --output data/router_features/gaia.jsonl \
#       --batch-size 4 --K 5
#
# After all shards finish the script auto-merges.  You can also merge manually:
#   python scripts/generate_router_features.py --merge \
#       --output data/router_features/gaia.jsonl
#
# Resume after a crash: just re-run the same command — each worker detects
# already-completed episodes in its shard file and skips them.

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <NUM_GPUS> [generate_router_features.py args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generate_router_features.py"

# Derive a run tag from --output so log names are unique per model/run.
RUN_TAG="router_features"
ARGS_ARRAY=("$@")
for ((j=0; j<${#ARGS_ARRAY[@]}; j++)); do
    if [[ "${ARGS_ARRAY[$j]}" == "--output" ]] && ((j+1 < ${#ARGS_ARRAY[@]})); then
        RUN_TAG="$(basename "${ARGS_ARRAY[$((j+1))]}" .jsonl)"
        break
    fi
done

echo "=== Launching ${NUM_GPUS} router-feature workers (${RUN_TAG}) ==="
echo "Args: $@"
echo ""

# GPU_IDS env var overrides default 0..N-1 assignment (e.g. GPU_IDS=1,2,3,4)
if [[ -n "${GPU_IDS:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
    if [[ ${#GPU_LIST[@]} -ne ${NUM_GPUS} ]]; then
        echo "ERROR: GPU_IDS has ${#GPU_LIST[@]} entries but NUM_GPUS=${NUM_GPUS}"
        exit 1
    fi
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
fi

PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    LOG="${RUN_TAG}_shard${i}.log"
    echo "[GPU ${GPU_LIST[$i]}] Starting shard ${i}/${NUM_GPUS} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python "${GEN_SCRIPT}" \
        --shard-id ${i} \
        --num-shards ${NUM_GPUS} \
        "$@" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "[GPU ${GPU_LIST[$i]}] PID=${PIDS[-1]}"
done

echo ""
echo "All ${NUM_GPUS} workers launched.  PIDs: ${PIDS[*]}"
echo "Monitor progress:  tail -f ${RUN_TAG}_shard*.log"
echo ""

# Wait for all workers
FAILED=0
for ((i=0; i<NUM_GPUS; i++)); do
    if wait ${PIDS[$i]}; then
        echo "[GPU ${i}] ✓ Completed (PID ${PIDS[$i]})"
    else
        echo "[GPU ${i}] ✗ FAILED (PID ${PIDS[$i]}, exit=$?)"
        FAILED=$((FAILED+1))
    fi
done

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "ERROR: ${FAILED} shard(s) failed. Check logs before merging."
    exit 1
fi

# Generate-only writes *.shard_NNN.gen_cache.jsonl (no per-shard feature JSONL yet).
# Merging only applies to scored *.shard_NNN.jsonl files (full run or after --score-only).
GENERATE_ONLY=0
for a in "$@"; do
    if [[ "$a" == "--generate-only" ]]; then
        GENERATE_ONLY=1
        break
    fi
done

if [[ "${GENERATE_ONLY}" -eq 1 ]]; then
    echo ""
    echo "=== Generate-only finished (no merge). Candidate caches:"
    echo "    ${RUN_TAG}.shard_*.gen_cache.jsonl"
    echo "Run CPU scoring:  bash run_score_cpu.sh --model <shortname>"
    echo "Then merge runs automatically after --score-only, or:"
    echo "  python scripts/generate_router_features.py --merge --output <same --output path>"
    exit 0
fi

# Auto-merge (full two-phase or post-scoring shard outputs)
echo ""
echo "=== All shards complete, merging... ==="

# Find the --output arg value from the original args
OUTPUT=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--output" ]] && ((i+1 < ${#ARGS[@]})); then
        OUTPUT="${ARGS[$((i+1))]}"
        break
    fi
done

if [ -n "${OUTPUT}" ]; then
    python "${GEN_SCRIPT}" --merge --output "${OUTPUT}"
    echo "=== Done! Final output: ${OUTPUT} ==="
else
    echo "Could not find --output arg; merge manually:"
    echo "  python scripts/generate_router_features.py --merge --output <path>"
fi
