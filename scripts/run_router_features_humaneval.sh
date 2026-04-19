#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Router Feature Extraction — HumanEval (Heuristic Verifier)
# ══════════════════════════════════════════════════════════════
#
# Generates 24-dim feature vectors from the DPO-trained policy
# and the heuristic (rule-based, execution-backed) verifier.
#
# Usage:
#   bash scripts/run_router_features_humaneval.sh
#   bash scripts/run_router_features_humaneval.sh --dry-run
#   K=3 BATCH_SIZE=2 bash scripts/run_router_features_humaneval.sh
#
# Output: data/router_features/humaneval_noisy_heuristic.jsonl
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configurable knobs ────────────────────────────────────────
POLICY_PATH="${POLICY_PATH:-outputs/policy/humaneval_noisy_dpo/final}"
TRAJECTORIES="${TRAJECTORIES:-data/trajectories/humaneval_noisy/trajectories.jsonl}"
CONFIG="${CONFIG:-configs/humaneval/noisy.yaml}"
OUTPUT="${OUTPUT:-data/router_features/humaneval_noisy_heuristic.jsonl}"
K="${K:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"
# Tiny periodic GPU matmul during CPU-heavy heuristic scoring (HPC low-util watchdogs)
GPU_KEEPALIVE_INTERVAL="${GPU_KEEPALIVE_INTERVAL:-2}"
GENERATE_ONLY="${GENERATE_ONLY:-false}"
SCORE_ONLY="${SCORE_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }
header(){ echo -e "\n${BOLD}═══ $* ═══${NC}"; }

# Handle --dry-run flag
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Pre-flight checks ────────────────────────────────────────
header "Pre-flight checks"

ERRORS=0

# 1. Config
if [[ -f "$CONFIG" ]]; then
    ok "Config: $CONFIG"
else
    err "Config not found: $CONFIG"; ((ERRORS++))
fi

# 2. Policy checkpoint (local dir or HuggingFace model ID)
if [[ -d "$POLICY_PATH" ]]; then
    ADAPTER_FILES=$(ls "$POLICY_PATH"/adapter_model* 2>/dev/null | wc -l)
    ok "Policy: $POLICY_PATH  (${ADAPTER_FILES} adapter file(s))"
    if [[ $ADAPTER_FILES -eq 0 ]]; then
        warn "No adapter_model* files — make sure the DPO checkpoint is complete"
    fi
elif [[ "$POLICY_PATH" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_./-]+$ ]]; then
    ok "Policy: $POLICY_PATH  (HuggingFace model ID — will be downloaded at runtime)"
else
    err "Policy dir not found: $POLICY_PATH"; ((ERRORS++))
fi

# 3. Trajectories
if [[ -f "$TRAJECTORIES" ]]; then
    N_LINES=$(wc -l < "$TRAJECTORIES")
    ok "Trajectories: $TRAJECTORIES  (${N_LINES} lines)"
else
    err "Trajectories not found: $TRAJECTORIES"; ((ERRORS++))
fi

# 4. GPU
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')")
    ok "GPU: ${GPU_NAME} (${GPU_MEM})"
else
    warn "No GPU detected — will be extremely slow"
fi

# 5. Existing output
if [[ -f "$OUTPUT" ]]; then
    EXISTING=$(wc -l < "$OUTPUT")
    warn "Output file exists with ${EXISTING} records — will RESUME (append)"
else
    info "Output: $OUTPUT (fresh run)"
fi

# Abort if critical checks failed
if [[ $ERRORS -gt 0 ]]; then
    err "Pre-flight failed ($ERRORS error(s)). Fix the above and re-run."
    exit 1
fi

# ── Score-only: how many candidate caches exist (from prior generate phase) ──
NUM_SCORE_SHARDS=0
if [[ "$SCORE_ONLY" == "true" ]]; then
    OUT_DIR="$(dirname "$OUTPUT")"
    OUTPUT_STEM="$(basename "$OUTPUT" .jsonl)"
    shopt -s nullglob
    CACHE_SHARD_FILES=( "$OUT_DIR/${OUTPUT_STEM}.shard_"*.gen_cache.jsonl )
    shopt -u nullglob
    if (( ${#CACHE_SHARD_FILES[@]} > 0 )); then
        NUM_SCORE_SHARDS=${#CACHE_SHARD_FILES[@]}
    elif [[ -f "$OUT_DIR/${OUTPUT_STEM}.gen_cache.jsonl" ]]; then
        NUM_SCORE_SHARDS=1
    else
        err "Score-only: no candidate cache found. Expected either:"
        err "  $OUT_DIR/${OUTPUT_STEM}.shard_*.gen_cache.jsonl"
        err "  or  $OUT_DIR/${OUTPUT_STEM}.gen_cache.jsonl"
        err "Run GPU generate-only (pipeline stage 4) first."
        exit 1
    fi
fi

# ── Print run config ─────────────────────────────────────────
header "Run configuration"
echo -e "  Policy:        ${BOLD}${POLICY_PATH}${NC}"
echo -e "  Trajectories:  ${TRAJECTORIES}"
echo -e "  Config:        ${CONFIG}"
echo -e "  Output:        ${OUTPUT}"
echo -e "  K (candidates):${K}"
echo -e "  Batch size:    ${BATCH_SIZE}"
if [[ "$SCORE_ONLY" == "true" ]]; then
    echo -e "  Backend:       ${BOLD}none${NC} (score-only: no vLLM)"
    echo -e "  Verifier:      ${BOLD}heuristic${NC} (rule-based, execution-backed)"
    echo -e "  Mode:          ${BOLD}SCORE-ONLY${NC} (sequential CPU; one process per cache shard)"
    echo -e "  Cache shards:  ${NUM_SCORE_SHARDS}  (auto-detected; logs stream here, no *_shard*.log)"
elif [[ "$GENERATE_ONLY" == "true" ]]; then
    echo -e "  Batch note:    ${BATCH_SIZE}  (vLLM continuous-batching)"
    echo -e "  Backend:       ${BOLD}vLLM${NC} (PagedAttention + continuous batching)"
    echo -e "  Verifier:      ${BOLD}heuristic${NC} (rule-based, execution-backed)"
    echo -e "  GPUs / launch: ${NUM_GPUS}  (multi-GPU uses scripts/launch_router_features.sh)"
    echo -e "  Mode:          ${BOLD}GENERATE-ONLY${NC} (GPU at 100%, no scoring)"
else
    echo -e "  Batch note:    ${BATCH_SIZE}  (vLLM continuous-batching)"
    echo -e "  Backend:       ${BOLD}vLLM${NC} (PagedAttention + continuous batching)"
    echo -e "  Verifier:      ${BOLD}heuristic${NC} (rule-based, execution-backed)"
    echo -e "  GPUs / launch: ${NUM_GPUS}"
fi

# Verifier override: force heuristic mode
VERIFIER_OVERRIDE="verifier.mode=heuristic verifier.heuristic.run_code=true verifier.heuristic.benchmark=${BENCHMARK:-humaneval}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# ── Sanity probe: load 1 trajectory and print stats ──────────
header "Trajectory sanity check"
python -c "
import json, sys
from collections import Counter

path = '$TRAJECTORIES'
n_episodes = 0
n_steps = 0
successes = 0
sources = Counter()
pert_types = Counter()

with open(path) as f:
    for line in f:
        r = json.loads(line)
        # Each line is a full episode record with episode_id, steps, success, etc.
        if 'episode_id' not in r:
            continue
        n_episodes += 1
        steps = r.get('steps', [])
        n_steps += len(steps)
        if r.get('success'):
            successes += 1
        for s in steps:
            sources[s.get('action_source', 'unknown')] += 1
            pt = s.get('perturbation_type', 'none')
            pert_types[pt] += 1

print(f'  Episodes:     {n_episodes}')
print(f'  Total steps:  {n_steps}')
print(f'  SLM success:  {successes}/{n_episodes} ({100*successes/max(n_episodes,1):.1f}%)')
print(f'  Action sources: {dict(sources)}')
print(f'  Perturbation types (top 5): {dict(pert_types.most_common(5))}')
print()
print(f'  Expected output: ~{n_steps} feature records (one per step)')
print(f'  Estimated time:  ~{n_steps * 1.5 / 60:.0f}-{n_steps * 3.0 / 60:.0f} min (depends on GPU)')
" 2>&1 || warn "Could not parse trajectories for stats"

if [[ "$DRY_RUN" == "true" ]]; then
    header "DRY RUN — would execute the following"
    if [[ "$SCORE_ONLY" == "true" ]]; then
        echo "# Sequential CPU score-only (${NUM_SCORE_SHARDS} cache shard(s), stdout only):"
        for ((s = 0; s < NUM_SCORE_SHARDS; s++)); do
            echo "python scripts/generate_router_features.py \\"
            [[ "${ROUTER_NO_RESUME:-}" == "true" ]] && echo "    --no-resume \\"
            echo "    --config $CONFIG \\"
            echo "    --policy-path $POLICY_PATH \\"
            echo "    --trajectories $TRAJECTORIES \\"
            echo "    --output $OUTPUT \\"
            echo "    --batch-size $BATCH_SIZE --K $K \\"
            echo "    --gpu-keepalive-interval $GPU_KEEPALIVE_INTERVAL \\"
            echo "    --shard-id $s --num-shards $NUM_SCORE_SHARDS \\"
            echo "    --score-only \\"
            echo "    --overrides $VERIFIER_OVERRIDE \\"
            echo "        $EXTRA_OVERRIDES logging.wandb_mode=disabled"
            echo ""
        done
        if (( NUM_SCORE_SHARDS > 1 )); then
            echo "python scripts/generate_router_features.py --merge --output \"$OUTPUT\""
        fi
    else
        echo "python scripts/generate_router_features.py \\"
        [[ "$GENERATE_ONLY" == "true" ]] && echo "    --generate-only \\"
        echo "    --config $CONFIG \\"
        echo "    --policy-path $POLICY_PATH \\"
        echo "    --trajectories $TRAJECTORIES \\"
        echo "    --output $OUTPUT \\"
        echo "    --batch-size $BATCH_SIZE --K $K \\"
        echo "    --gpu-keepalive-interval $GPU_KEEPALIVE_INTERVAL \\"
        echo "    --overrides $VERIFIER_OVERRIDE \\"
        echo "        $EXTRA_OVERRIDES logging.wandb_mode=disabled"
    fi
    exit 0
fi

# ── Run feature generation ────────────────────────────────────
header "Generating router features"
info "Started at $(date '+%Y-%m-%d %H:%M:%S')"
info "Tail the output:  tail -f $OUTPUT | python -c \"import sys,json; [print(f'ep={json.loads(l).get(\\\"episode_id\\\",\\\"?\\\")[:20]}  step={json.loads(l).get(\\\"step_idx\\\",\\\"?\\\")}  features[0:3]={json.loads(l).get(\\\"features\\\",[])[:3]}') for l in sys.stdin]\""
echo ""

START_TIME=$(date +%s)

PHASE_FLAGS=""
[[ "$GENERATE_ONLY" == "true" ]] && PHASE_FLAGS="--generate-only"
[[ "$SCORE_ONLY" == "true" ]]    && PHASE_FLAGS="--score-only"

RESUME_FLAG=()
[[ "${ROUTER_NO_RESUME:-}" == "true" ]] && RESUME_FLAG=(--no-resume)

if [[ "$SCORE_ONLY" == "true" ]]; then
    info "Score-only: running ${NUM_SCORE_SHARDS} scorer(s) sequentially on CPU (single log stream)"
    for ((s = 0; s < NUM_SCORE_SHARDS; s++)); do
        info "── Scoring cache shard ${s}/${NUM_SCORE_SHARDS} (shard-id ${s}) ──"
        python scripts/generate_router_features.py \
            "${RESUME_FLAG[@]}" \
            --config "$CONFIG" \
            --policy-path "$POLICY_PATH" \
            --trajectories "$TRAJECTORIES" \
            --output "$OUTPUT" \
            --batch-size "$BATCH_SIZE" --K "$K" \
            --gpu-keepalive-interval "$GPU_KEEPALIVE_INTERVAL" \
            --shard-id "$s" \
            --num-shards "$NUM_SCORE_SHARDS" \
            $PHASE_FLAGS \
            --overrides \
                $VERIFIER_OVERRIDE \
                $EXTRA_OVERRIDES \
                logging.wandb_mode=disabled
    done
    if (( NUM_SCORE_SHARDS > 1 )); then
        info "Merging per-shard feature files → $OUTPUT"
        python scripts/generate_router_features.py --merge --output "$OUTPUT"
    fi
elif [[ "$NUM_GPUS" -gt 1 ]]; then
    info "Multi-GPU mode: ${NUM_GPUS} workers"
    bash scripts/launch_router_features.sh "$NUM_GPUS" \
        "${RESUME_FLAG[@]}" \
        --config "$CONFIG" \
        --policy-path "$POLICY_PATH" \
        --trajectories "$TRAJECTORIES" \
        --output "$OUTPUT" \
        --batch-size "$BATCH_SIZE" --K "$K" \
        --gpu-keepalive-interval "$GPU_KEEPALIVE_INTERVAL" \
        $PHASE_FLAGS \
        --overrides \
            $VERIFIER_OVERRIDE \
            $EXTRA_OVERRIDES \
            logging.wandb_mode=disabled

    if [[ "$GENERATE_ONLY" != "true" ]]; then
        info "Merging shards..."
        python scripts/generate_router_features.py --merge --output "$OUTPUT"
    fi
else
    python scripts/generate_router_features.py \
        "${RESUME_FLAG[@]}" \
        --config "$CONFIG" \
        --policy-path "$POLICY_PATH" \
        --trajectories "$TRAJECTORIES" \
        --output "$OUTPUT" \
        --batch-size "$BATCH_SIZE" --K "$K" \
        --gpu-keepalive-interval "$GPU_KEEPALIVE_INTERVAL" \
        $PHASE_FLAGS \
        --overrides \
            $VERIFIER_OVERRIDE \
            $EXTRA_OVERRIDES \
            logging.wandb_mode=disabled
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REMAINING=$(( ELAPSED % 60 ))

# ── Post-run analysis ─────────────────────────────────────────
header "Post-run analysis"

if [[ ! -f "$OUTPUT" ]]; then
    err "Output file was not created!"
    exit 1
fi

TOTAL_RECORDS=$(wc -l < "$OUTPUT")
ok "Generated ${TOTAL_RECORDS} feature records in ${MINUTES}m ${SECONDS_REMAINING}s"

python -c "
import json, sys
import numpy as np

path = '$OUTPUT'

features_all = []
entropies = []
v_means = []
v_spreads = []
v_bests = []
lp_bests = []
consistencies = []
sem_entropies = []
successes = []
episode_ids = set()

with open(path) as f:
    for line in f:
        r = json.loads(line)
        feats = r['features']
        features_all.append(feats)
        episode_ids.add(r.get('episode_id', '?'))
        successes.append(r.get('slm_success', 0))

        # Feature layout (24-dim):
        #  0: entropy
        #  1: verifier_score_spread
        #  2: verifier_score_mean
        #  3: verifier_score_std
        #  4: verifier_score_best
        #  5: verifier_score_worst
        #  6: log_prob_best
        #  7: log_prob_mean
        #  8: log_prob_std
        #  9: candidate_consistency
        # 10: semantic_entropy
        # 11: horizon_fraction
        # 12: step_number
        # 13: context_length_norm
        # 14: goal_length_norm
        # 15-18: benchmark_onehot
        # 19-23: perturbation_onehot
        entropies.append(feats[0])
        v_spreads.append(feats[1])
        v_means.append(feats[2])
        v_bests.append(feats[4])
        lp_bests.append(feats[6])
        consistencies.append(feats[9])
        sem_entropies.append(feats[10])

F = np.array(features_all)
dim = F.shape[1] if F.ndim == 2 else 0

print(f'  Total records:    {len(features_all)}')
print(f'  Unique episodes:  {len(episode_ids)}')
print(f'  Feature dim:      {dim}')
print(f'  SLM success rate: {np.mean(successes):.3f}')
print()

# ── Feature distribution summary ──
print('  Feature distributions (sanity check):')
print(f'  ┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┐')
print(f'  │ Feature                 │   Mean   │   Std    │   Min    │   Max    │')
print(f'  ├─────────────────────────┼──────────┼──────────┼──────────┼──────────┤')
names = ['SLM entropy', 'V-score spread', 'V-score mean', 'V-score std',
         'V-score best', 'V-score worst', 'LogProb best', 'LogProb mean',
         'LogProb std', 'Consistency', 'Sem. entropy', 'Horizon frac',
         'Step number', 'Context len', 'Goal len']
for i, name in enumerate(names):
    if i < dim:
        col = F[:, i]
        print(f'  │ {name:<23s} │ {np.mean(col):>8.4f} │ {np.std(col):>8.4f} │ {np.min(col):>8.4f} │ {np.max(col):>8.4f} │')
print(f'  └─────────────────────────┴──────────┴──────────┴──────────┴──────────┘')
print()

# ── Theory checks ──
print('  Theory validation checks:')
print('  ─────────────────────────────────────────────────────────────')

# 1. Entropy should be higher on failed episodes
ent = np.array(entropies)
succ = np.array(successes)
if succ.sum() > 0 and (1-succ).sum() > 0:
    ent_success = ent[succ == 1].mean()
    ent_failure = ent[succ == 0].mean()
    direction = '>' if ent_failure > ent_success else '<='
    verdict = 'PASS' if ent_failure > ent_success else 'CHECK'
    print(f'  [{verdict}] Entropy on failures ({ent_failure:.4f}) {direction} entropy on successes ({ent_success:.4f})')
    print(f'        Theory: higher entropy => more uncertain => more likely to fail')
else:
    print(f'  [SKIP] Cannot compare entropy by outcome (all same label)')

# 2. Verifier scores should be higher on successful episodes
vm = np.array(v_means)
if succ.sum() > 0 and (1-succ).sum() > 0:
    vm_success = vm[succ == 1].mean()
    vm_failure = vm[succ == 0].mean()
    direction = '>' if vm_success > vm_failure else '<='
    verdict = 'PASS' if vm_success > vm_failure else 'CHECK'
    print(f'  [{verdict}] V-score mean on successes ({vm_success:.4f}) {direction} failures ({vm_failure:.4f})')
    print(f'        Theory: higher verifier score => more confident => more success')

# 3. Verifier spread should be higher on failures (disagreement)
vs = np.array(v_spreads)
if succ.sum() > 0 and (1-succ).sum() > 0:
    vs_success = vs[succ == 1].mean()
    vs_failure = vs[succ == 0].mean()
    direction = '>' if vs_failure > vs_success else '<='
    verdict = 'PASS' if vs_failure > vs_success else 'CHECK'
    print(f'  [{verdict}] V-score spread on failures ({vs_failure:.4f}) {direction} successes ({vs_success:.4f})')
    print(f'        Theory: more spread => more disagreement among candidates => harder step')

# 4. Candidate consistency should be higher on successes
cons = np.array(consistencies)
if succ.sum() > 0 and (1-succ).sum() > 0:
    cons_success = cons[succ == 1].mean()
    cons_failure = cons[succ == 0].mean()
    direction = '>' if cons_success > cons_failure else '<='
    verdict = 'PASS' if cons_success > cons_failure else 'CHECK'
    print(f'  [{verdict}] Consistency on successes ({cons_success:.4f}) {direction} failures ({cons_failure:.4f})')
    print(f'        Theory: candidates agreeing more => policy is confident => success')

# 5. Semantic entropy should be lower on successes
se = np.array(sem_entropies)
if succ.sum() > 0 and (1-succ).sum() > 0:
    se_success = se[succ == 1].mean()
    se_failure = se[succ == 0].mean()
    direction = '>' if se_failure > se_success else '<='
    verdict = 'PASS' if se_failure > se_success else 'CHECK'
    print(f'  [{verdict}] Semantic entropy on failures ({se_failure:.4f}) {direction} successes ({se_success:.4f})')
    print(f'        Theory: diverse first tokens => more uncertain => harder')

# 6. Check feature variance — any constant features are useless
print()
print('  Feature variance check (constant features are useless for routing):')
for i, name in enumerate(names):
    if i < dim:
        std = np.std(F[:, i])
        status = 'OK' if std > 1e-6 else 'DEAD'
        if status == 'DEAD':
            print(f'    [{status}] {name}: std={std:.6f} -- NO SIGNAL')
        elif std < 0.01:
            print(f'    [LOW]  {name}: std={std:.6f} -- very low variance')

# 7. Correlation between entropy and verifier score
corr = np.corrcoef(ent, vm)[0, 1]
print()
print(f'  Correlation(entropy, v-score mean): {corr:.4f}')
print(f'    Expected: negative (high entropy => low verifier confidence)')
print(f'    Actual:   {\"negative (good)\" if corr < 0 else \"positive (unexpected -- check verifier calibration)\"}')

print()
print('  ─────────────────────────────────────────────────────────────')
print('  Done. If most checks show PASS, the features carry useful signal.')
print('  CHECK items may indicate the heuristic verifier behaves differently')
print('  than expected -- inspect those features manually.')
" 2>&1

ok "Feature file: $OUTPUT"
ok "Finished at $(date '+%Y-%m-%d %H:%M:%S') (${MINUTES}m ${SECONDS_REMAINING}s)"
echo ""
info "Next steps:"
echo "  1. Train router:   python scripts/train_router.py --config $CONFIG --features $OUTPUT --output outputs/router/humaneval_noisy_heuristic"
echo "  2. Evaluate:       python scripts/evaluate.py --config $CONFIG --features $OUTPUT --trajectories $TRAJECTORIES --router-path outputs/router/humaneval_noisy_heuristic/router_final.pt --output results/humaneval_noisy_heuristic --methods r2v slm_only llm_only entropy_router"
