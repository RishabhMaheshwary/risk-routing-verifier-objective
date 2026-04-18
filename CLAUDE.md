# Project context for Claude

## What this repo is
`risk-routing-verifier-objective` — a pipeline for training small-model policies with
behaviour cloning (BC) + DPO on **noisy** trajectories, then extracting features for a
router that chooses between a fast local policy and a larger fallback model. Supported
benchmarks: humaneval, textworld, terminalbench (in progress), webarena, gaia, alfworld.

## Cluster setup
- Pod: Run:ai interactive job `rishabh-interactive-router-4gpu-0-0` (or `-8gpu-` variants)
- Project path on pod: `/mnt/queue4/rishabh/risk-routing-verifier-objective`
- Python env: `source .venv/bin/activate`
- GPUs: NVIDIA H100 80GB HBM3, CUDA 12.8
- Branch: `humaneval_exp` (fork `RishabhMaheshwary/...`, upstream `RaghuHemadri/...`)
- Remotes: `origin` = upstream (RaghuHemadri), `fork` = RishabhMaheshwary
- Push with: `git push fork humaneval_exp`

## Must-do: nohup for long jobs
**Always** launch cluster training/inference as:
```bash
nohup bash <script> > logs/<name>.log 2>&1 &
echo "PID: $!"
```
Bare `bash ...` gets SIGHUP'd when the interactive session closes. This has bitten us
multiple times (qwen14 and llama BC runs died mid-epoch).

## Pipeline stages (per model)
Driver: `run_pipeline_updated.sh --model <short> --benchmark <bench> --gpus <N> [--from <S>|--only <S>]`

1. **BC** (`--stage bc`): supervised train on successful trajectory steps.
   Output: `outputs/policy/<bench>_noisy_bc_<MODEL_TAG>/{best,final,epoch_N}`
2. **Collect preferences** (`scripts/launch_candidates.sh`): K candidates per
   trajectory, scored by heuristic verifier. Output:
   `data/candidates/<bench>_noisy_dpo_prefs_heuristic_<model>.jsonl`
3. **DPO** (`--stage preference`): preference fine-tuning on top of BC checkpoint.
   Output: `outputs/policy/<bench>_noisy_dpo_<model>/final`
4. **Router features** (`scripts/run_router_features_humaneval.sh`): 24-dim feature
   vectors from the DPO policy × heuristic verifier. Output:
   `data/router_features/<bench>_noisy_router_features_heuristic_<model>.jsonl`

Stage 4 script name says "humaneval" but is benchmark-agnostic (all paths via env vars).

## Model short names → HF IDs
| short    | HF ID                                      | BC dir tag              |
|----------|--------------------------------------------|-------------------------|
| qwen7    | Qwen/Qwen2.5-Coder-7B-Instruct             | qwen_coder_7b           |
| qwen14   | Qwen/Qwen2.5-Coder-14B-Instruct            | qwen_coder_14b          |
| llama    | meta-llama/Llama-3.1-8B-Instruct           | llama_3_1_8b_instruct   |
| gemma    | google/gemma-2-9b-it                       | gemma_2_9b_it           |
| deepseek | deepseek-ai/deepseek-coder-6.7b-instruct   | deepseek_coder_6_7b     |

Registry lives in `run_pipeline_updated.sh:114-129` (HF_ID + BC_DIR_TAG).

## Launcher scripts
All are 8-GPU sequential launchers designed for one model at a time on one node.
- `scripts/bc_job1.sh` — BC for llama → deepseek → qwen7
- `scripts/bc_job2.sh` — BC for qwen14 → gemma
- `scripts/pipeline_job1.sh` — stages 2→4 for llama, deepseek, qwen7
- `scripts/pipeline_job2.sh` — stages 2→4 for qwen14, gemma

## Data locations (terminalbench)
- Trajectories: `data/trajectories/terminalbench_noisy/trajectories.jsonl` (~6936 eps)
- BC splits: `data/trajectories/terminalbench_noisy/bc_{train,val}.jsonl`
  (1206 train eps across 22 tasks, 282 val eps; 17456 train / 6114 val BCDataset rows)
- BC split is `--success-only --data-fraction 0.4 --max-perturbations-per-task 2`

## Config
- `configs/terminalbench/noisy.yaml` inherits `../base`, now includes a `training:`
  section (bc.epochs=5, bc.batch_size=1, bc.grad_accum=8, bc.lr=1.5e-4; preference.*)
- `run_pipeline_updated.sh` derives `CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"`
  **after** arg parsing (line 150). Earlier bug: was at line 43 and always used humaneval.

## BC training: expected numbers
- On 8× H100 80GB, 7–9B models: ~1.2–1.5 sec/step, ~2.3 h for 5 epochs (~6815 steps)
- qwen14 (14B): ~1.9 sec/step after warmup, ~3.5 h for 5 epochs
- Logs print `[BC] Epoch N/5 Step S/1363 Loss=X`. Steps-per-epoch = 1363 on 8 GPUs.

## Known blocker: terminalbench heuristic verifier doesn't exist
Pipeline Job 2 stage 2 failed for qwen14 + gemma. Root cause:
- `r2v/data/labeling.py:360` `create_labeler()` only knows webarena/gaia/alfworld/humaneval
- `r2v/models/heuristic_verifier.py` only has humaneval + textworld scoring

Options when resuming:
1. **Add a TerminalBenchLabeler** + heuristic scorer (exit codes, file presence, task
   grading script). Dozens → hundreds of lines.
2. **Swap to LLM-judge or learned verifier** via `verifier.mode` override — config-only.
3. **Use trajectory `success` label directly** as the preference signal — cheapest.

Pick a path before re-running `pipeline_job{1,2}.sh` stage 2 onward.

Before doing anything: read the actual shard log to confirm the failure:
```bash
tail -100 terminalbench_noisy_dpo_prefs_heuristic_qwen14_shard0.log
```

## Run:ai watchdog
Low GPU util during CPU-heavy heuristic scoring has previously triggered job
termination. `run_pipeline_updated.sh:142-147` sets
`PREF_GPU_KEEPALIVE_INTERVAL=0.2` and `DPO_GPU_KEEPALIVE_INTERVAL=0.2` for qwen14
by default; override with `--pref-gpu-keepalive-interval` / `--dpo-gpu-keepalive-interval`.

## Ablation sweep
After all 5 models have stages 1–4 done:
```bash
bash run_ablations_terminalbench.sh --gpus 4
```
Synced with upstream: includes `ensure_pref_val_half`, VERIFIER_OVERRIDE,
BC best/final fallback, GPU_KEEPALIVE for qwen14/gemma, `--skip-bc` flag.

## Monitoring
Stage-transition log (top level): `tail -f logs/pipeline_job{1,2}.log`
Per-stage detail: `tail -f logs/{preferences,dpo,router_features}_terminalbench_<model>.log`
Shard logs (stage 2): `tail -f terminalbench_noisy_dpo_prefs_heuristic_<model>_shard*.log` (written to repo root)
