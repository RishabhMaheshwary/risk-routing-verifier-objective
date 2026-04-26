#!/usr/bin/env python3
"""Generate candidate actions for DPO preference training.

Usage (single GPU):
    python scripts/generate_candidates_heuristic.py \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/candidates/gaia_noisy.jsonl \
        --K 5

Usage (multi-GPU via launch_candidates.sh — 4 GPUs):
    bash scripts/launch_candidates.sh 4 \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/candidates/gaia_noisy.jsonl \
        --K 5

Resume after a crash (just re-run the same command — skips completed episodes):
    # Same command as above; completed episode IDs are read from the output file.

Merge shards after all finish:
    python scripts/generate_candidates_heuristic.py --merge \
        --output data/candidates/gaia_noisy.jsonl

For each step in teacher trajectories, samples K candidates from the
SLM policy and scores them with a heuristic verifier to create preference pairs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import glob
import json
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import TrajectoryStore
from r2v.models.heuristic_verifier import HeuristicVerifier
from r2v.utils.async_writer import AsyncJSONLWriter
from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging


_keepalive_stop = threading.Event()


def _gpu_keepalive(device: str, interval: float):
    import torch

    # Tiny periodic matmul to keep GPU from appearing idle to cluster watchdogs.
    x = torch.randn((128, 128), device=device, dtype=torch.float16)
    while not _keepalive_stop.is_set():
        try:
            _ = x @ x
        except Exception:
            pass
        _keepalive_stop.wait(interval)


def start_gpu_keepalive(device: str, interval: float):
    _keepalive_stop.clear()
    t = threading.Thread(target=_gpu_keepalive, args=(device, interval), daemon=True)
    t.start()
    return t


def stop_gpu_keepalive():
    _keepalive_stop.set()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def completed_episode_ids(output_file: Path) -> set[str]:
    """Return set of episode_ids that have ALL their steps already done."""
    if not output_file.exists():
        return set()
    ep_steps: dict[str, set[int]] = {}
    with open(output_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
                eid = rec["episode_id"]
                ep_steps.setdefault(eid, set()).add(rec["step_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    # We can't know total steps per episode from the output alone,
    # so we return all episode_ids that appear at all (conservative: skip them).
    return set(ep_steps.keys())


def build_context_index(store: TrajectoryStore) -> dict[str, dict[str, list[str]]]:
    """First streaming pass: build a lightweight context index for consistency pairs.

    Returns:
        task_id -> {episode_id -> [context_at_step_0, context_at_step_1, ...]}
    """
    index: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for ep in store.iter_episodes():
        task_id = ep.metadata.task_id if ep.metadata else ep.episode_id
        contexts: list[str] = []
        ctx = ""
        for step in ep.steps:
            ctx += step.observation.raw_text + "\n"
            contexts.append(ctx)
            ctx += step.action.raw_text + "\n"
        index[task_id][ep.episode_id] = contexts
    return dict(index)


def iter_step_items(episodes, context_index: dict | None = None):
    """Yield step items lazily to avoid large pre-collection stalls/memory.

    If context_index is provided, each item may include ``context_alt`` from a
    different perturbation seed of the same task at the same step index.
    """
    for episode in episodes:
        goal = episode.metadata.goal if episode.metadata else ""
        task_id = episode.metadata.task_id if episode.metadata else episode.episode_id

        partner_contexts: list[str] | None = None
        if context_index is not None:
            task_eps = context_index.get(task_id, {})
            peers = sorted(eid for eid in task_eps if eid != episode.episode_id)
            if peers:
                partner_eid = peers[hash(episode.episode_id) % len(peers)]
                partner_contexts = task_eps[partner_eid]

        context = ""
        for step_idx, step in enumerate(episode.steps):
            context += step.observation.raw_text + "\n"
            item = {
                "context": context,
                "step_idx": step_idx,
                "goal": goal,
                "episode_id": episode.episode_id,
            }
            if partner_contexts is not None and step_idx < len(partner_contexts):
                item["context_alt"] = partner_contexts[step_idx]
            yield item
            context += step.action.raw_text + "\n"


def iter_sharded_episodes(store: TrajectoryStore, shard_id: int, num_shards: int, done_eids: set[str]):
    """Stream episodes for one shard directly from JSONL to keep RAM bounded."""
    for idx, ep in enumerate(store.iter_episodes()):
        if idx % num_shards != shard_id:
            continue
        if ep.episode_id in done_eids:
            continue
        yield ep


def count_sharded_steps_with_logging(
    store: TrajectoryStore,
    shard_id: int,
    num_shards: int,
    done_eids: set[str],
    logger,
    log_every_episodes: int = 200,
) -> tuple[int, int]:
    """Streaming count with progress logs so long scans don't look stalled."""
    total_steps = 0
    total_eps = 0
    scanned = 0
    t0 = time.time()
    every = max(1, log_every_episodes)

    for idx, ep in enumerate(store.iter_episodes()):
        scanned += 1
        if idx % num_shards != shard_id:
            continue
        if ep.episode_id in done_eids:
            continue
        total_eps += 1
        total_steps += len(ep.steps)

        if total_eps % every == 0:
            elapsed = max(time.time() - t0, 1e-6)
            logger.info(
                f"Shard scan progress: kept={total_eps} episodes, "
                f"steps={total_steps}, scanned={scanned}, "
                f"rate={scanned / elapsed:.1f} eps/s"
            )

    elapsed = max(time.time() - t0, 1e-6)
    logger.info(
        f"Shard scan complete: kept={total_eps} episodes, "
        f"steps={total_steps}, scanned={scanned}, "
        f"elapsed={elapsed:.1f}s"
    )
    return total_steps, total_eps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate actions")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument(
        "--heuristic-benchmark",
        type=str,
        choices=["humaneval", "textworld"],
        default=None,
        help=(
            "Force heuristic benchmark mode. "
            "Default: infer from config/trajectory context"
        ),
    )
    parser.add_argument(
        "--no-run-code",
        action="store_true",
        help=(
            "Disable subprocess code execution in heuristic HumanEval scoring "
            "(faster, but less faithful)"
        ),
    )
    parser.add_argument("--trajectories", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5, help="Number of candidates per step")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of steps to batch together for GPU inference")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max tokens to generate (default: config or 128)")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help=(
            "Max prompt tokens before left-truncation. "
            "Default: max_model_len - max_new_tokens"
        ),
    )
    parser.add_argument(
        "--gpu-keepalive-interval",
        type=float,
        default=1.5,
        help="Seconds between tiny GPU keepalive kernels (0 to disable)",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=3,
        help="Number of scorer/write worker threads (must be >=1)",
    )
    parser.add_argument(
        "--max-pending-batches",
        type=int,
        default=0,
        help=(
            "Hard cap on queued scorer futures before generation waits "
            "(0 = auto, derived from pipeline depth)"
        ),
    )
    parser.add_argument(
        "--verifier-batch-size",
        type=int,
        default=4,
        help="Max verifier scoring chunk size over flattened candidates",
    )
    parser.add_argument(
        "--store-all-candidates",
        action="store_true",
        help=(
            "Store full sorted candidate list in output JSONL. "
            "Disabled by default for faster generation and smaller files."
        ),
    )
    parser.add_argument(
        "--compute-logprobs",
        action="store_true",
        help="Compute candidate log-probabilities (slower; not needed for DPO pairs)",
    )
    # Multi-GPU sharding
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard outputs instead of generating")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume; overwrite existing output")
    parser.add_argument(
        "--write-queue-size",
        type=int,
        default=50000,
        help="Max pending JSON records buffered by async writer (0 = unbounded)",
    )
    parser.add_argument(
        "--write-flush-every",
        type=int,
        default=1024,
        help="Async writer flush interval in records",
    )
    parser.add_argument(
        "--no-consistency-pairs",
        action="store_true",
        help="Skip context index pass; omit context_alt from output pairs",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument(
        "--scan-log-every-episodes",
        type=int,
        default=200,
        help="Log shard-scan progress every N kept episodes",
    )
    return parser.parse_args()


def merge_shards(output_path: str, logger):
    """Merge all shard files into the final output."""
    output = Path(output_path)
    shard_pattern = str(output.parent / f"{output.stem}.shard_*{output.suffix}")
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        logger.error(f"No shard files found matching {shard_pattern}")
        sys.exit(1)

    total = 0
    with open(output, "w") as out_f:
        for sf in shard_files:
            with open(sf) as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1
            logger.info(f"  Merged {sf}")

    logger.info(f"Merged {len(shard_files)} shards → {output} ({total} pairs)")


def _score_and_write_batch(
    verifier,
    writer: AsyncJSONLWriter,
    batch: list[dict],
    all_candidates: list[list[dict]],
    store_all_candidates: bool,
    verifier_batch_size: int,
    logger=None,
) -> int:
    """Score one generated batch and enqueue preference pairs for async JSON write."""
    B = len(batch)
    K = len(all_candidates[0]) if B > 0 and all_candidates[0] else 0

    flat_ctx: list[str] = []
    flat_cand: list[str] = []
    flat_goal: list[str] = []
    for b in range(B):
        for c in all_candidates[b]:
            flat_ctx.append(batch[b]["context"])
            flat_cand.append(c["text"])
            flat_goal.append(batch[b]["goal"])

    total = len(flat_ctx)
    flat_scores: list[float] = []
    chunk = max(1, verifier_batch_size)
    start = 0
    while start < total:
        cur = min(chunk, total - start)
        try:
            scores = verifier.score_batch(
                flat_ctx[start : start + cur],
                flat_cand[start : start + cur],
                flat_goal[start : start + cur],
            )
            flat_scores.extend(scores)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Verifier scoring failed for a chunk; using 0.5 fallback. "
                    f"Error: {exc}"
                )
            flat_scores.extend([0.5] * cur)
        start += cur

    all_scores = [
        flat_scores[b * K : (b + 1) * K] for b in range(B)
    ]

    written = 0
    for i, item in enumerate(batch):
        scored = [
            {
                "action": c["text"],
                "log_prob": c["log_prob"],
                "verifier_score": sc,
            }
            for c, sc in zip(all_candidates[i], all_scores[i])
        ]

        if len(scored) >= 2:
            # Pick best normally.
            best = max(scored, key=lambda x: x["verifier_score"])

            # For rejected, prefer the lowest-scored candidate with an action text
            # different from chosen. This avoids degenerate chosen==rejected pairs
            # when scores tie or verifier falls back to constant values.
            ranked_worst_first = sorted(scored, key=lambda x: x["verifier_score"])
            worst = None
            for cand in ranked_worst_first:
                if cand["action"] != best["action"]:
                    worst = cand
                    break

            # If all candidate texts are identical, skip this step pair.
            if worst is None:
                continue

            pair = {
                "context": item["context"],
                "chosen": best["action"],
                "rejected": worst["action"],
                "chosen_score": best["verifier_score"],
                "rejected_score": worst["verifier_score"],
                "episode_id": item["episode_id"],
                "step_idx": item["step_idx"],
            }
            if "context_alt" in item:
                pair["context_alt"] = item["context_alt"]
            if store_all_candidates:
                pair["all_candidates"] = sorted(
                    scored, key=lambda x: x["verifier_score"], reverse=True
                )
            writer.write(pair)
            written += 1

    return written


def _iter_batches(item_iter, batch_size: int):
    """Yield fixed-size lists from a streaming iterator."""
    batch = []
    for item in item_iter:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _empty_candidates(batch_size: int, K: int) -> list[list[dict]]:
    return [
        [{"text": "", "log_prob": 0.0, "num_tokens": 0} for _ in range(K)]
        for _ in range(batch_size)
    ]


def _drain_completed_futures(pending: list[Future[int]]) -> tuple[list[Future[int]], int]:
    """Collect completed scorer futures without blocking."""
    done = [f for f in pending if f.done()]
    if not done:
        return pending, 0

    written = 0
    for fut in done:
        written += fut.result()

    live = [f for f in pending if not f.done()]
    return live, written


def _init_vllm_backend(policy_cfg: dict, policy_path: str, logger):
    """Initialize vLLM engine.

    Returns
    -------
    dict
        {"llm", "sampling_params_cls", "lora_request", "model_ref"}
    """
    from vllm import LLM, SamplingParams

    lora_request = None
    policy_dir = Path(policy_path)
    is_lora_adapter = (policy_dir / "adapter_config.json").exists()

    llm_kwargs = {
        "model": str(policy_path),
        "trust_remote_code": True,
    }

    if is_lora_adapter:
        base_model = str(policy_cfg.get("model_name", "")).strip()
        if not base_model:
            raise ValueError(
                "policy.model_name must be set when policy-path is a LoRA adapter"
            )

        llm_kwargs["model"] = base_model
        llm_kwargs["enable_lora"] = True

        lora_rank = int(policy_cfg.get("lora", {}).get("r", 64))
        llm_kwargs["max_lora_rank"] = max(8, lora_rank)

        try:
            from vllm.lora.request import LoRARequest
        except Exception:
            from vllm import LoRARequest  # type: ignore

        lora_request = LoRARequest("policy_adapter", 1, str(policy_dir))
        logger.info(
            f"vLLM loading base model '{base_model}' with LoRA adapter '{policy_path}'"
        )
    else:
        logger.info(f"vLLM loading model from '{policy_path}'")

    # Gemma 2 uses tanh logit soft-capping in attention, which the default
    # FLASH_ATTN (FA3) backend does not support.  Force FLASHINFER via the
    # vLLM V1 attention_backend engine arg (the old VLLM_ATTENTION_BACKEND
    # env var is NOT read by the V1 engine).
    resolved_model = llm_kwargs["model"].lower()
    if "gemma-2" in resolved_model or "gemma2" in resolved_model:
        logger.info(
            "Gemma 2 detected — forcing FLASHINFER attention backend "
            "(required for tanh softcapping support)"
        )
        llm_kwargs["attention_backend"] = "FLASHINFER"

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    try:
        max_model_len = llm.llm_engine.model_config.max_model_len
    except AttributeError:
        max_model_len = getattr(llm, "max_model_len", 8192)

    return {
        "llm": llm,
        "sampling_params_cls": SamplingParams,
        "lora_request": lora_request,
        "model_ref": llm_kwargs["model"],
        "tokenizer": tokenizer,
        "max_model_len": max_model_len,
    }


def _generate_candidates_vllm(
    llm,
    sampling_params_cls,
    prompts: list[str],
    *,
    K: int,
    gen_max_tokens: int,
    gen_temperature: float,
    compute_logprobs: bool,
    logger,
    shard_id: int,
    lora_request=None,
    tokenizer=None,
    max_prompt_tokens: int = 0,
):
    """Generate candidates using vLLM.

    When *tokenizer* and *max_prompt_tokens* are provided, prompts that exceed
    the token limit are individually skipped (given empty candidates) instead
    of letting one over-limit prompt crash the entire batch.
    """
    # --- per-prompt length guard: skip over-limit prompts, keep the rest ---
    if tokenizer is not None and max_prompt_tokens > 0:
        valid_indices: list[int] = []
        skipped_indices: list[int] = []
        for i, p in enumerate(prompts):
            n_tok = len(tokenizer.encode(p, add_special_tokens=False))
            if n_tok <= max_prompt_tokens:
                valid_indices.append(i)
            else:
                skipped_indices.append(i)
                logger.debug(
                    f"[shard {shard_id}] Skipping prompt {i} "
                    f"({n_tok} tokens > {max_prompt_tokens} limit)"
                )
        if skipped_indices:
            logger.warning(
                f"[shard {shard_id}] Skipping {len(skipped_indices)}/{len(prompts)} "
                f"over-limit prompts in batch (keeping {len(valid_indices)})"
            )
        if not valid_indices:
            return _empty_candidates(len(prompts), K)

        valid_prompts = [prompts[i] for i in valid_indices]
    else:
        valid_indices = list(range(len(prompts)))
        skipped_indices = []
        valid_prompts = prompts

    sampling_params = sampling_params_cls(
        n=K,
        temperature=gen_temperature,
        top_p=0.95,
        max_tokens=gen_max_tokens,
        logprobs=1 if compute_logprobs else None,
    )

    kwargs = {}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    try:
        outputs = llm.generate(valid_prompts, sampling_params, **kwargs)
    except Exception as exc:
        logger.warning(
            f"[shard {shard_id}] vLLM generation failed for batch of "
            f"{len(valid_prompts)}: {exc}"
        )
        return _empty_candidates(len(prompts), K)

    valid_candidates: list[list[dict]] = []
    for req_out in outputs:
        req_outputs = sorted(req_out.outputs, key=lambda o: getattr(o, "index", 0))
        cur: list[dict] = []
        for out in req_outputs[:K]:
            text = (out.text or "").strip()
            token_ids = getattr(out, "token_ids", None) or []
            lp = 0.0
            if compute_logprobs:
                cum_lp = getattr(out, "cumulative_logprob", None)
                if cum_lp is not None:
                    lp = float(cum_lp)
            cur.append(
                {
                    "text": text,
                    "log_prob": lp,
                    "num_tokens": len(token_ids),
                }
            )
        while len(cur) < K:
            cur.append({"text": "", "log_prob": 0.0, "num_tokens": 0})
        valid_candidates.append(cur)

    if len(valid_candidates) < len(valid_prompts):
        valid_candidates.extend(
            _empty_candidates(len(valid_prompts) - len(valid_candidates), K)
        )

    # Reassemble full-batch results: valid candidates at their original
    # positions, empty candidates for skipped prompts.
    if not skipped_indices:
        return valid_candidates

    all_candidates: list[list[dict]] = _empty_candidates(len(prompts), K)
    for slot, cands in zip(valid_indices, valid_candidates):
        all_candidates[slot] = cands
    return all_candidates


def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    # --- Merge mode ---
    if args.merge:
        merge_shards(args.output, logger)
        return

    # --- Generation mode: require config, policy-path, trajectories ---
    if not args.config or not args.policy_path or not args.trajectories:
        logger.error("--config, --policy-path, and --trajectories are required "
                      "for generation mode (omit --merge)")
        sys.exit(1)

    cfg = load_config(args.config, args.overrides)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine shard output path
    if args.num_shards > 1:
        shard_output = (output_path.parent
                        / f"{output_path.stem}.shard_{args.shard_id:03d}{output_path.suffix}")
        logger.info(f"Shard {args.shard_id}/{args.num_shards} → {shard_output}")
    else:
        shard_output = output_path

    jsonl_log = JSONLLogger(
        output_path.parent / f"candidates_log_shard{args.shard_id}.jsonl"
    )

    # --------------- Resume: check what's already done ---------------
    if args.no_resume:
        done_eids: set[str] = set()
        file_mode = "w"
    else:
        done_eids = completed_episode_ids(shard_output)
        file_mode = "a"  # append to existing output
        if done_eids:
            logger.info(f"Resuming: {len(done_eids)} episodes already completed in {shard_output}")

    # --------------- Load policy generation backend ---------------
    logger.info(f"Preparing policy backend from {args.policy_path}")
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    try:
        vllm_backend = _init_vllm_backend(policy_cfg, args.policy_path, logger)
        logger.info(
            f"Using vLLM backend for policy inference (model={vllm_backend['model_ref']})"
        )
    except Exception as exc:
        logger.error(f"Failed to initialize vLLM backend: {exc}")
        sys.exit(1)

    # Load heuristic verifier (CPU rule-based; no model checkpoint required)
    logger.info("Loading heuristic verifier...")
    cfg_verifier = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    heuristic_cfg = cfg_verifier.get("heuristic", {}) if isinstance(cfg_verifier, dict) else {}

    benchmark = args.heuristic_benchmark
    if benchmark is None and isinstance(heuristic_cfg, dict):
        cfg_benchmark = heuristic_cfg.get("benchmark")
        if isinstance(cfg_benchmark, str) and cfg_benchmark.lower() in {
            "humaneval", "textworld", "terminalbench"
        }:
            benchmark = cfg_benchmark.lower()

    if benchmark is None:
        cfg_top_benchmark = str(cfg.get("benchmark", "")).lower()
        if cfg_top_benchmark == "alfworld":
            benchmark = "textworld"
        elif cfg_top_benchmark in {"humaneval", "textworld", "terminalbench"}:
            benchmark = cfg_top_benchmark

    run_code = not args.no_run_code
    if isinstance(heuristic_cfg, dict) and not args.no_run_code:
        run_code = bool(heuristic_cfg.get("run_code", run_code))

    verifier = HeuristicVerifier(run_code=run_code, benchmark=benchmark)
    logger.info(
        "Heuristic verifier ready "
        f"(run_code={run_code}, benchmark={benchmark or 'auto'})"
    )

    assert vllm_backend is not None

    # Keep a tiny CUDA workload active between generation bursts when enabled.
    keepalive_active = False
    if args.gpu_keepalive_interval > 0:
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                start_gpu_keepalive("cuda", args.gpu_keepalive_interval)
                keepalive_active = True
                logger.info(
                    "GPU keepalive enabled "
                    f"(interval={args.gpu_keepalive_interval:.2f}s)"
                )
        except Exception as exc:
            logger.warning(f"Could not start GPU keepalive: {exc}")

    # --------------- Load & shard trajectories (streamed) ---------------
    store = TrajectoryStore(args.trajectories)

    # Resolve generation hyperparams
    inf_cfg = cfg.get("inference", {})
    gen_temperature = float(inf_cfg.get("temperature", 0.7)) if inf_cfg else 0.7
    if args.max_new_tokens is not None:
        gen_max_tokens = args.max_new_tokens
    else:
        gen_max_tokens = int(inf_cfg.get("max_tokens", 256)) if inf_cfg else 256

    model_max_len = vllm_backend["max_model_len"]
    if args.max_prompt_tokens is not None:
        max_prompt_tokens = args.max_prompt_tokens
    else:
        max_prompt_tokens = model_max_len - gen_max_tokens
    logger.info(
        f"Prompt truncation: max_prompt_tokens={max_prompt_tokens} "
        f"(model_max_len={model_max_len}, gen_max_tokens={gen_max_tokens})"
    )

    # --------------- Stream step items (no giant pre-collect stall) ---------------
    batch_size = args.batch_size
    logger.info("Starting shard scan (streaming pass for step/episode counts)...")
    total_steps, total_episodes = count_sharded_steps_with_logging(
        store,
        args.shard_id,
        args.num_shards,
        done_eids,
        logger,
        log_every_episodes=args.scan_log_every_episodes,
    )
    total_batches = math.ceil(total_steps / batch_size) if total_steps else 0
    logger.info(
        f"Processing {total_steps} steps from {total_episodes} episodes "
        f"(skipped {len(done_eids)} done), "
        f"K={args.K}, batch_size={batch_size}, "
        f"max_new_tokens={gen_max_tokens}, temp={gen_temperature}, "
        f"compute_logprobs={args.compute_logprobs}"
    )

    if not args.no_consistency_pairs:
        logger.info("First pass: building context index for consistency pairs (streaming)...")
        t_idx = time.time()
        context_index = build_context_index(store)
        n_tasks = len(context_index)
        n_eps = sum(len(v) for v in context_index.values())
        logger.info(
            f"Context index built: {n_tasks} tasks, {n_eps} episodes "
            f"in {time.time() - t_idx:.1f}s"
        )
    else:
        context_index = None
        logger.info("Skipping context index (--no-consistency-pairs)")

    # --------------- Batched main loop ---------------
    total_pairs = 0
    t0 = time.time()

    writer = AsyncJSONLWriter(
        shard_output,
        mode=file_mode,
        maxsize=max(0, args.write_queue_size),
        flush_every=max(1, args.write_flush_every),
    ).start()

    pipeline_depth = max(1, args.pipeline_depth)
    if args.max_pending_batches > 0:
        max_pending_batches = args.max_pending_batches
    else:
        max_pending_batches = max(2, pipeline_depth * 8)

    logger.info(
        f"Scoring pipeline: workers={pipeline_depth}, max_pending={max_pending_batches}"
    )

    executor = ThreadPoolExecutor(max_workers=pipeline_depth)
    pending_futures: list[Future[int]] = []

    try:
        batch_stream = _iter_batches(
            iter_step_items(
                iter_sharded_episodes(store, args.shard_id, args.num_shards, done_eids),
                context_index=context_index,
            ),
            batch_size,
        )

        for batch_idx, batch in enumerate(batch_stream):
            batch_t0 = time.time()
            gen_t0 = time.time()
            B = len(batch)

            # ---------- 1. Batched candidate generation ---------------
            try:
                prompts = [f"{it['context']}\nAction:" for it in batch]
                all_candidates = _generate_candidates_vllm(
                    vllm_backend["llm"],
                    vllm_backend["sampling_params_cls"],
                    prompts,
                    K=args.K,
                    gen_max_tokens=gen_max_tokens,
                    gen_temperature=gen_temperature,
                    compute_logprobs=args.compute_logprobs,
                    logger=logger,
                    shard_id=args.shard_id,
                    lora_request=vllm_backend["lora_request"],
                    tokenizer=vllm_backend["tokenizer"],
                    max_prompt_tokens=max_prompt_tokens,
                )
                gen_elapsed = time.time() - gen_t0
            except Exception as exc:
                logger.warning(f"Generation failed for batch {batch_idx}: {exc}")
                all_candidates = _empty_candidates(B, args.K)
                gen_elapsed = time.time() - gen_t0

            # ---------- 2. Pipeline verifier scoring + write ----------
            score_submit_t0 = time.time()
            pending_futures.append(executor.submit(
                _score_and_write_batch,
                verifier,
                writer,
                batch,
                all_candidates,
                args.store_all_candidates,
                args.verifier_batch_size,
                logger,
            ))
            submit_elapsed = time.time() - score_submit_t0

            # Opportunistically collect finished scoring tasks without blocking.
            pending_futures, gained = _drain_completed_futures(pending_futures)
            total_pairs += gained

            # Only block when queued work reaches a hard cap.
            wait_elapsed = 0.0
            while len(pending_futures) >= max_pending_batches:
                wait_t0 = time.time()
                done, not_done = wait(pending_futures, return_when=FIRST_COMPLETED)
                pending_futures = list(not_done)
                for fut in done:
                    total_pairs += fut.result()
                wait_elapsed += time.time() - wait_t0

            # ---------- 4. Progress logging ---------------------------
            batch_elapsed = time.time() - batch_t0
            elapsed = time.time() - t0
            steps_done = min((batch_idx + 1) * batch_size, total_steps)
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining = total_steps - steps_done
            eta = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * steps_done / total_steps if total_steps else 100.0

            logger.info(
                f"[shard {args.shard_id}] "
                f"Batch {batch_idx + 1}/{total_batches} "
                f"({pct:5.1f}%)  |  "
                f"{total_pairs} pairs  |  "
                f"{rate:.1f} steps/s  |  "
                f"gen {gen_elapsed:.2f}s  |  "
                f"submit {submit_elapsed:.3f}s  |  "
                f"wait {wait_elapsed:.2f}s  |  "
                f"batch {batch_elapsed:.2f}s  |  "
                f"pending={len(pending_futures)}  |  "
                f"write_q={writer.pending}  |  "
                f"ETA {eta / 60:.1f}min"
            )

        for fut in pending_futures:
            total_pairs += fut.result()
    finally:
        if keepalive_active:
            stop_gpu_keepalive()
        executor.shutdown(wait=False)
        # Ensure pending writes are fully persisted before process exit.
        writer.close()

    # --------------- Cleanup ---------------
    elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_pairs} pairs from "
        f"{total_episodes} episodes ({total_steps} steps) in {elapsed / 60:.1f}min "
        f"({total_steps / max(elapsed, 1e-6):.1f} steps/s) → {shard_output}"
    )
    jsonl_log.log("summary", {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_pairs": total_pairs,
        "num_episodes": total_episodes,
        "K": args.K,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
    })


if __name__ == "__main__":
    main()
