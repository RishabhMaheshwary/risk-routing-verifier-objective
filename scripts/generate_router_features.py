#!/usr/bin/env python3
"""
Generate router training features from collected trajectories.

Uses vLLM for fast batched inference.  For each decision point, generates
K candidate actions and computes a 24-dim feature vector.

Single-GPU usage:
    python scripts/generate_router_features.py \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/router_features/gaia.jsonl

Multi-GPU usage (via launch script — shards episodes across GPUs):
    bash scripts/launch_router_features.sh 4 \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/router_features/gaia.jsonl

Merge shards after all GPUs finish:
    python scripts/generate_router_features.py --merge \
        --output data/router_features/gaia.jsonl

Resume after a crash (re-run the same command — skips completed steps):
    # Same command; already-written episode_ids are detected automatically.

Features extracted per decision point (15-dim):
- SLM entropy (H(π_θ))  [approximated from vLLM top-k logprobs]
- Verifier score spread, mean, std, best, worst (over K candidates)
- Action log-probability best, mean, std (over K candidates)
- Candidate consistency (fraction sharing the same leading token)
- Semantic entropy over K candidates (entropy over first-token cluster distribution)
- Step count / horizon fraction
- Context length
- Goal length (task complexity proxy)
- Benchmark one-hot and perturbation one-hot are currently commented out
    (former indices 15-23 in the 24-dim schema)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import ActionSource, PerturbationType, TrajectoryStore
from r2v.models.verifier import create_verifier
from r2v.utils.config import load_config
from r2v.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate router features")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--trajectories", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of prompts per vLLM generate() call.  vLLM handles "
             "internal continuous-batching; larger values amortise overhead.",
    )
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers / GPUs")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard outputs instead of generating")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume; overwrite existing output")
    parser.add_argument(
        "--num-logprobs", type=int, default=20,
        help="Top-k logprobs per generated token (used for entropy approx)",
    )
    parser.add_argument(
        "--gpu-keepalive-interval",
        type=float,
        default=2.0,
        help="Seconds between GPU matmul kernels during CPU-heavy verifier "
             "phases (0 disables). Helps HPC schedulers that kill low-util jobs.",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument(
        "--two-phase", action="store_true", default=True,
        help="Two-phase mode: generate all candidates first (GPU-heavy), "
             "then score them (CPU-heavy). Maximizes GPU utilization "
             "and prevents HPC schedulers from killing low-util jobs.",
    )
    parser.add_argument(
        "--no-two-phase", action="store_false", dest="two_phase",
        help="Disable two-phase mode; use pipelined GPU+CPU overlap.",
    )
    parser.add_argument(
        "--generate-only", action="store_true", default=False,
        help="Only run candidate generation (Phase 1). Keeps GPU at 100%%. "
             "Skips verifier scoring entirely. Run --score-only later.",
    )
    parser.add_argument(
        "--score-only", action="store_true", default=False,
        help="Only run verifier scoring (Phase 2) from an existing candidate "
             "cache. Does not load vLLM or use the GPU.",
    )
    return parser.parse_args()


# ================================================================
# GPU keepalive (CPU-bound heuristic verifier leaves GPU idle)
# ================================================================

_keepalive_stop = threading.Event()


def _gpu_keepalive(device: str, interval: float) -> None:
    import torch

    # Large enough to actually register on nvidia-smi's utilization sampling
    N = 2048
    a = torch.randn((N, N), device=device, dtype=torch.float16)
    b = torch.randn((N, N), device=device, dtype=torch.float16)
    while not _keepalive_stop.is_set():
        try:
            for _ in range(4):
                _ = a @ b
            torch.cuda.synchronize()
        except Exception:
            pass
        _keepalive_stop.wait(interval)


def _start_gpu_keepalive(device: str, interval: float) -> None:
    _keepalive_stop.clear()
    t = threading.Thread(
        target=_gpu_keepalive, args=(device, interval), daemon=True,
    )
    t.start()


def _stop_gpu_keepalive() -> None:
    _keepalive_stop.set()


# ================================================================
# Resume helpers
# ================================================================

def _completed_episode_ids(output_file: Path) -> set[str]:
    """Return set of episode_ids already present in *output_file*."""
    if not output_file.exists():
        return set()
    eids: set[str] = set()
    with open(output_file) as f:
        for line in f:
            try:
                eids.add(json.loads(line)["episode_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return eids


# ================================================================
# Merge shards
# ================================================================

def _merge_shards(output_path: str, logger):
    """Combine all shard JSONL files into a single output."""
    out = Path(output_path)
    pattern = str(out.parent / f"{out.stem}.shard_*{out.suffix}")
    # Exclude *.shard_NNN.gen_cache.jsonl (two-phase caches); only merge real shard outputs.
    shard_files = sorted(
        sf for sf in glob.glob(pattern) if not sf.endswith(".gen_cache.jsonl")
    )
    if not shard_files:
        logger.error(f"No shard files matching {pattern}")
        sys.exit(1)

    total = 0
    with open(out, "w") as fout:
        for sf in shard_files:
            with open(sf) as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
            logger.info(f"  Merged {sf}")
    logger.info(f"Merged {len(shard_files)} shards -> {out} ({total} records)")


# ================================================================
# Feature encoding helpers
# ================================================================

_PERT_TYPES = [
    None, "tool_flakiness", "partial_observability",
    "prompt_injection", "distractors",
]

_BENCHMARKS = ["gaia", "alfworld", "humaneval", "webarena"]


def _detect_benchmark(config_path: str | None) -> str | None:
    if not config_path:
        return None
    for bench in _BENCHMARKS:
        if bench in config_path.lower():
            return bench
    return None


def _benchmark_onehot(benchmark: str | None) -> list[float]:
    one_hot = [0.0] * len(_BENCHMARKS)
    if benchmark in _BENCHMARKS:
        one_hot[_BENCHMARKS.index(benchmark)] = 1.0
    return one_hot


def _candidate_consistency(texts: list[str]) -> float:
    """Fraction of K candidates sharing the most common leading token."""
    if not texts:
        return 1.0
    first_tokens = [t.split()[0].lower() if t.strip() else "" for t in texts]
    most_common_count = Counter(first_tokens).most_common(1)[0][1]
    return most_common_count / len(first_tokens)


def _semantic_entropy(texts: list[str]) -> float:
    """Shannon entropy over first-token cluster distribution of K candidates."""
    if not texts:
        return 0.0
    first_tokens = [t.split()[0].lower() if t.strip() else "" for t in texts]
    counts = Counter(first_tokens)
    total = len(first_tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    return float(entropy)


def _perturbation_onehot(pert_type: str | None) -> list[float]:
    one_hot = [0.0] * len(_PERT_TYPES)
    if pert_type in _PERT_TYPES:
        one_hot[_PERT_TYPES.index(pert_type)] = 1.0
    else:
        one_hot[0] = 1.0
    return one_hot


def _assemble_features(
    entropy: float,
    scores: list[float],
    log_probs: list[float],
    step_idx: int,
    max_steps: int,
    context_len: int,
    pert_type: str | None,
    candidate_texts: list[str],
    benchmark: str | None,
    goal: str,
) -> list[float]:
    """Build the 15-dim feature vector from pre-computed components.

    Dims:
      0        entropy
      1-5      verifier scores: spread, mean, std, best, worst
      6-8      log-prob stats: best, mean, std
      9        candidate consistency (fraction with same leading token)
      10       semantic entropy (entropy over first-token cluster distribution)
      11       horizon fraction (step_idx / max_steps)
      12       step number (absolute)
      13       normalized context length
      14       goal length (task complexity proxy)
      15-23    reserved (benchmark one-hot + perturbation one-hot commented out)
    """
    features: list[float] = []
    features.append(entropy)
    features.append(max(scores) - min(scores))    # spread
    features.append(float(np.mean(scores)))        # mean
    features.append(float(np.std(scores)))         # std
    features.append(max(scores))                   # best
    features.append(min(scores))                   # worst
    features.append(max(log_probs))                # log_prob_best
    features.append(float(np.mean(log_probs)))     # log_prob_mean
    features.append(float(np.std(log_probs)))      # log_prob_std
    features.append(_candidate_consistency(candidate_texts))
    features.append(_semantic_entropy(candidate_texts))
    features.append(step_idx / max(max_steps, 1))  # horizon fraction
    features.append(float(step_idx))               # absolute step
    features.append(context_len / 10000.0)         # normalized context length
    features.append(len(goal) / 1000.0)            # goal length
    # features.extend(_benchmark_onehot(benchmark))
    # features.extend(_perturbation_onehot(pert_type))
    return features  # length 15


# ================================================================
# vLLM backend
# ================================================================

def _init_vllm_backend(policy_cfg: dict, policy_path: str, logger):
    """Initialize vLLM engine with optional LoRA adapter.

    Returns dict with keys: llm, sampling_params_cls, lora_request, vocab_size.
    """
    from vllm import LLM, SamplingParams

    lora_request = None
    policy_dir = Path(policy_path)
    is_lora_adapter = (policy_dir / "adapter_config.json").exists()

    llm_kwargs: dict = {
        "model": str(policy_path),
        "trust_remote_code": True,
        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.75")),
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
            f"vLLM: base='{base_model}' + LoRA adapter='{policy_path}'"
        )
    else:
        logger.info(f"vLLM: model='{policy_path}'")

    # Gemma 2 uses tanh logit soft-capping in attention, which the default
    # FLASH_ATTN (FA3) backend does not support.  Force FLASHINFER via the
    # vLLM V1 attention_backend engine arg.
    resolved_model = llm_kwargs["model"].lower()
    if "gemma-2" in resolved_model or "gemma2" in resolved_model:
        logger.info(
            "Gemma 2 detected — forcing FLASHINFER attention backend "
            "(required for tanh softcapping support)"
        )
        llm_kwargs["attention_backend"] = "FLASHINFER"

    llm = LLM(**llm_kwargs)

    vocab_size = 152064  # Qwen2.5 default
    try:
        tokenizer = llm.get_tokenizer()
        vocab_size = len(tokenizer)
    except Exception:
        pass

    return {
        "llm": llm,
        "sampling_params_cls": SamplingParams,
        "lora_request": lora_request,
        "vocab_size": vocab_size,
    }


# ================================================================
# Entropy approximation from vLLM top-k logprobs
# ================================================================

def _approx_entropy(logprobs_dict, vocab_size: int = 152064) -> float:
    """Approximate next-token entropy from the top-k logprobs vLLM returns.

    For tokens outside the top-k, remaining probability mass is assumed
    uniform — this preserves relative ordering for the router while
    avoiding a full forward pass over the vocabulary.
    """
    if not logprobs_dict:
        return 0.0

    probs = []
    for logprob_obj in logprobs_dict.values():
        lp = logprob_obj.logprob if hasattr(logprob_obj, "logprob") else float(logprob_obj)
        probs.append(math.exp(lp))

    probs_arr = np.array(probs, dtype=np.float64)
    covered = float(probs_arr.sum())
    remaining = max(0.0, 1.0 - covered)

    entropy = float(-np.sum(probs_arr * np.log(probs_arr + 1e-30)))

    n_other = vocab_size - len(probs)
    if remaining > 1e-8 and n_other > 0:
        p_other = remaining / n_other
        entropy += -remaining * math.log(p_other + 1e-30)

    return entropy


# ================================================================
# vLLM generation — candidates + entropy in one call
# ================================================================

def _generate_batch_vllm(
    backend: dict,
    prompts: list[str],
    K: int,
    gen_max_tokens: int,
    gen_temperature: float,
    num_logprobs: int = 20,
) -> tuple[list[list[dict]], list[float]]:
    """Generate K candidates per prompt using vLLM.

    Returns
    -------
    all_candidates : list[list[dict]]
        ``all_candidates[i]`` has K dicts with keys text, log_prob, num_tokens.
    entropies : list[float]
        Approximate next-token entropy per prompt (from first-token logprobs).
    """
    llm = backend["llm"]
    SamplingParams = backend["sampling_params_cls"]
    lora_request = backend["lora_request"]
    vocab_size = backend["vocab_size"]

    sampling_params = SamplingParams(
        n=K,
        temperature=gen_temperature,
        top_p=0.95,
        max_tokens=gen_max_tokens,
        logprobs=num_logprobs,
    )

    kwargs: dict = {}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    outputs = llm.generate(prompts, sampling_params, **kwargs)

    all_candidates: list[list[dict]] = []
    entropies: list[float] = []

    for req_out in outputs:
        req_outputs = sorted(
            req_out.outputs, key=lambda o: getattr(o, "index", 0)
        )

        candidates: list[dict] = []
        for out in req_outputs[:K]:
            text = (out.text or "").strip()
            token_ids = getattr(out, "token_ids", None) or []
            lp = float(getattr(out, "cumulative_logprob", 0.0) or 0.0)
            candidates.append({
                "text": text,
                "log_prob": lp,
                "num_tokens": len(token_ids),
            })
        while len(candidates) < K:
            candidates.append({"text": "", "log_prob": 0.0, "num_tokens": 0})
        all_candidates.append(candidates)

        entropy = 0.0
        if (req_outputs
                and req_outputs[0].logprobs
                and len(req_outputs[0].logprobs) > 0):
            entropy = _approx_entropy(req_outputs[0].logprobs[0], vocab_size)
        entropies.append(entropy)

    while len(all_candidates) < len(prompts):
        all_candidates.append(
            [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * K
        )
        entropies.append(0.0)

    return all_candidates, entropies


# ================================================================
# Diagnostics
# ================================================================

_DIAG_INTERVAL = 10


def _make_diag_state() -> dict:
    return {
        "entropy_errors": 0,
        "gen_errors": 0,
        "verifier_errors": 0,
        "all_entropies": [],
        "all_v_means": [],
        "all_v_spreads": [],
        "all_lp_means": [],
        "all_consistencies": [],
        "all_successes": [],
    }


def _log_running_report(logger, diag: dict, steps_done: int, total: int):
    ent = np.array(diag["all_entropies"])
    vm  = np.array(diag["all_v_means"])
    vs  = np.array(diag["all_v_spreads"])
    lp  = np.array(diag["all_lp_means"])
    con = np.array(diag["all_consistencies"])
    suc = np.array(diag["all_successes"])

    lines = [
        f"  ── Running report ({steps_done}/{total} steps) ──",
        f"  Entropy      : mean={ent.mean():.4f}  std={ent.std():.4f}  "
        f"[{ent.min():.3f}, {np.median(ent):.3f}, {ent.max():.3f}]",
        f"  V-score mean : mean={vm.mean():.4f}  std={vm.std():.4f}  "
        f"[{vm.min():.3f}, {np.median(vm):.3f}, {vm.max():.3f}]",
        f"  V-score spread: mean={vs.mean():.4f}  std={vs.std():.4f}",
        f"  LogProb mean : mean={lp.mean():.4f}  std={lp.std():.4f}",
        f"  Consistency  : mean={con.mean():.4f}  std={con.std():.4f}",
    ]

    succ_mask = suc == 1.0
    fail_mask = suc == 0.0
    if succ_mask.sum() > 0 and fail_mask.sum() > 0:
        ent_s, ent_f = ent[succ_mask].mean(), ent[fail_mask].mean()
        vm_s, vm_f   = vm[succ_mask].mean(),  vm[fail_mask].mean()
        lines.append(
            f"  Theory check : ent(fail)={ent_f:.4f} vs ent(succ)={ent_s:.4f}  "
            f"{'OK' if ent_f > ent_s else 'UNEXPECTED'}  |  "
            f"v_mean(succ)={vm_s:.4f} vs v_mean(fail)={vm_f:.4f}  "
            f"{'OK' if vm_s > vm_f else 'UNEXPECTED'}"
        )

    errs = diag["entropy_errors"] + diag["gen_errors"] + diag["verifier_errors"]
    if errs:
        lines.append(
            f"  Silent errors: entropy={diag['entropy_errors']}  "
            f"gen={diag['gen_errors']}  verifier={diag['verifier_errors']}"
        )

    logger.info("\n".join(lines))


def _theory_summary(diag: dict) -> str:
    """One-line running theory check for the per-batch log.

    Returns e.g. 'theory[3/5]: ent(F>S) OK | v(S>F) OK | spread(F>S) OK | cons(S>F) FAIL | se(F>S) OK'
    or empty string if not enough data yet.
    """
    suc = np.array(diag["all_successes"])
    if suc.sum() == 0 or (1 - suc).sum() == 0:
        return ""

    succ_mask = suc == 1.0
    fail_mask = suc == 0.0

    checks: list[str] = []
    passed = 0
    total = 0

    ent = np.array(diag["all_entropies"])
    ent_ok = ent[fail_mask].mean() > ent[succ_mask].mean()
    checks.append(f"ent(F>S) {'OK' if ent_ok else 'FAIL'}")
    passed += ent_ok
    total += 1

    vm = np.array(diag["all_v_means"])
    vm_ok = vm[succ_mask].mean() > vm[fail_mask].mean()
    checks.append(f"v(S>F) {'OK' if vm_ok else 'FAIL'}")
    passed += vm_ok
    total += 1

    vs = np.array(diag["all_v_spreads"])
    vs_ok = vs[fail_mask].mean() > vs[succ_mask].mean()
    checks.append(f"spread(F>S) {'OK' if vs_ok else 'FAIL'}")
    passed += vs_ok
    total += 1

    con = np.array(diag["all_consistencies"])
    con_ok = con[succ_mask].mean() > con[fail_mask].mean()
    checks.append(f"cons(S>F) {'OK' if con_ok else 'FAIL'}")
    passed += con_ok
    total += 1

    lp = np.array(diag["all_lp_means"])
    lp_ok = lp[succ_mask].mean() > lp[fail_mask].mean()
    checks.append(f"lp(S>F) {'OK' if lp_ok else 'FAIL'}")
    passed += lp_ok
    total += 1

    return f"theory[{passed}/{total}]: {' | '.join(checks)}"


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    # ---- Merge mode --------------------------------------------------
    if args.merge:
        _merge_shards(args.output, logger)
        return

    # ---- Generation mode: require config, policy-path, trajectories --
    if not args.config or not args.policy_path or not args.trajectories:
        logger.error(
            "--config, --policy-path, and --trajectories are required "
            "for generation mode (omit --merge)"
        )
        sys.exit(1)

    cfg = load_config(args.config, args.overrides)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine shard output path
    if args.num_shards > 1:
        shard_output = (
            output_path.parent
            / f"{output_path.stem}.shard_{args.shard_id:03d}{output_path.suffix}"
        )
        logger.info(
            f"[shard {args.shard_id}/{args.num_shards}] "
            f"Writing to {shard_output}"
        )
    else:
        shard_output = output_path

    # ---- Resume: detect already-completed episodes -------------------
    if args.no_resume:
        done_eids: set[str] = set()
        file_mode = "w"
    else:
        done_eids = _completed_episode_ids(shard_output)
        file_mode = "a"
        if done_eids:
            logger.info(
                f"Resuming: {len(done_eids)} episodes already in {shard_output}"
            )

    # ---- Load vLLM backend -------------------------------------------
    backend = None
    if not args.score_only:
        logger.info("Initializing vLLM backend...")
        policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
        try:
            backend = _init_vllm_backend(policy_cfg, args.policy_path, logger)
        except Exception as exc:
            logger.error(f"Failed to initialize vLLM backend: {exc}")
            sys.exit(1)
        logger.info(f"vLLM ready (vocab_size={backend['vocab_size']})")
    else:
        logger.info("--score-only: skipping vLLM backend (no GPU needed)")

    keepalive_active = False
    if args.gpu_keepalive_interval > 0 and not args.two_phase:
        # In two-phase mode, the batch runner manages keepalive itself
        try:
            import torch

            if torch.cuda.is_available():
                _start_gpu_keepalive("cuda", args.gpu_keepalive_interval)
                keepalive_active = True
                logger.info(
                    "GPU keepalive enabled "
                    f"(interval={args.gpu_keepalive_interval:.1f}s)"
                )
        except Exception as exc:
            logger.warning(f"Could not start GPU keepalive: {exc}")

    # ---- Load verifier -----------------------------------------------
    verifier = None
    if not args.generate_only:
        logger.info("Loading verifier...")
        vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
        verifier = create_verifier(vcfg)
        if hasattr(verifier, "eval"):
            verifier.eval()
    else:
        logger.info("--generate-only: skipping verifier (GPU generation only)")

    # ---- Load & shard trajectories -----------------------------------
    store = TrajectoryStore(args.trajectories)
    all_episodes = store.load_episodes()
    episodes = all_episodes[args.shard_id :: args.num_shards]

    if done_eids and not args.two_phase:
        episodes = [ep for ep in episodes if ep.episode_id not in done_eids]
    elif done_eids and args.two_phase:
        logger.info(
            f"Two-phase mode: keeping all {len(episodes)} episodes "
            f"(Phase 2 resumes via output record count)"
        )

    logger.info(
        f"[shard {args.shard_id}] {len(episodes)} episodes to process "
        f"(total {len(all_episodes)}, skipped {len(done_eids)} done)"
    )

    benchmark = _detect_benchmark(args.config)
    logger.info(f"Benchmark detected from config path: {benchmark!r}")

    max_steps = (
        cfg.get("inference", {}).get("step_limit", 15)
        if cfg.get("inference") else 15
    )
    inf_cfg = (
        OmegaConf.to_container(cfg.get("inference", {}), resolve=True)
        if cfg.get("inference") else {}
    )
    gen_temperature = float(inf_cfg.get("temperature", 0.7)) if inf_cfg else 0.7
    gen_max_tokens  = int(inf_cfg.get("max_tokens", 512))   if inf_cfg else 512
    K = args.K

    # ---- Pre-collect every (context, metadata) pair ------------------
    logger.info("Pre-collecting step items...")
    all_items: list[dict] = []
    for episode in episodes:
        goal = episode.metadata.goal if episode.metadata else ""
        slm_success = 1.0 if episode.success else 0.0
        llm_steps = sum(
            1 for s in episode.steps
            if s.action_source == ActionSource.TEACHER
        )
        cost = llm_steps / max(len(episode.steps), 1)

        context = ""
        for step_idx, step in enumerate(episode.steps):
            context += step.observation.raw_text + "\n"

            pert_type = None
            if (step.perturbation_type
                    and step.perturbation_type != PerturbationType.NONE):
                pert_type = step.perturbation_type.value

            all_items.append({
                "context": context,
                "step_idx": step_idx,
                "goal": goal,
                "pert_type": pert_type,
                "slm_success": slm_success,
                "cost": cost,
                "episode_id": episode.episode_id,
                "benchmark": benchmark,
            })

            context += step.action.raw_text + "\n"

    total_batches = (len(all_items) + args.batch_size - 1) // args.batch_size
    logger.info(
        f"[shard {args.shard_id}] Collected {len(all_items)} step items "
        f"from {len(episodes)} episodes -> {total_batches} batches "
        f"(batch_size={args.batch_size})"
    )

    # ---- Batched processing with vLLM --------------------------------
    t0 = time.time()

    try:
        if args.two_phase:
            _run_batches_two_phase(
                args, shard_output, file_mode, all_items, total_batches,
                backend, verifier, K, gen_max_tokens, gen_temperature,
                max_steps, t0, logger, len(episodes),
                keepalive_interval=args.gpu_keepalive_interval,
            )
        else:
            _run_batches_pipelined(
                args, shard_output, file_mode, all_items, total_batches,
                backend, verifier, K, gen_max_tokens, gen_temperature,
                max_steps, t0, logger, len(episodes),
            )
    finally:
        if keepalive_active:
            _stop_gpu_keepalive()


def _score_and_assemble(
    verifier,
    batch: list[dict],
    all_candidates: list[list[dict]],
    entropies: list[float],
    K: int,
    max_steps: int,
    logger,
    _diag: dict,
    batch_idx: int,
) -> tuple[list[dict], list[float], list[float], list[float], bool]:
    """Run verifier scoring + feature assembly on a background thread.

    Returns (records, batch_ent, batch_v, batch_v_spread, verifier_fallback).
    """
    B = len(batch)
    flat_ctx, flat_cand, flat_goal = [], [], []
    for b in range(B):
        for c in all_candidates[b]:
            flat_ctx.append(batch[b]["context"])
            flat_cand.append(c["text"])
            flat_goal.append(batch[b]["goal"])

    verifier_fallback = False
    try:
        flat_scores = verifier.score_batch(flat_ctx, flat_cand, flat_goal)
    except Exception as exc:
        flat_scores = [0.5] * len(flat_ctx)
        verifier_fallback = True
        _diag["verifier_errors"] += 1
        if _diag["verifier_errors"] <= 3:
            logger.warning(f"Verifier failed (batch {batch_idx}): {exc}")

    all_scores = [flat_scores[b * K : (b + 1) * K] for b in range(B)]

    records: list[dict] = []
    for i, item in enumerate(batch):
        features = _assemble_features(
            entropy=entropies[i],
            scores=all_scores[i],
            log_probs=[c["log_prob"] for c in all_candidates[i]],
            step_idx=item["step_idx"],
            max_steps=max_steps,
            context_len=len(item["context"]),
            pert_type=item["pert_type"],
            candidate_texts=[c["text"] for c in all_candidates[i]],
            benchmark=item["benchmark"],
            goal=item["goal"],
        )
        records.append({
            "features": features,
            "slm_success": item["slm_success"],
            "cost": item["cost"],
            "episode_id": item["episode_id"],
            "step_idx": item["step_idx"],
            "_entropy": entropies[i],
            "_v_mean": float(np.mean(all_scores[i])),
            "_v_spread": max(all_scores[i]) - min(all_scores[i]),
            "_lp_mean": float(np.mean([c["log_prob"] for c in all_candidates[i]])),
            "_consistency": features[9],
        })

    batch_v = [float(np.mean(s)) for s in all_scores]
    batch_v_spread = [max(s) - min(s) for s in all_scores]
    return records, entropies, batch_v, batch_v_spread, verifier_fallback


def _run_batches_pipelined(
    args,
    shard_output: Path,
    file_mode: str,
    all_items: list[dict],
    total_batches: int,
    backend: dict,
    verifier,
    K: int,
    gen_max_tokens: int,
    gen_temperature: float,
    max_steps: int,
    t0: float,
    logger,
    num_episodes: int,
) -> None:
    """Pipelined main loop: generates batch N+1 on GPU while scoring batch N on CPU."""
    total_features = 0
    _diag = _make_diag_state()

    scorer = ThreadPoolExecutor(max_workers=2)
    pending: list[tuple[int, Future, bool, bool]] = []

    def _drain_and_write(f, block: bool = False):
        """Collect finished scoring futures and write results."""
        nonlocal total_features
        still_pending = []
        for bid, fut, gen_fb, ent_fb in pending:
            if block or fut.done():
                records, batch_ent, batch_v, batch_v_spread, ver_fb = fut.result()
                for rec in records:
                    _diag["all_entropies"].append(rec.pop("_entropy"))
                    _diag["all_v_means"].append(rec.pop("_v_mean"))
                    _diag["all_v_spreads"].append(rec.pop("_v_spread"))
                    _diag["all_lp_means"].append(rec.pop("_lp_mean"))
                    _diag["all_consistencies"].append(rec.pop("_consistency"))
                    _diag["all_successes"].append(rec["slm_success"])
                    f.write(json.dumps(rec) + "\n")
                    total_features += 1

                elapsed = time.time() - t0
                steps_done = total_features
                rate = steps_done / elapsed if elapsed > 0 else 0
                remaining = len(all_items) - steps_done
                eta = remaining / rate if rate > 0 else float("inf")
                pct = 100.0 * steps_done / len(all_items) if all_items else 100.0

                fb_flags = []
                if ent_fb:
                    fb_flags.append("ENT")
                if gen_fb:
                    fb_flags.append("GEN")
                if ver_fb:
                    fb_flags.append("VER")
                fb_str = f"  FALLBACK:[{','.join(fb_flags)}]" if fb_flags else ""

                theory_str = ""
                if steps_done >= 64:
                    ts = _theory_summary(_diag)
                    if ts:
                        theory_str = f"  |  {ts}"

                logger.info(
                    f"[shard {args.shard_id}] "
                    f"Batch {bid + 1}/{total_batches} "
                    f"({pct:5.1f}%)  |  "
                    f"{steps_done}/{len(all_items)} steps  |  "
                    f"{rate:.1f} steps/s  |  "
                    f"ETA {eta / 60:.1f}min  |  "
                    f"ent={np.mean(batch_ent):.3f}±{np.std(batch_ent):.3f}  "
                    f"v_mean={np.mean(batch_v):.3f}  "
                    f"v_spread={np.mean(batch_v_spread):.3f}"
                    f"{fb_str}{theory_str}"
                )

                if (bid + 1) % _DIAG_INTERVAL == 0 and _diag["all_entropies"]:
                    _log_running_report(logger, _diag, steps_done, len(all_items))

                f.flush()
            else:
                still_pending.append((bid, fut, gen_fb, ent_fb))
        pending[:] = still_pending

    try:
        with open(shard_output, file_mode) as f:
            for batch_idx in range(total_batches):
                start = batch_idx * args.batch_size
                end = min(start + args.batch_size, len(all_items))
                batch = all_items[start:end]
                B = len(batch)

                prompts = [f"{it['context']}\nAction:" for it in batch]

                gen_fallback = False
                entropy_fallback = False
                try:
                    all_candidates, entropies = _generate_batch_vllm(
                        backend, prompts, K=K,
                        gen_max_tokens=gen_max_tokens,
                        gen_temperature=gen_temperature,
                        num_logprobs=args.num_logprobs,
                    )
                except Exception as exc:
                    all_candidates = [
                        [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * K
                    ] * B
                    entropies = [0.0] * B
                    gen_fallback = True
                    entropy_fallback = True
                    _diag["gen_errors"] += 1
                    _diag["entropy_errors"] += 1
                    if _diag["gen_errors"] <= 3:
                        logger.warning(
                            f"vLLM generation failed (batch {batch_idx}): {exc}"
                        )

                fut = scorer.submit(
                    _score_and_assemble,
                    verifier, batch, all_candidates, entropies, K,
                    max_steps, logger, _diag, batch_idx,
                )
                pending.append((batch_idx, fut, gen_fallback, entropy_fallback))

                _drain_and_write(f, block=False)

                if len(pending) >= 3:
                    _drain_and_write(f, block=True)

            _drain_and_write(f, block=True)
    finally:
        scorer.shutdown(wait=True)

    elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_features} feature vectors "
        f"from {num_episodes} episodes in {elapsed / 60:.1f}min "
        f"({total_features / max(elapsed, 1e-6):.1f} steps/s) "
        f"-> {shard_output}"
    )

    if _diag["all_entropies"]:
        _log_running_report(logger, _diag, total_features, len(all_items))
        total_errs = (
            _diag["entropy_errors"] + _diag["gen_errors"]
            + _diag["verifier_errors"]
        )
        if total_errs:
            logger.warning(
                f"Total silent fallbacks: entropy={_diag['entropy_errors']}  "
                f"gen={_diag['gen_errors']}  verifier={_diag['verifier_errors']}  "
                f"({100 * total_errs / max(total_batches, 1):.1f}% of batches)"
            )


def _run_batches_two_phase(
    args,
    shard_output: Path,
    file_mode: str,
    all_items: list[dict],
    total_batches: int,
    backend: dict,
    verifier,
    K: int,
    gen_max_tokens: int,
    gen_temperature: float,
    max_steps: int,
    t0: float,
    logger,
    num_episodes: int,
    keepalive_interval: float,
) -> None:
    """Two-phase processing: generate all candidates (GPU), then score (CPU).

    Phase 1 keeps the GPU at ~100% by running all vLLM generation back-to-back
    without interleaving CPU-bound verifier scoring.  Candidates are streamed
    to a cache JSONL file for crash-resume support.

    Phase 2 reads the cache and scores every candidate with the verifier
    (CPU-bound).  A GPU keepalive prevents HPC schedulers from killing the job.
    """
    _diag = _make_diag_state()
    cache_path = shard_output.with_suffix(".gen_cache.jsonl")

    if file_mode == "w" and not args.score_only:
        cache_path.unlink(missing_ok=True)

    if args.score_only and not cache_path.exists():
        logger.error(
            f"--score-only but candidate cache not found: {cache_path}\n"
            f"Run --generate-only first to create it."
        )
        sys.exit(1)

    # ── Phase 1: Generate all candidates (GPU at ~100%) ───────────
    _stop_gpu_keepalive()

    cached_batches = 0
    cache_valid = False
    if cache_path.exists():
        with open(cache_path) as f:
            header_line = f.readline()
            try:
                header = json.loads(header_line)
                if (
                    header.get("batch_size") == args.batch_size
                    and header.get("total_items") == len(all_items)
                    and header.get("K") == K
                ):
                    cache_valid = True
                    cached_batches = sum(1 for _ in f)
            except (json.JSONDecodeError, KeyError):
                pass

        if not cache_valid:
            logger.warning(
                "Candidate cache parameters changed; regenerating"
            )
            cache_path.unlink()

    if cached_batches >= total_batches:
        logger.info(
            f"[Phase 1] Candidate cache already complete "
            f"({cached_batches}/{total_batches} batches)"
        )
    else:
        if cached_batches > 0:
            logger.info(
                f"[Phase 1] Resuming generation from batch "
                f"{cached_batches}/{total_batches}"
            )
        else:
            logger.info(
                f"[Phase 1] Generating candidates for {len(all_items)} steps "
                f"in {total_batches} batches (GPU at ~100%)..."
            )

        t_gen = time.time()
        mode = "a" if cached_batches > 0 else "w"

        with open(cache_path, mode) as cf:
            if mode == "w":
                cf.write(
                    json.dumps(
                        {
                            "batch_size": args.batch_size,
                            "total_items": len(all_items),
                            "K": K,
                        }
                    )
                    + "\n"
                )
                cf.flush()

            for batch_idx in range(cached_batches, total_batches):
                start = batch_idx * args.batch_size
                end = min(start + args.batch_size, len(all_items))
                batch = all_items[start:end]
                B = len(batch)
                prompts = [f"{it['context']}\nAction:" for it in batch]

                try:
                    all_candidates, entropies = _generate_batch_vllm(
                        backend,
                        prompts,
                        K=K,
                        gen_max_tokens=gen_max_tokens,
                        gen_temperature=gen_temperature,
                        num_logprobs=args.num_logprobs,
                    )
                except Exception as exc:
                    all_candidates = [
                        [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * K
                    ] * B
                    entropies = [0.0] * B
                    _diag["gen_errors"] += 1
                    if _diag["gen_errors"] <= 3:
                        logger.warning(
                            f"vLLM generation failed (batch {batch_idx}): {exc}"
                        )

                cf.write(
                    json.dumps(
                        {"candidates": all_candidates, "entropies": entropies}
                    )
                    + "\n"
                )
                cf.flush()

                done = batch_idx + 1 - cached_batches
                elapsed = time.time() - t_gen
                rate = done / elapsed if elapsed > 0 else 0
                remaining = total_batches - batch_idx - 1
                eta = remaining / rate if rate > 0 else float("inf")
                pct = 100.0 * (batch_idx + 1) / total_batches
                logger.info(
                    f"[Phase 1] Batch {batch_idx + 1}/{total_batches} "
                    f"({pct:.1f}%) | {rate:.1f} batch/s | "
                    f"ETA {eta / 60:.1f}min"
                )

        gen_elapsed = time.time() - t_gen
        logger.info(
            f"[Phase 1] Complete: {total_batches} batches generated "
            f"in {gen_elapsed / 60:.1f}min"
        )

    if args.generate_only:
        total_elapsed = time.time() - t0
        logger.info(
            f"[shard {args.shard_id}] --generate-only: Phase 1 done. "
            f"{total_batches} batches ({len(all_items)} steps) cached "
            f"in {total_elapsed / 60:.1f}min -> {cache_path}"
        )
        logger.info(
            f"Run with --score-only to score candidates later (no GPU needed)."
        )
        if _diag["gen_errors"]:
            logger.warning(
                f"Generation fallbacks: {_diag['gen_errors']}/{total_batches} batches"
            )
        return

    # ── Phase 2: Score candidates with verifier (CPU-heavy) ───────
    logger.info(
        f"[Phase 2] Scoring {len(all_items)} steps with verifier "
        f"(CPU-heavy, GPU keepalive active)..."
    )

    if keepalive_interval > 0:
        try:
            import torch

            if torch.cuda.is_available():
                _start_gpu_keepalive("cuda", keepalive_interval)
                logger.info(
                    f"[Phase 2] GPU keepalive enabled "
                    f"(interval={keepalive_interval:.1f}s)"
                )
        except Exception:
            pass

    existing_records = 0
    if shard_output.exists() and file_mode != "w":
        with open(shard_output) as f:
            existing_records = sum(1 for _ in f)

    start_batch = 0
    records_counted = 0
    for bi in range(total_batches):
        bs = bi * args.batch_size
        be = min(bs + args.batch_size, len(all_items))
        batch_len = be - bs
        if records_counted + batch_len <= existing_records:
            records_counted += batch_len
            start_batch = bi + 1
        else:
            break

    if start_batch > 0:
        logger.info(
            f"[Phase 2] Resuming from batch {start_batch}/{total_batches} "
            f"({existing_records} records already written)"
        )

    total_features = existing_records
    t_score = time.time()

    out_mode = file_mode if start_batch == 0 else "a"

    with open(shard_output, out_mode) as f, open(cache_path) as cache_f:
        cache_f.readline()  # skip header

        for line_idx, cache_line in enumerate(cache_f):
            if line_idx < start_batch:
                continue

            batch_idx = line_idx
            bs = batch_idx * args.batch_size
            be = min(bs + args.batch_size, len(all_items))
            batch = all_items[bs:be]

            cache_entry = json.loads(cache_line)
            all_candidates = cache_entry["candidates"]
            entropies = cache_entry["entropies"]

            records, batch_ent, batch_v, batch_v_spread, ver_fb = (
                _score_and_assemble(
                    verifier,
                    batch,
                    all_candidates,
                    entropies,
                    K,
                    max_steps,
                    logger,
                    _diag,
                    batch_idx,
                )
            )

            for rec in records:
                _diag["all_entropies"].append(rec.pop("_entropy"))
                _diag["all_v_means"].append(rec.pop("_v_mean"))
                _diag["all_v_spreads"].append(rec.pop("_v_spread"))
                _diag["all_lp_means"].append(rec.pop("_lp_mean"))
                _diag["all_consistencies"].append(rec.pop("_consistency"))
                _diag["all_successes"].append(rec["slm_success"])
                f.write(json.dumps(rec) + "\n")
                total_features += 1

            f.flush()

            scored = total_features - existing_records
            elapsed = time.time() - t_score
            rate = scored / elapsed if elapsed > 0 else 0
            remaining = len(all_items) - total_features
            eta = remaining / rate if rate > 0 else float("inf")
            pct = (
                100.0 * total_features / len(all_items)
                if all_items
                else 100.0
            )

            fb_str = " FALLBACK:[VER]" if ver_fb else ""
            theory_str = ""
            if total_features >= 64:
                ts = _theory_summary(_diag)
                if ts:
                    theory_str = f" | {ts}"

            logger.info(
                f"[Phase 2] Batch {batch_idx + 1}/{total_batches} "
                f"({pct:.1f}%) | {total_features}/{len(all_items)} steps | "
                f"{rate:.1f} steps/s | ETA {eta / 60:.1f}min | "
                f"v_mean={np.mean(batch_v):.3f} "
                f"v_spread={np.mean(batch_v_spread):.3f}"
                f"{fb_str}{theory_str}"
            )

            if (
                (batch_idx + 1) % _DIAG_INTERVAL == 0
                and _diag["all_entropies"]
            ):
                _log_running_report(
                    logger, _diag, total_features, len(all_items)
                )

    _stop_gpu_keepalive()

    total_elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_features} feature vectors "
        f"from {num_episodes} episodes in {total_elapsed / 60:.1f}min "
        f"({total_features / max(total_elapsed, 1e-6):.1f} steps/s) "
        f"-> {shard_output}"
    )

    if _diag["all_entropies"]:
        _log_running_report(logger, _diag, total_features, len(all_items))
        total_errs = (
            _diag["entropy_errors"]
            + _diag["gen_errors"]
            + _diag["verifier_errors"]
        )
        if total_errs:
            logger.warning(
                f"Total silent fallbacks: entropy={_diag['entropy_errors']}  "
                f"gen={_diag['gen_errors']}  "
                f"verifier={_diag['verifier_errors']}  "
                f"({100 * total_errs / max(total_batches, 1):.1f}% of batches)"
            )

    if not args.score_only:
        try:
            cache_path.unlink()
            logger.info(f"Cleaned up candidate cache: {cache_path}")
        except Exception:
            pass
    else:
        logger.info(f"Keeping candidate cache (--score-only): {cache_path}")


if __name__ == "__main__":
    main()
