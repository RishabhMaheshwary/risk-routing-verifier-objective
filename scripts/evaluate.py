#!/usr/bin/env python3
"""
Evaluate trained R2V-Agent and baselines on benchmarks (offline).

Performs offline evaluation using pre-computed router features and
trajectory data.  For each episode the router decides SLM vs LLM
at every step; the episode outcome is simulated based on whether
the SLM originally succeeded and whether the router routed to the
LLM teacher.

Usage:
    python scripts/evaluate.py \
        --config configs/gaia/noisy.yaml \
        --features data/router_features/gaia.jsonl \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --router-path outputs/router/gaia_noisy/router_final.pt \
        --output results/gaia_noisy \
        --seeds 1 2 3 \
        --methods r2v slm_only llm_only entropy_router

Produces:
- Structured JSON results (for LLM paper writing)
- CSV tables (for plotting)
- LaTeX tables (for paper)
- Statistical comparisons with CIs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.splits import load_and_split
from r2v.data.trajectory import Episode, TrajectoryStore
from r2v.evaluation.calibration import compute_calibration_metrics
from r2v.evaluation.statistical import bootstrap_ci, paired_mcnemar_test
from r2v.models.router import Router
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, setup_logging
from r2v.utils.results import (
    ComparisonResult,
    EvalResult,
    ResultsBundle,
    ResultsManager,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate R2V-Agent (offline)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features", type=str, required=True,
                        help="Router features (.jsonl or .parquet/.parquet-dir)")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Trajectory JSONL for perturbation-seed metadata")
    parser.add_argument("--split", type=str, default=None,
                        help="Optional split filter when features include a split column")
    parser.add_argument("--benchmark-filter", type=str, default=None,
                        help="Optional benchmark filter for Parquet dataset")
    parser.add_argument("--model-filter", type=str, default=None,
                        help="Optional model filter for Parquet dataset")
    parser.add_argument("--variant-filter", type=str, default=None,
                        help="Optional variant filter for Parquet dataset")
    parser.add_argument("--category-filter", type=str, default=None,
                        help="Optional category filter for Parquet dataset")
    parser.add_argument(
        "--feature-transform",
        type=str,
        default="none",
        choices=["none", "no_entropy", "verifier_pseudo_entropy"],
        help=(
            "Optional transform over feature vectors before evaluation: "
            "none | no_entropy | verifier_pseudo_entropy"
        ),
    )
    parser.add_argument("--router-path", type=str, default=None,
                        help="Path to router_final.pt checkpoint")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Perturbation seeds to evaluate (for robustness)")
    parser.add_argument("--methods", nargs="+",
                        default=["r2v", "slm_only", "llm_only", "entropy_router",
                                 "oracle_router", "heuristic_router"])
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument(
        "--router-threshold-sweep", type=float, nargs="+", default=None,
        help=(
            "Optional sweep for R2V threshold (e.g., 0.2 0.3 0.4 0.5). "
            "When set, method 'r2v' expands into 'r2v@<threshold>' variants."
        ),
    )
    parser.add_argument("--entropy-threshold", type=float, default=2.0,
                        help="Entropy threshold for entropy_router baseline")
    parser.add_argument("--verifier-threshold", type=float, default=0.5,
                        help="Best-verifier-score threshold for verifier_router baseline")
    parser.add_argument(
        "--feature-mask", type=int, nargs="+", default=None,
        help=(
            "Indices of features to KEEP (all others zeroed). "
            "E.g., --feature-mask 0 1 2 3 4 keeps only entropy+verifier stats. "
            "Used for inference-time feature ablation without router retraining."
        ),
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device for router inference (default: cuda if available, else cpu)",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


# ============================================================
# Data loading helpers
# ============================================================

def _is_parquet_source(path: str) -> bool:
    p = Path(path)
    return p.is_dir() or p.suffix.lower() in {".parquet", ".pq"}


def _normalize_record(record: dict) -> dict:
    out = dict(record)
    out["features"] = [float(x) for x in out["features"]]
    out["slm_success"] = float(out.get("slm_success", 0.0))
    out["cost"] = float(out.get("cost", 1.0))
    out["episode_id"] = str(out.get("episode_id", ""))
    out["step_idx"] = int(out.get("step_idx", 0))
    if "perturbation_seed" in out and out["perturbation_seed"] is not None:
        out["perturbation_seed"] = int(out["perturbation_seed"])
    if out.get("verifier_scores") is not None:
        out["verifier_scores"] = [float(x) for x in out["verifier_scores"]]
    else:
        out["verifier_scores"] = None
    return out


def _softmax_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = [float(s) for s in scores]
    m = max(arr)
    exps = [math.exp(x - m) for x in arr]
    z = sum(exps)
    if z <= 1e-12:
        return 0.0
    probs = [e / z for e in exps]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _compute_verifier_pseudo_entropy(features: list[float], verifier_scores: list[float] | None) -> float:
    if verifier_scores:
        return _softmax_entropy(verifier_scores)

    # Fallback approximation from summary stats.
    if len(features) < 6:
        return 0.0
    mean = float(features[2])
    std = abs(float(features[3]))
    best = float(features[4])
    worst = float(features[5])
    approx_scores = [best, mean + std, mean, mean - std, worst]
    return _softmax_entropy(approx_scores)


def _apply_feature_transform(record: dict, mode: str) -> list[float]:
    features = record["features"]
    if mode == "none":
        return [float(x) for x in features]

    out = [float(x) for x in features]
    if not out:
        return out

    if mode == "no_entropy":
        out[0] = 0.0
        return out

    if mode == "verifier_pseudo_entropy":
        out[0] = _compute_verifier_pseudo_entropy(out, record.get("verifier_scores"))
        return out

    raise ValueError(f"Unknown feature transform mode: {mode}")


def load_features(
    path: str,
    feature_transform: str = "none",
    split: str | None = None,
    benchmark_filter: str | None = None,
    model_filter: str | None = None,
    variant_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Load router features from JSONL or Parquet into normalized record dicts."""
    if _is_parquet_source(path):
        if pd is None:
            raise RuntimeError("pandas is required to read Parquet router datasets")
        df = pd.read_parquet(path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if benchmark_filter is not None and "benchmark" in df.columns:
            df = df[df["benchmark"] == benchmark_filter]
        if model_filter is not None and "model" in df.columns:
            df = df[df["model"] == model_filter]
        if variant_filter is not None and "variant" in df.columns:
            df = df[df["variant"] == variant_filter]
        if category_filter is not None and "category" in df.columns:
            df = df[df["category"] == category_filter]
        out = [_normalize_record(r) for r in df.to_dict(orient="records")]
        for r in out:
            r["features"] = _apply_feature_transform(r, feature_transform)
        return out

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = _normalize_record(json.loads(line))
                r["features"] = _apply_feature_transform(r, feature_transform)
                records.append(r)
    return records


def load_episode_metadata(traj_path: str) -> dict[str, dict]:
    """Load trajectory JSONL and extract per-episode metadata.

    Returns dict: episode_id → {perturbation_seed, perturbation_type, success}
    """
    meta = {}
    store = TrajectoryStore(traj_path)
    for ep in store.iter_episodes():
        pt = ep.perturbation_type
        # Handle both enum and raw string
        if hasattr(pt, "value"):
            pt_str = pt.value
        elif pt:
            pt_str = str(pt)
        else:
            pt_str = "none"
        meta[ep.episode_id] = {
            "perturbation_seed": ep.perturbation_seed or 0,
            "perturbation_type": pt_str,
            "success": ep.success,
            "num_steps": ep.num_steps,
        }
    return meta


def group_by_episode(records: list[dict]) -> dict[str, list[dict]]:
    """Group step-level feature records by episode_id."""
    episodes = defaultdict(list)
    for r in records:
        episodes[r["episode_id"]].append(r)
    # Sort steps within each episode by step_idx
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda r: r["step_idx"])
    return dict(episodes)


def load_episode_metadata_from_records(records: list[dict]) -> dict[str, dict]:
    """Build per-episode metadata from record-level fields when trajectories are absent."""
    meta: dict[str, dict] = {}
    for r in records:
        eid = r.get("episode_id")
        if not eid:
            continue
        if eid not in meta:
            meta[eid] = {
                "perturbation_seed": int(r.get("perturbation_seed", 0)) if r.get("perturbation_seed") is not None else 0,
                "perturbation_type": str(r.get("perturbation_type", "unknown")),
            }
    return meta


# ============================================================
# Router loading
# ============================================================

def load_router(router_path: str, cfg, device: str = "cpu") -> tuple[Router, float, np.ndarray, np.ndarray]:
    """Load trained router from checkpoint.

    Returns (router, temperature, feature_mean, feature_std) tuple.
    """
    ckpt = torch.load(router_path, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", 13)

    # Reconstruct Router config (same logic as train_router.py)
    rcfg = OmegaConf.to_container(cfg.get("router", {}), resolve=True)
    router_config = dict(rcfg)
    router_config["input_features"] = {
        "verifier_score": False,
        "entropy": False,
        "step_number": False,
        "token_count": False,
        "policy_hidden_dim": input_dim - 1,
    }

    router = Router(router_config)
    router.load_state_dict(ckpt["router_state_dict"])
    router.to(device)
    router.eval()

    temperature = ckpt.get("temperature", 1.0)
    feature_mean = np.array(ckpt["feature_mean"], dtype=np.float32) if "feature_mean" in ckpt else np.zeros(input_dim, dtype=np.float32)
    feature_std = np.array(ckpt["feature_std"], dtype=np.float32) if "feature_std" in ckpt else np.ones(input_dim, dtype=np.float32)
    return router, temperature, feature_mean, feature_std


# ============================================================
# Per-method episode evaluation
# ============================================================

def evaluate_episode_r2v(
    steps: list[dict],
    router: Router,
    temperature: float,
    threshold: float,
    cost_slm: float,
    cost_llm: float,
    feature_mean: "np.ndarray",
    feature_std: "np.ndarray",
    feature_mask: list[int] | None = None,
) -> dict:
    """Evaluate one episode under R2V routing.

    For each step, the router decides SLM vs LLM.
    - If any step is routed to LLM and slm_success=0 → episode success = 1
      (LLM assumed to fix the failing trajectory).
    - Cost is the sum of per-step routing costs.

    Args:
        feature_mask: If set, indices of features to KEEP; all others are zeroed.
            Used for inference-time feature ablation (feat_* ablations).
    """
    slm_success = steps[0]["slm_success"]
    raw = np.array([s["features"] for s in steps], dtype=np.float32)
    normed = (raw - feature_mean) / feature_std
    if feature_mask is not None:
        mask = np.zeros(normed.shape[1], dtype=np.float32)
        for idx in feature_mask:
            if idx < normed.shape[1]:
                mask[idx] = 1.0
        normed = normed * mask
    features = torch.tensor(normed, dtype=torch.float32)

    with torch.no_grad():
        # Get raw logits → apply temperature → sigmoid
        logits = router.mlp(features).squeeze(-1)
        scaled_logits = logits / max(temperature, 0.01)
        probs = torch.sigmoid(scaled_logits).numpy()

    decisions = (probs > threshold).astype(float)  # 1 = LLM, 0 = SLM
    llm_steps = int(decisions.sum())
    slm_steps = len(steps) - llm_steps

    # Episode success logic:
    # If SLM would have succeeded anyway → success regardless of routing
    # If SLM would have failed → success only if router routed to LLM
    if slm_success >= 0.5:
        success = 1.0
    else:
        success = 1.0 if llm_steps > 0 else 0.0

    cost = slm_steps * cost_slm + llm_steps * cost_llm
    llm_call_rate = llm_steps / max(len(steps), 1)

    return {
        "success": success,
        "cost": cost,
        "llm_call_rate": llm_call_rate,
        "llm_steps": llm_steps,
        "total_steps": len(steps),
        "fallback_probs": probs.tolist(),
    }


def evaluate_episode_slm_only(
    steps: list[dict],
    cost_slm: float,
) -> dict:
    """SLM-only baseline: never route to LLM."""
    slm_success = steps[0]["slm_success"]
    return {
        "success": float(slm_success >= 0.5),
        "cost": len(steps) * cost_slm,
        "llm_call_rate": 0.0,
        "llm_steps": 0,
        "total_steps": len(steps),
    }


def evaluate_episode_llm_only(
    steps: list[dict],
    cost_llm: float,
) -> dict:
    """LLM-only baseline: always route to LLM."""
    return {
        "success": 1.0,
        "cost": len(steps) * cost_llm,
        "llm_call_rate": 1.0,
        "llm_steps": len(steps),
        "total_steps": len(steps),
    }


def evaluate_episode_entropy_router(
    steps: list[dict],
    entropy_threshold: float,
    cost_slm: float,
    cost_llm: float,
) -> dict:
    """Entropy-router baseline: route to LLM when entropy exceeds threshold.

    Entropy is features[0] (first element of feature vector).
    """
    slm_success = steps[0]["slm_success"]
    entropies = [s["features"][0] for s in steps]
    decisions = [1.0 if e > entropy_threshold else 0.0 for e in entropies]
    llm_steps = int(sum(decisions))
    slm_steps = len(steps) - llm_steps

    if slm_success >= 0.5:
        success = 1.0
    else:
        success = 1.0 if llm_steps > 0 else 0.0

    cost = slm_steps * cost_slm + llm_steps * cost_llm
    llm_call_rate = llm_steps / max(len(steps), 1)

    return {
        "success": success,
        "cost": cost,
        "llm_call_rate": llm_call_rate,
        "llm_steps": llm_steps,
        "total_steps": len(steps),
    }


def evaluate_episode_oracle_router(
    steps: list[dict],
    cost_slm: float,
    cost_llm: float,
) -> dict:
    """Oracle router: routes to LLM exactly when SLM would fail.

    Uses ground-truth slm_success — impossible in deployment, but establishes
    the theoretical ceiling for any routing strategy at the same LLM call rate.
    """
    slm_success = steps[0]["slm_success"]
    # Oracle: route ALL steps to LLM iff episode would fail under SLM
    llm_steps = 0 if slm_success >= 0.5 else len(steps)
    slm_steps = len(steps) - llm_steps

    cost = slm_steps * cost_slm + llm_steps * cost_llm
    return {
        "success": 1.0,  # oracle always achieves success (by construction)
        "cost": cost,
        "llm_call_rate": llm_steps / max(len(steps), 1),
        "llm_steps": llm_steps,
        "total_steps": len(steps),
    }


def evaluate_episode_heuristic_router(
    steps: list[dict],
    cost_slm: float,
    cost_llm: float,
    entropy_threshold: float = 2.5,
    verifier_mean_threshold: float = 0.4,
) -> dict:
    """Rule-based heuristic router using entropy and verifier score.

    Feature layout (from generate_router_features.py):
      0: entropy
      1: verifier_score_spread
      2: verifier_score_mean
      3: verifier_score_std
      4: verifier_score_best
      5: horizon_fraction
      ...

    Escalates to LLM when entropy is high OR verifier mean is low.
    This is the training-free baseline that tests whether a learned
    router adds value over hand-crafted heuristics.
    """
    slm_success = steps[0]["slm_success"]
    decisions = []
    for s in steps:
        feats = s["features"]
        entropy = feats[0] if len(feats) > 0 else 0.0
        verifier_mean = feats[2] if len(feats) > 2 else 1.0
        escalate = (entropy > entropy_threshold) or (verifier_mean < verifier_mean_threshold)
        decisions.append(1.0 if escalate else 0.0)

    llm_steps = int(sum(decisions))
    slm_steps = len(steps) - llm_steps

    if slm_success >= 0.5:
        success = 1.0
    else:
        success = 1.0 if llm_steps > 0 else 0.0

    cost = slm_steps * cost_slm + llm_steps * cost_llm
    return {
        "success": success,
        "cost": cost,
        "llm_call_rate": llm_steps / max(len(steps), 1),
        "llm_steps": llm_steps,
        "total_steps": len(steps),
    }


def evaluate_episode_verifier_router(
    steps: list[dict],
    verifier_threshold: float,
    cost_slm: float,
    cost_llm: float,
) -> dict:
    """Verifier-score-only router: escalates when best verifier score < threshold.

    Feature 4 = verifier_score_best. Tests whether the verifier signal alone
    (without entropy or step features) is sufficient for good routing.
    """
    slm_success = steps[0]["slm_success"]
    decisions = []
    for s in steps:
        feats = s["features"]
        best_score = feats[4] if len(feats) > 4 else 1.0
        decisions.append(1.0 if best_score < verifier_threshold else 0.0)

    llm_steps = int(sum(decisions))
    slm_steps = len(steps) - llm_steps

    if slm_success >= 0.5:
        success = 1.0
    else:
        success = 1.0 if llm_steps > 0 else 0.0

    cost = slm_steps * cost_slm + llm_steps * cost_llm
    return {
        "success": success,
        "cost": cost,
        "llm_call_rate": llm_steps / max(len(steps), 1),
        "llm_steps": llm_steps,
        "total_steps": len(steps),
    }


def _expand_methods(
    methods: list[str], threshold_sweep: list[float] | None,
) -> list[str]:
    """Expand 'r2v' into threshold-specific method names when sweep is enabled."""
    if not threshold_sweep:
        return list(methods)

    expanded: list[str] = []
    for method in methods:
        if method == "r2v":
            for thr in threshold_sweep:
                expanded.append(f"r2v@{thr:g}")
        else:
            expanded.append(method)
    return expanded


def _method_to_base_and_threshold(method: str, default_threshold: float) -> tuple[str, float | None]:
    """Parse method name into base method and optional threshold override."""
    if method.startswith("r2v@"):
        try:
            return "r2v", float(method.split("@", 1)[1])
        except ValueError:
            return "r2v", default_threshold
    if method == "r2v":
        return "r2v", default_threshold
    return method, None


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    benchmark = cfg.get("benchmark", cfg.get("data", {}).get("benchmark", "unknown"))
    condition = "noisy" if cfg.get("perturbations", {}).get(
        "tool_flakiness", {}).get("enabled", False) else "clean"

    # Cost parameters
    train_router_cfg = cfg.get("training", {}).get("router", {})
    cost_slm = float(train_router_cfg.get("cost_slm", 1.0))
    cost_llm = float(train_router_cfg.get("cost_llm", 50.0))

    jsonl_log = JSONLLogger(output_dir / "evaluation_log.jsonl")
    results_manager = ResultsManager(output_dir / "structured_results")

    bundle = ResultsBundle(
        experiment_name=f"eval_{benchmark}_{condition}",
        config=config_to_dict(cfg),
    )

    # ── Load data (test split only) ──
    logger.info(f"Loading router features from {args.features}...")
    records = load_features(
        args.features,
        feature_transform=args.feature_transform,
        split=args.split,
        benchmark_filter=args.benchmark_filter,
        model_filter=args.model_filter,
        variant_filter=args.variant_filter,
        category_filter=args.category_filter,
    )
    logger.info(f"  feature_transform={args.feature_transform}")
    logger.info(f"  {len(records)} step-level records (before split filter)")

    # Build test-split episode_id allowlist from trajectories
    test_eids: set[str] | None = None
    ep_meta = load_episode_metadata_from_records(records)
    if args.trajectories:
        logger.info(f"Loading trajectory metadata from {args.trajectories}...")
        splits = load_and_split(
            args.trajectories,
            max_perturbations_per_task=int(
                cfg.get("data", {}).get("max_perturbations_per_task", 2)
            ),
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )
        test_eids = {ep.episode_id for ep in splits["test"]}
        logger.info(
            f"  Using test split: {len(test_eids)} episode IDs "
            f"(train={len(splits['train'])}, val={len(splits['val'])})"
        )
        ep_meta = load_episode_metadata(args.trajectories)
        logger.info(f"  {len(ep_meta)} episodes with metadata")
    else:
        if args.split and _is_parquet_source(args.features):
            logger.info(
                "No --trajectories provided; using split-filtered records from features dataset"
            )
        else:
            logger.warning("No --trajectories provided; evaluating ALL episodes "
                           "(no test-split filter).")

    if test_eids is not None:
        records = [r for r in records if r.get("episode_id") in test_eids]
        logger.info(f"  {len(records)} step-level records after test-split filter")

    ep_groups = group_by_episode(records)
    logger.info(f"  {len(ep_groups)} episodes")

    eval_methods = _expand_methods(args.methods, args.router_threshold_sweep)

    # ── Resolve device ──
    import torch as _torch
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    # ── Load router (if evaluating R2V) ──
    router, temperature = None, 1.0
    feature_mean = feature_std = None
    if any(m == "r2v" or m.startswith("r2v@") for m in eval_methods):
        if args.router_path is None:
            logger.error("--router-path required for R2V evaluation")
            sys.exit(1)
        logger.info(f"Loading router from {args.router_path}...")
        router, temperature, feature_mean, feature_std = load_router(args.router_path, cfg, device=device)
        logger.info(f"  Temperature: {temperature:.4f}")
        if feature_mean is not None:
            logger.info("  Feature normalization stats loaded from checkpoint")
        if args.router_threshold_sweep:
            logger.info(
                "  Threshold sweep: "
                + ", ".join(f"{t:g}" for t in args.router_threshold_sweep)
            )

    # ── Pre-compute router probabilities for all episodes in one batched pass ──
    # This replaces the per-episode MLP call with a single vectorised forward,
    # which is ~N_episodes× faster (GPU or CPU) due to kernel launch amortisation.
    ep_router_probs: dict[str, np.ndarray] = {}
    if router is not None and feature_mean is not None:
        device = next(router.parameters()).device
        ep_ids_ordered = list(ep_groups.keys())
        step_counts = [len(ep_groups[eid]) for eid in ep_ids_ordered]

        # Build feature matrix for all steps across all episodes.
        raw_all = np.vstack([
            np.array([s["features"] for s in ep_groups[eid]], dtype=np.float32)
            for eid in ep_ids_ordered
        ])
        normed_all = (raw_all - feature_mean) / feature_std

        # Optional feature mask (zeroes out unused feature dimensions).
        if args.feature_mask is not None:
            mask = np.zeros(normed_all.shape[1], dtype=np.float32)
            for idx in args.feature_mask:
                if idx < normed_all.shape[1]:
                    mask[idx] = 1.0
            normed_all = normed_all * mask

        # Single batched inference — chunk to avoid OOM on very large datasets.
        _CHUNK = 131072
        all_probs_list = []
        router.eval()
        with torch.no_grad():
            for start in range(0, len(normed_all), _CHUNK):
                chunk_t = torch.tensor(
                    normed_all[start : start + _CHUNK], dtype=torch.float32
                ).to(device)
                logits = router.mlp(chunk_t).squeeze(-1)
                probs  = torch.sigmoid(logits / max(temperature, 0.01))
                all_probs_list.append(probs.cpu().numpy())
        router.train()

        all_probs_np = np.concatenate(all_probs_list)
        ptr = 0
        for eid, cnt in zip(ep_ids_ordered, step_counts):
            ep_router_probs[eid] = all_probs_np[ptr : ptr + cnt]
            ptr += cnt
        logger.info(f"  Batched router inference: {len(normed_all)} steps in one pass")

    # ── Evaluate each method ──
    # For robustness analysis, we group episodes by perturbation_seed
    # and evaluate per-seed success rates.
    all_method_results = {}  # method → list of per-episode result dicts

    for method in eval_methods:
        logger.info(f"=== Evaluating: {method} ===")
        episode_results = []
        base_method, method_threshold = _method_to_base_and_threshold(
            method, args.router_threshold,
        )

        for ep_id, steps in ep_groups.items():
            meta = ep_meta.get(ep_id, {})
            seed = meta.get("perturbation_seed", 0)

            if base_method == "r2v":
                threshold = (
                    method_threshold
                    if method_threshold is not None
                    else args.router_threshold
                )
                # Re-use pre-computed probabilities; fall back to per-episode
                # path only if pre-computation was skipped.
                if ep_id in ep_router_probs:
                    probs = ep_router_probs[ep_id]
                    decisions = (probs > threshold).astype(float)
                    llm_steps = int(decisions.sum())
                    slm_steps = len(steps) - llm_steps
                    slm_success = steps[0]["slm_success"]
                    success = 1.0 if slm_success >= 0.5 else (1.0 if llm_steps > 0 else 0.0)
                    cost = slm_steps * cost_slm + llm_steps * cost_llm
                    res = {
                        "success": success,
                        "cost": cost,
                        "llm_call_rate": llm_steps / max(len(steps), 1),
                        "llm_steps": llm_steps,
                        "total_steps": len(steps),
                        "fallback_probs": probs.tolist(),
                    }
                else:
                    res = evaluate_episode_r2v(
                        steps, router, temperature,
                        threshold, cost_slm, cost_llm,
                        feature_mean, feature_std,
                        feature_mask=args.feature_mask,
                    )
            elif base_method == "slm_only":
                res = evaluate_episode_slm_only(steps, cost_slm)
            elif base_method == "llm_only":
                res = evaluate_episode_llm_only(steps, cost_llm)
            elif base_method == "entropy_router":
                res = evaluate_episode_entropy_router(
                    steps, args.entropy_threshold, cost_slm, cost_llm,
                )
            elif base_method == "oracle_router":
                res = evaluate_episode_oracle_router(steps, cost_slm, cost_llm)
            elif base_method == "heuristic_router":
                res = evaluate_episode_heuristic_router(steps, cost_slm, cost_llm)
            elif base_method == "verifier_router":
                res = evaluate_episode_verifier_router(
                    steps, args.verifier_threshold, cost_slm, cost_llm,
                )
            else:
                logger.warning(f"Unknown method: {base_method}, skipping")
                continue

            res["episode_id"] = ep_id
            res["perturbation_seed"] = seed
            res["perturbation_type"] = meta.get("perturbation_type", "unknown")
            episode_results.append(res)

        all_method_results[method] = episode_results

        # ── Aggregate metrics ──
        successes = np.array([r["success"] for r in episode_results])
        costs = np.array([r["cost"] for r in episode_results])
        llm_rates = np.array([r["llm_call_rate"] for r in episode_results])

        sr, sr_lo, sr_hi = bootstrap_ci(successes)
        avg_cost, cost_lo, cost_hi = bootstrap_ci(costs)

        # Per-seed success rates
        seed_srs = defaultdict(list)
        for r in episode_results:
            seed_srs[r["perturbation_seed"]].append(r["success"])
        per_seed_sr = {s: float(np.mean(v)) for s, v in seed_srs.items()}

        worst_seed_sr = min(per_seed_sr.values()) if per_seed_sr else sr

        # CVaR failure
        if len(per_seed_sr) > 1:
            alpha = cfg.get("evaluation", {}).get("cvar_alpha", 0.2)
            failure_rates = sorted([1.0 - v for v in per_seed_sr.values()],
                                   reverse=True)
            k = max(1, int(len(failure_rates) * alpha))
            cvar_failure = float(np.mean(failure_rates[:k]))
        else:
            cvar_failure = 1.0 - sr

        # Router calibration (R2V only)
        ece, brier = None, None
        if base_method == "r2v":
            if router is None:
                logger.error("Router is not loaded for R2V calibration analysis")
                sys.exit(1)
            all_probs = []
            all_labels = []
            for r in episode_results:
                if "fallback_probs" in r:
                    all_probs.extend(r["fallback_probs"])
                    # label = 1 if SLM failed (should have fallen back)
                    label = 0.0 if r["success"] == 1.0 and r["llm_steps"] == 0 else 1.0
                    # Actually: label based on whether SLM would fail
                    slm_ep = steps[0]["slm_success"] if steps else 0
                    # Need to get slm_success from the episode's steps
            # Recalculate with access to raw data
            all_probs = []
            all_labels = []
            router.eval()
            for ep_id, steps in ep_groups.items():
                slm_success = steps[0]["slm_success"]
                raw = np.array([s["features"] for s in steps], dtype=np.float32)
                normed = (raw - feature_mean) / feature_std
                features = torch.tensor(normed, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = router.mlp(features).squeeze(-1)
                    scaled = logits / max(temperature, 0.01)
                    probs = torch.sigmoid(scaled).cpu().numpy()
                all_probs.extend(probs.tolist())
                # Ground truth: should router have routed to LLM?
                # 1 = yes (SLM failed), 0 = no (SLM succeeded)
                all_labels.extend([1.0 - slm_success] * len(steps))

            cal_metrics = compute_calibration_metrics(
                np.array(all_probs), np.array(all_labels)
            )
            ece = cal_metrics["ece"]
            brier = cal_metrics["brier"]

        eval_result = EvalResult(
            method=method,
            benchmark=benchmark,
            condition=condition,
            seed=0,  # aggregated across all seeds
            success_rate=sr,
            success_rate_ci=(sr_lo, sr_hi),
            worst_seed_sr=worst_seed_sr,
            cvar_failure=cvar_failure,
            avg_cost=float(avg_cost),
            llm_call_rate=float(np.mean(llm_rates)),
            ece=ece,
            brier=brier,
        )
        bundle.eval_results.append(eval_result)

        logger.info(
            f"  {method}: SR={sr:.3f} [{sr_lo:.3f}, {sr_hi:.3f}], "
            f"Worst-Seed={worst_seed_sr:.3f}, CVaR-Fail={cvar_failure:.3f}, "
            f"Cost={avg_cost:.2f}, LLM-Rate={float(np.mean(llm_rates)):.3f}"
        )
        if ece is not None:
            logger.info(f"  Calibration: ECE={ece:.4f}, Brier={brier:.4f}")

        jsonl_log.log_evaluation({
            "method": method,
            "base_method": base_method,
            "router_threshold": method_threshold if base_method == "r2v" else None,
            "success_rate": sr,
            "ci": (sr_lo, sr_hi),
            "worst_seed_sr": worst_seed_sr,
            "cvar_failure": cvar_failure,
            "avg_cost": float(avg_cost),
            "llm_call_rate": float(np.mean(llm_rates)),
            "ece": ece,
            "brier": brier,
            "per_seed_sr": per_seed_sr,
        })

    # ── Per-seed results (for each method, one EvalResult per seed) ──
    logger.info("Computing per-seed results...")
    for method in eval_methods:
        results = all_method_results[method]
        seed_groups = defaultdict(list)
        for r in results:
            seed_groups[r["perturbation_seed"]].append(r)

        for seed, seed_results in sorted(seed_groups.items()):
            successes = np.array([r["success"] for r in seed_results])
            costs = np.array([r["cost"] for r in seed_results])
            sr_val, lo, hi = bootstrap_ci(successes)

            bundle.eval_results.append(EvalResult(
                method=method,
                benchmark=benchmark,
                condition=condition,
                seed=seed,
                success_rate=sr_val,
                success_rate_ci=(lo, hi),
                avg_cost=float(np.mean(costs)),
                llm_call_rate=float(np.mean([r["llm_call_rate"] for r in seed_results])),
            ))

    # ── Statistical comparisons ──
    logger.info("Computing statistical comparisons...")
    r2v_methods = [m for m in eval_methods if m == "r2v" or m.startswith("r2v@")] 
    if r2v_methods:
        non_r2v_methods = [m for m in eval_methods if m not in r2v_methods]

        for r2v_method in r2v_methods:
            r2v_results = all_method_results[r2v_method]
            r2v_by_ep = {r["episode_id"]: r["success"] for r in r2v_results}

            for other in non_r2v_methods:
                other_results = all_method_results[other]
                other_by_ep = {r["episode_id"]: r["success"] for r in other_results}

                # Paired comparison on common episodes
                common_eps = sorted(set(r2v_by_ep.keys()) & set(other_by_ep.keys()))
                if not common_eps:
                    logger.warning(f"No common episodes for {r2v_method} vs {other}")
                    continue

                r2v_succ = np.array([r2v_by_ep[e] for e in common_eps])
                other_succ = np.array([other_by_ep[e] for e in common_eps])

                # McNemar test
                stat, p_value = paired_mcnemar_test(r2v_succ, other_succ)

                # Paired bootstrap CI for the difference
                diff_mean = float(np.mean(r2v_succ) - np.mean(other_succ))
                diff = r2v_succ - other_succ
                _, ci_lo, ci_hi = bootstrap_ci(diff)

                significance = cfg.get("evaluation", {}).get("mcnemar_significance", 0.05)

                bundle.comparisons.append(ComparisonResult(
                    method_a=r2v_method,
                    method_b=other,
                    benchmark=benchmark,
                    condition=condition,
                    metric="success_rate",
                    value_a=float(np.mean(r2v_succ)),
                    value_b=float(np.mean(other_succ)),
                    difference=diff_mean,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    p_value=p_value,
                    significant=p_value < significance,
                ))

                logger.info(
                    f"  {r2v_method} vs {other}: Δ={diff_mean:+.3f} "
                    f"[{ci_lo:.3f}, {ci_hi:.3f}], p={p_value:.4f}"
                    f"{' *' if p_value < significance else ''}"
                )

    # ── Per-perturbation-type analysis ──
    logger.info("Computing per-perturbation-type analysis...")
    for method in eval_methods:
        results = all_method_results[method]
        type_groups = defaultdict(list)
        for r in results:
            type_groups[r["perturbation_type"]].append(r["success"])

        for ptype, succs in sorted(type_groups.items()):
            logger.info(f"  {method} / {ptype}: SR={np.mean(succs):.3f} "
                        f"(n={len(succs)})")

    # ── Save all results ──
    bundle_path = results_manager.save_bundle(bundle)
    logger.info(f"Results bundle saved: {bundle_path}")

    csv_path = results_manager.generate_main_table_csv()
    logger.info(f"Main table CSV: {csv_path}")

    comp_csv = results_manager.generate_comparison_csv()
    logger.info(f"Comparisons CSV: {comp_csv}")

    llm_path = results_manager.generate_llm_summary()
    logger.info(f"LLM summary: {llm_path}")

    latex_path = results_manager.generate_latex_table()
    logger.info(f"LaTeX table: {latex_path}")

    # ── Summary table to stdout ──
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Method':<20} {'SR':>8} {'Worst':>8} {'CVaR-F':>8} "
                f"{'Cost':>8} {'LLM%':>8}")
    logger.info("-" * 70)
    seen_methods = set()
    for er in bundle.eval_results:
        # Print only one (aggregated) row per method.
        # Per-seed rows can also have seed=0 when metadata is missing,
        # so use presence of aggregate-only metrics and method dedup.
        if er.method in seen_methods:
            continue
        if er.worst_seed_sr is None and er.cvar_failure is None:
            continue
        seen_methods.add(er.method)
        worst = f"{er.worst_seed_sr:.3f}" if er.worst_seed_sr is not None else "—"
        cvar = f"{er.cvar_failure:.3f}" if er.cvar_failure is not None else "—"
        cost = f"{er.avg_cost:.1f}" if er.avg_cost is not None else "—"
        llm = f"{er.llm_call_rate:.3f}" if er.llm_call_rate is not None else "—"
        logger.info(f"{er.method:<20} {er.success_rate:>8.3f} {worst:>8} "
                     f"{cvar:>8} {cost:>8} {llm:>8}")
    logger.info("=" * 70)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
