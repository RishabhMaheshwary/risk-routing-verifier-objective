#!/usr/bin/env python3
"""
Train the risk-calibrated router with Lagrangian CVaR objective.

Usage:
    python scripts/train_router.py \
        --config configs/webarena/noisy.yaml \
        --output outputs/router/webarena \
        --features data/router_features/webarena.jsonl

The router decides when to escalate from SLM to teacher LLM
based on calibrated confidence scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import RouterDataset, RouterExample
from r2v.data.splits import load_and_split
from r2v.models.router import Router
from r2v.training.router_trainer import RouterTrainer
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def _to_json_safe(value: Any) -> Any:
    """Recursively convert numpy/torch scalar types into JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Train risk-calibrated router")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--features", type=str, required=True,
        help="Router features (.jsonl or .parquet/.parquet-dir)",
    )
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Trajectory JSONL for task-level split (episode_id → task_id)")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Split value to use for training when features contain a split column")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Split value to use for validation when features contain a split column")
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
            "Optional transform over feature vectors before training: "
            "none | no_entropy | verifier_pseudo_entropy"
        ),
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def _is_parquet_source(path: str) -> bool:
    p = Path(path)
    return p.is_dir() or p.suffix.lower() in {".parquet", ".pq"}


def _normalize_record(record: dict) -> dict:
    """Normalize one feature record from JSONL/Parquet into a common dict format."""
    out = dict(record)
    out["features"] = [float(x) for x in out["features"]]
    out["slm_success"] = float(out.get("slm_success", 0.0))
    out["cost"] = float(out.get("cost", 1.0))
    out["episode_id"] = str(out.get("episode_id", ""))
    out["step_idx"] = int(out.get("step_idx", 0))
    out["perturbation_seed"] = int(out.get("perturbation_seed", 0))
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
    """Compute pseudo entropy as entropy(softmax(verifier_scores)).

    Falls back to a score-summary approximation only when raw verifier score
    populations are unavailable in the dataset.
    """
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
        # Zero-out entropy feature while preserving dimensionality.
        out[0] = 0.0
        return out

    if mode == "verifier_pseudo_entropy":
        # Replace entropy feature with verifier-driven pseudo entropy.
        out[0] = _compute_verifier_pseudo_entropy(out, record.get("verifier_scores"))
        return out

    raise ValueError(f"Unknown feature transform mode: {mode}")


_PARQUET_NEEDED_COLS = [
    "features", "slm_success", "cost", "episode_id",
    "step_idx", "perturbation_seed", "verifier_scores",
    "split", "benchmark", "model", "variant", "category",
]


def load_feature_records(
    path: str,
    feature_transform: str = "none",
    split: str | None = None,
    benchmark_filter: str | None = None,
    model_filter: str | None = None,
    variant_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Load feature records from JSONL or Parquet with optional metadata filters."""
    if _is_parquet_source(path):
        if pd is None:
            raise RuntimeError("pandas is required to read Parquet router datasets")

        # Only load the columns actually needed — much faster than reading all columns.
        available = set(pd.read_parquet(path, columns=[]).columns) if False else None
        try:
            df = pd.read_parquet(path, columns=_PARQUET_NEEDED_COLS)
        except Exception:
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

        return _load_records_from_df(df, feature_transform)

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _normalize_record(json.loads(line))
            r["features"] = _apply_feature_transform(r, feature_transform)
            records.append(r)
    return records


def _load_records_from_df(df: "pd.DataFrame", feature_transform: str) -> list[dict]:
    """Vectorised conversion of a filtered DataFrame → list[dict].

    Avoids the slow ``df.to_dict(orient='records')`` path by extracting arrays
    with numpy and only building one Python dict per row at the very end.
    """
    import numpy as np

    n = len(df)
    if n == 0:
        return []

    # Feature matrix — stack list-of-lists column into a 2-D array.
    raw_feats = df["features"].to_numpy()
    try:
        feats = np.stack(raw_feats).astype(np.float32)
    except ValueError:
        feats = np.array([list(x) for x in raw_feats], dtype=np.float32)

    # Apply feature transform in-place (vectorised where possible).
    if feature_transform == "no_entropy":
        feats[:, 0] = 0.0
    elif feature_transform == "verifier_pseudo_entropy":
        # verifier_pseudo_entropy is per-row (needs verifier_scores); keep the
        # Python loop only for this branch.
        vs_col = df["verifier_scores"].to_numpy() if "verifier_scores" in df.columns else None
        for i in range(n):
            vs = list(vs_col[i]) if (vs_col is not None and vs_col[i] is not None) else None
            feats[i, 0] = _compute_verifier_pseudo_entropy(feats[i].tolist(), vs)
    elif feature_transform != "none":
        raise ValueError(f"Unknown feature transform: {feature_transform}")

    slm_success  = df["slm_success"].to_numpy().astype(np.float32)
    cost         = df["cost"].to_numpy().astype(np.float32) if "cost" in df.columns else np.ones(n, np.float32)
    episode_ids  = df["episode_id"].to_numpy().astype(str) if "episode_id" in df.columns else np.array([str(i) for i in range(n)])
    step_idxs    = df["step_idx"].to_numpy().astype(np.int64) if "step_idx" in df.columns else np.zeros(n, np.int64)
    pert_seeds   = df["perturbation_seed"].to_numpy().astype(np.int64) if "perturbation_seed" in df.columns else np.zeros(n, np.int64)

    vs_col = df["verifier_scores"].to_numpy() if "verifier_scores" in df.columns else None

    records = []
    for i in range(n):
        records.append({
            "features":         feats[i].tolist(),
            "slm_success":      float(slm_success[i]),
            "cost":             float(cost[i]),
            "episode_id":       str(episode_ids[i]),
            "step_idx":         int(step_idxs[i]),
            "perturbation_seed": int(pert_seeds[i]),
            "verifier_scores":  (list(vs_col[i]) if vs_col is not None and vs_col[i] is not None else None),
        })
    return records


def load_router_features(
    records: list[dict],
    allowed_episode_ids: set[str] | None = None,
) -> list[RouterExample]:
    """Convert normalized feature records into RouterExample objects.

    Each line: {"features": [...], "slm_success": 0/1, "cost": 1.0,
                "episode_id": ..., "step_idx": ...}

    Parameters
    ----------
    records
        List of feature records from JSONL/Parquet.
    allowed_episode_ids
        If provided, only include records whose episode_id is in this set.
    """
    examples = []
    skipped = 0
    for record in records:
        if allowed_episode_ids is not None:
            eid = record.get("episode_id")
            if eid is not None and eid not in allowed_episode_ids:
                skipped += 1
                continue
        slm_success = float(record.get("slm_success", 0))
        examples.append(RouterExample(
            features=record["features"],
            label=1.0 - slm_success,
            success=slm_success,
            perturbation_seed=record.get("perturbation_seed", 0),
            cost=record.get("cost", 1.0),
        ))
    if skipped:
        logging.getLogger(__name__).info(f"Skipped {skipped} records outside split")
    return examples


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    jsonl_log = JSONLLogger(output_dir / "training_log.jsonl")
    jsonl_log.log_config(config_to_dict(cfg))

    init_wandb(
        project=cfg.get("logging", {}).get("project", "r2v-agent"),
        name=f"router_{cfg.get('data', {}).get('benchmark', 'unknown')}",
        config=config_to_dict(cfg),
        tags=["router"],
        mode=cfg.get("logging", {}).get("wandb_mode", "online"),
    )

    train_eids: set[str] | None = None
    val_eids: set[str] | None = None

    if _is_parquet_source(args.features):
        train_records = load_feature_records(
            args.features,
            feature_transform=args.feature_transform,
            split=args.train_split,
            benchmark_filter=args.benchmark_filter,
            model_filter=args.model_filter,
            variant_filter=args.variant_filter,
            category_filter=args.category_filter,
        )
        val_records = load_feature_records(
            args.features,
            feature_transform=args.feature_transform,
            split=args.val_split,
            benchmark_filter=args.benchmark_filter,
            model_filter=args.model_filter,
            variant_filter=args.variant_filter,
            category_filter=args.category_filter,
        )
        if args.trajectories:
            logger.warning("Ignoring --trajectories because split-aware Parquet features were provided")
    else:
        all_records = load_feature_records(args.features, feature_transform=args.feature_transform)

        # Build task-level split from trajectories (if provided)
        if args.trajectories and Path(args.trajectories).exists():
            max_perturbations = int(cfg.get("data", {}).get("max_perturbations_per_task", 2))
            splits = load_and_split(
                args.trajectories,
                max_perturbations_per_task=max_perturbations,
                seed=int(cfg.get("project", {}).get("seed", 42)),
            )
            train_eids = {ep.episode_id for ep in splits["train"]}
            val_eids = {ep.episode_id for ep in splits["val"]}
            logger.info(
                f"Task-level split: {len(train_eids)} train, "
                f"{len(val_eids)} val, {len(splits['test'])} test episode IDs"
            )

        train_records = all_records
        val_records = all_records

    train_examples = load_router_features(train_records, allowed_episode_ids=train_eids)
    val_examples = load_router_features(val_records, allowed_episode_ids=val_eids)
    logger.info(f"Router features: {len(train_examples)} train, {len(val_examples)} val samples")

    if not train_examples:
        raise RuntimeError("No training examples found after applying filters")
    if not val_examples:
        raise RuntimeError("No validation examples found after applying filters")

    all_examples = train_examples + val_examples
    successes = [ex.success for ex in all_examples]
    logger.info(f"  SLM success rate: {sum(successes) / len(successes):.3f}")
    logger.info(f"  Mean cost: {sum(ex.cost for ex in all_examples) / len(all_examples):.3f}")

    train_dataset = RouterDataset(train_examples)
    # Val/test must use training-set normalization stats to avoid leakage.
    val_dataset = RouterDataset(val_examples, mean=train_dataset.mean, std=train_dataset.std)

    # Create router — Router takes a config dict, not keyword args.
    # Override input_features to match actual feature dimensionality from data.
    rcfg = OmegaConf.to_container(cfg.get("router", {}), resolve=True)
    input_dim = len(all_examples[0].features)

    # Build a config that tells Router the exact input dim
    router_config = dict(rcfg)
    # Override _compute_input_dim by setting policy_hidden_dim so total = input_dim
    # Simplest: just set the feature flags to False and use policy_hidden_dim as the dim
    router_config["input_features"] = {
        "verifier_score": False,
        "entropy": False,
        "step_number": False,
        "token_count": False,
        "policy_hidden_dim": input_dim - 1,  # +1 for the always-included risk_score
    }
    router = Router(router_config)
    logger.info(f"Router input_dim={input_dim} (from data)")
    logger.info(f"Router feature_transform={args.feature_transform}")

    # Train
    train_cfg = OmegaConf.to_container(cfg.get("training", {}).get("router", {}), resolve=True)
    trainer = RouterTrainer(
        router=router,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=train_cfg,
        checkpoint_metadata={
            "input_dim": input_dim,
            "feature_mean": train_dataset.mean.tolist(),
            "feature_std": train_dataset.std.tolist(),
            "feature_transform": args.feature_transform,
            "full_config": config_to_dict(cfg),
        },
        output_dir=str(output_dir),
    )

    train_result = _to_json_safe(trainer.train())
    logger.info(
        "Router training artifacts: "
        f"best_epoch={train_result.get('best_epoch')}, "
        f"best_eval_brier={train_result.get('best_eval_brier')}, "
        f"best_checkpoint_path={train_result.get('best_checkpoint_path')}"
    )

    # RouterTrainer already performs post-hoc calibration on eval data.
    # Re-calibrating here can drift temperature and, if labels are mismatched,
    # push to boundary values (e.g., T=10.0).
    calibrated_temperature = float(router.temperature.item())
    logger.info(f"Using trainer-calibrated temperature: {calibrated_temperature:.3f}")

    # Save
    torch.save({
        "router_state_dict": router.state_dict(),
        "temperature": calibrated_temperature,
        "input_dim": input_dim,
        "config": config_to_dict(cfg),
        "feature_mean": train_dataset.mean.tolist(),
        "feature_std": train_dataset.std.tolist(),
        "feature_transform": args.feature_transform,
        "training_summary": train_result,
    }, output_dir / "router_final.pt")

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(train_result, f, indent=2)

    logger.info(f"Router saved to {output_dir / 'router_final.pt'}")
    jsonl_log.log("training_complete", {"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
