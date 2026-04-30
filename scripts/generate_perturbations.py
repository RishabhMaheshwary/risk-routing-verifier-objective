#!/usr/bin/env python3
"""
Apply perturbations to collected trajectories.

Usage:
    python -m scripts.generate_perturbations \
        --config configs/gaia/noisy.yaml \
        --input data/runs/gaia_teacher/trajectories.jsonl \
        --output data/trajectories/gaia_noisy/trajectories.jsonl \
        --seeds 1 2 3

This script:
1. Loads clean trajectories from a JSONL file
2. Builds a perturbation pipeline from the noisy config
3. Applies each perturbation seed → N_clean × N_seeds perturbed episodes
4. Appends the *original* clean episodes + perturbed episodes to the output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.perturbations import (
    DistractorPerturbation,
    PartialObservabilityPerturbation,
    PerturbationPipeline,
    PerturbationRegistry,
    PromptInjectionPerturbation,
    ToolFlakinessPerturbation,
)
from r2v.data.trajectory import PerturbationType, TrajectoryStore
from omegaconf import OmegaConf

from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging


def build_pipeline(cfg) -> PerturbationPipeline:
    """Build perturbation pipeline from config using PerturbationRegistry."""
    pcfg = cfg.get("perturbations", {})
    # OmegaConf DictConfig is not isinstance(dict), so convert to plain dict
    # before passing to PerturbationRegistry.create_pipeline which checks
    # isinstance(sub_config, dict).
    if hasattr(pcfg, "items") and not isinstance(pcfg, dict):
        pcfg = OmegaConf.to_container(pcfg, resolve=True)
    return PerturbationRegistry.create_pipeline(pcfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate perturbed trajectories")
    parser.add_argument("--config", type=str, required=True,
                        help="Noisy config YAML (e.g. configs/gaia/noisy.yaml)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with clean trajectories")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for perturbed trajectories")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Perturbation seeds (default: 1 2 3)")
    parser.add_argument("--include-clean", action="store_true", default=False,
                        help="Also copy clean episodes to the output file")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")

    # ── Load clean episodes ──────────────────────────────────────
    input_store = TrajectoryStore(args.input)
    episodes = input_store.load_episodes()
    logger.info(f"Loaded {len(episodes)} clean episodes from {args.input}")

    if not episodes:
        logger.error("No episodes found — check --input path")
        sys.exit(1)

    # ── Build perturbation pipeline ──────────────────────────────
    pipeline = build_pipeline(cfg)
    pert_names = [type(p).__name__ for p in pipeline.perturbations]
    logger.info(f"Perturbation pipeline ({len(pert_names)} operators): {pert_names}")

    if not pipeline.perturbations:
        logger.warning("No perturbations enabled in config — nothing to do")
        sys.exit(0)

    # ── Prepare output store ─────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_store = TrajectoryStore(output_path)
    log_path = output_path.parent / "perturbation_log.jsonl"
    jsonl_log = JSONLLogger(log_path)

    # ── Optionally copy clean episodes ───────────────────────────
    clean_copied = 0
    if args.include_clean:
        logger.info("Copying clean episodes to output...")
        output_store.save_episodes(episodes)
        clean_copied = len(episodes)
        logger.info(f"  ✓ {clean_copied} clean episodes written")

    # ── Apply perturbations ──────────────────────────────────────
    # Only perturb successful episodes; failures stay as-is (clean only).
    success_episodes = [ep for ep in episodes if ep.success]
    failure_episodes = [ep for ep in episodes if not ep.success]
    logger.info(
        f"Success: {len(success_episodes)}, Failure: {len(failure_episodes)} "
        f"— perturbing only successes"
    )

    total_perturbed = 0
    t0 = time.time()

    for seed in args.seeds:
        logger.info(f"=== Perturbation seed {seed} ===")
        seed_count = 0

        for i, episode in enumerate(success_episodes):
            perturbed = pipeline.perturb_episode(episode, seed=seed)
            # Give the perturbed episode a unique ID
            perturbed.episode_id = f"{episode.episode_id}_perturbed_seed{seed}"
            # Perturbed trajectories keep original success label
            output_store.save_episode(perturbed)
            seed_count += 1
            total_perturbed += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  [{i+1}/{len(success_episodes)}] episodes perturbed")

        logger.info(f"  Seed {seed}: {seed_count} perturbed episodes")

    elapsed = time.time() - t0
    logger.info(
        f"Done — {total_perturbed} perturbed episodes "
        f"({clean_copied} clean copied) in {elapsed:.1f}s"
    )
    logger.info(f"Output: {output_path}")

    jsonl_log.log("summary", {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "input_episodes": len(episodes),
        "seeds": args.seeds,
        "perturbations": pert_names,
        "total_perturbed": total_perturbed,
        "clean_copied": clean_copied,
        "elapsed_seconds": round(elapsed, 2),
    })


if __name__ == "__main__":
    main()
