#!/usr/bin/env python3
"""Rank saved router checkpoints by validation metrics.

Reads RouterTrainer checkpoint manifest and prints sorted checkpoint records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank router checkpoints by validation metrics")
    parser.add_argument(
        "--manifest",
        type=str,
        default="outputs/router/unified_single_router/checkpoint_manifest.jsonl",
        help="Path to checkpoint_manifest.jsonl",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_brier",
        choices=["eval_brier", "eval_ece", "eval_accuracy", "train_total_loss"],
        help="Metric used for ranking",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Whether smaller or larger metric is better",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Show top-k checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Keep only checkpoints with selected metric present
    rows = [r for r in rows if r.get(args.metric) is not None]
    if not rows:
        print("No checkpoint rows with metric:", args.metric)
        return

    reverse = args.mode == "max"
    rows.sort(key=lambda r: r[args.metric], reverse=reverse)

    print(f"Ranking by {args.metric} ({args.mode}), total={len(rows)}")
    print("=" * 120)
    print(f"{'rank':>4}  {'epoch':>5}  {'name':<28}  {args.metric:<12}  {'eval_ece':<12}  {'eval_accuracy':<12}  path")
    print("-" * 120)

    for i, row in enumerate(rows[: args.top_k], start=1):
        print(
            f"{i:>4}  {int(row.get('epoch', -1)):>5}  "
            f"{str(row.get('name', '')):<28}  "
            f"{float(row.get(args.metric, float('nan'))):<12.6f}  "
            f"{float(row.get('eval_ece', float('nan'))):<12.6f}  "
            f"{float(row.get('eval_accuracy', float('nan'))):<12.6f}  "
            f"{row.get('path', '')}"
        )


if __name__ == "__main__":
    main()
