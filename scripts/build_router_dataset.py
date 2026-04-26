#!/usr/bin/env python3
"""Build a unified router dataset (Parquet) from router_features_data JSONL files.

This script consolidates main-result and ablation feature files into one table,
and assigns train/val/test splits using trajectory data in data/trajectories.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required for Parquet export. Install it in the existing .venv."
    ) from exc

from r2v.data.splits import load_and_split
from r2v.data.trajectory import TrajectoryStore


KNOWN_MODELS = ("qwen7", "qwen14", "llama", "gemma", "deepseek")
ROW_PROGRESS_INTERVAL = 100000


def _p(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified router dataset (Parquet)")
    parser.add_argument(
        "--features-root",
        type=str,
        default="router_features_data",
        help="Directory containing *_noisy_router_features_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/router_dataset/unified_router_features.parquet",
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic task-level split reconstruction",
    )
    parser.add_argument(
        "--max-perturbations-per-task",
        type=int,
        default=2,
        help="Match split setting used in training scripts",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        help="Parquet compression codec (e.g., snappy, zstd, gzip)",
    )
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        default=False,
        help="Allow rows with no split mapping; assigns split='unmapped'",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Rows per in-memory chunk before writing to Parquet",
    )
    return parser.parse_args()


def _parse_feature_filename(path: Path) -> tuple[str, str, str] | None:
    name = path.name

    # Main format: {benchmark}_noisy_router_features_{variant}_{model}.jsonl
    m = re.match(
        rf"^(?P<benchmark>[^_]+)_noisy_router_features_(?P<variant>.+)_(?P<model>{'|'.join(KNOWN_MODELS)})\.jsonl$",
        name,
    )
    if not m:
        # Short ablation format: {benchmark}_{model}_{variant}.jsonl
        m = re.match(
            rf"^(?P<benchmark>[^_]+)_(?P<model>{'|'.join(KNOWN_MODELS)})_(?P<variant>.+)\.jsonl$",
            name,
        )
    if not m:
        # Ablation suffix format:
        # {benchmark}_noisy_router_features_heuristic_{model}_abl_{variant}.jsonl
        m = re.match(
            rf"^(?P<benchmark>[^_]+)_noisy_router_features_heuristic_(?P<model>{'|'.join(KNOWN_MODELS)})_abl_(?P<variant>.+)\.jsonl$",
            name,
        )
        if m:
            return m.group("benchmark"), m.group("variant"), m.group("model")

    if not m:
        _p(f"[PARSE][SKIP] Could not parse filename pattern: {path}")
        return None

    benchmark = m.group("benchmark")
    variant = m.group("variant")
    model = m.group("model")

    # Normalize common main-case naming.
    if variant == "heuristic":
        pass
    elif variant.startswith("heuristic_"):
        # Files like heuristic_qwen7_... should not happen here, but normalize if they do.
        variant = variant.replace("heuristic_", "", 1)

    _p(
        "[PARSE][OK] "
        f"file={path.name} -> benchmark={benchmark}, model={model}, variant={variant}"
    )

    return benchmark, variant, model


def _collect_episode_ids(path: Path) -> set[str]:
    _p(f"[SPLIT][LOAD] Reading episode ids from: {path}")
    store = TrajectoryStore(path)
    ids = {ep.episode_id for ep in store.iter_episodes()}
    _p(f"[SPLIT][LOAD] Loaded {len(ids)} episode ids from: {path}")
    return ids


def _build_split_map(
    benchmark: str,
    seed: int,
    max_perturbations_per_task: int,
) -> tuple[dict[str, str], str | None]:
    _p(
        "[SPLIT][START] "
        f"benchmark={benchmark}, seed={seed}, max_perturbations_per_task={max_perturbations_per_task}"
    )
    # 1) Preferred: reconstruct task-level split from a single noisy trajectory file.
    combined_candidates = [
        Path(f"data/trajectories/{benchmark}_noisy/trajectories.jsonl"),
        Path(f"data/trajectories/{benchmark}_trajectories.jsonl"),
        Path(f"data/trajectories/{benchmark}_noisy.jsonl"),
    ]
    for traj_path in combined_candidates:
        _p(f"[SPLIT][CHECK] Combined candidate: {traj_path}")
        if traj_path.exists():
            _p(f"[SPLIT][USE] Combined trajectory source found: {traj_path}")
            splits = load_and_split(
                str(traj_path),
                # Keep all perturbation variants to guarantee full episode-id coverage.
                max_perturbations_per_task=None,
                seed=seed,
            )
            mapping: dict[str, str] = {}
            for split_name in ("train", "val", "test"):
                for ep in splits[split_name]:
                    mapping[ep.episode_id] = split_name
            _p(
                "[SPLIT][DONE] Combined split map built: "
                f"benchmark={benchmark}, total_episode_ids={len(mapping)}, "
                f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
            )
            return mapping, str(traj_path)

    # 2) Fallback: explicit split files already present in trajectories dir.
    _p(f"[SPLIT][FALLBACK] No combined source found for benchmark={benchmark}; trying explicit split files")
    split_patterns = {
        "train": [
            Path(f"data/trajectories/{benchmark}_train_noisy.jsonl"),
            Path(f"data/trajectories/{benchmark}_train.jsonl"),
            Path(f"data/trajectories/{benchmark}_bc_train.jsonl"),
            Path(f"data/trajectories/{benchmark}_verifier_train.jsonl"),
            Path(f"data/trajectories/{benchmark}_noisy/bc_train.jsonl"),
            Path(f"data/trajectories/{benchmark}_noisy/verifier_train.jsonl"),
        ],
        "val": [
            Path(f"data/trajectories/{benchmark}_val_noisy.jsonl"),
            Path(f"data/trajectories/{benchmark}_val.jsonl"),
            Path(f"data/trajectories/{benchmark}_bc_val.jsonl"),
            Path(f"data/trajectories/{benchmark}_verifier_val.jsonl"),
            Path(f"data/trajectories/{benchmark}_noisy/bc_val.jsonl"),
            Path(f"data/trajectories/{benchmark}_noisy/verifier_val.jsonl"),
        ],
        "test": [
            Path(f"data/trajectories/{benchmark}_test_noisy.jsonl"),
            Path(f"data/trajectories/{benchmark}_test.jsonl"),
            Path(f"data/trajectories/{benchmark}_bc_test.jsonl"),
            Path(f"data/trajectories/{benchmark}_verifier_test.jsonl"),
            Path(f"data/trajectories/{benchmark}_noisy/bc_test.jsonl"),
            Path(f"data/trajectories/{benchmark}_noisy/verifier_test.jsonl"),
        ],
    }

    mapping = {}
    used_files: list[str] = []
    for split_name, candidates in split_patterns.items():
        _p(f"[SPLIT][FALLBACK] Searching {split_name} files for benchmark={benchmark}")
        for c in candidates:
            _p(f"[SPLIT][CANDIDATE] {split_name}: {c}")
        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is None:
            _p(
                "[SPLIT][ERROR] Missing fallback split file: "
                f"benchmark={benchmark}, split={split_name}"
            )
            return {}, None
        _p(f"[SPLIT][USE] benchmark={benchmark}, split={split_name}, file={chosen}")
        used_files.append(str(chosen))
        for eid in _collect_episode_ids(chosen):
            mapping[eid] = split_name
        _p(
            "[SPLIT][COUNT] "
            f"benchmark={benchmark}, split={split_name}, cumulative_ids={len(mapping)}"
        )
    _p(
        "[SPLIT][DONE] Fallback split map built: "
        f"benchmark={benchmark}, total_episode_ids={len(mapping)}"
    )
    return mapping, ",".join(used_files)


def _category_from_variant(variant: str) -> str:
    return "main" if variant == "heuristic" else "ablation"


def _empty_chunk() -> dict[str, list]:
    return {
        "benchmark": [],
        "model": [],
        "variant": [],
        "category": [],
        "source_file": [],
        "episode_id": [],
        "step_idx": [],
        "split": [],
        "split_source": [],
        "slm_success": [],
        "cost": [],
        "perturbation_seed": [],
        "verifier_scores": [],
        "features": [],
    }


def _append_row(chunk: dict[str, list], row: dict) -> None:
    for k in chunk:
        chunk[k].append(row[k])


def _chunk_to_table(chunk: dict[str, list]) -> pa.Table:
    schema = pa.schema(
        [
            ("benchmark", pa.string()),
            ("model", pa.string()),
            ("variant", pa.string()),
            ("category", pa.string()),
            ("source_file", pa.string()),
            ("episode_id", pa.string()),
            ("step_idx", pa.int32()),
            ("split", pa.string()),
            ("split_source", pa.string()),
            ("slm_success", pa.float32()),
            ("cost", pa.float32()),
            ("perturbation_seed", pa.int32()),
            ("verifier_scores", pa.list_(pa.float32())),
            ("features", pa.list_(pa.float32())),
        ]
    )
    return pa.Table.from_pydict(chunk, schema=schema)


def main() -> None:
    args = parse_args()

    _p("[START] build_router_dataset.py starting")
    _p(f"[ARGS] features_root={args.features_root}")
    _p(f"[ARGS] output={args.output}")
    _p(f"[ARGS] seed={args.seed}")
    _p(f"[ARGS] max_perturbations_per_task={args.max_perturbations_per_task}")
    _p(f"[ARGS] compression={args.compression}")
    _p(f"[ARGS] allow_unmapped={args.allow_unmapped}")
    _p(f"[ARGS] chunk_size={args.chunk_size}")

    features_root = Path(args.features_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _p(f"[PATH] Resolved features_root={features_root.resolve()}")
    _p(f"[PATH] Resolved output_path={output_path.resolve()}")

    feature_files = sorted(features_root.rglob("*.jsonl"))
    _p(f"[DISCOVERY] Found {len(feature_files)} jsonl files under features_root")
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {features_root}")

    for idx, fp in enumerate(feature_files, start=1):
        _p(f"[DISCOVERY][FILE {idx}/{len(feature_files)}] {fp}")

    split_maps: dict[str, dict[str, str]] = {}
    split_sources: dict[str, str] = {}
    skipped_unparsed = 0
    benchmark_file_counts: dict[str, int] = defaultdict(int)
    for f in feature_files:
        parsed = _parse_feature_filename(f)
        if parsed is None:
            skipped_unparsed += 1
            continue
        benchmark, _, _ = parsed
        benchmark_file_counts[benchmark] += 1
        if benchmark not in split_maps:
            m, source = _build_split_map(
                benchmark,
                seed=args.seed,
                max_perturbations_per_task=args.max_perturbations_per_task,
            )
            split_maps[benchmark] = m
            split_sources[benchmark] = source or "missing"
            _p(
                "[SPLIT][SUMMARY] "
                f"benchmark={benchmark}, split_source={split_sources[benchmark]}, mapped_episode_ids={len(m)}"
            )

    _p(f"[DISCOVERY][SUMMARY] Parsed benchmarks: {sorted(benchmark_file_counts.keys())}")
    for b in sorted(benchmark_file_counts.keys()):
        _p(f"[DISCOVERY][SUMMARY] benchmark={b}, parsed_files={benchmark_file_counts[b]}")
    _p(f"[DISCOVERY][SUMMARY] skipped_unparsed_files={skipped_unparsed}")

    writer: pq.ParquetWriter | None = None
    chunk = _empty_chunk()

    rows_written = 0
    unmapped_rows = 0
    counts_by_split = defaultdict(int)
    counts_by_group = defaultdict(int)
    rows_by_file: dict[str, int] = defaultdict(int)
    unmapped_by_file: dict[str, int] = defaultdict(int)
    file_count = len(feature_files)
    parsed_file_count = 0

    for file_idx, src in enumerate(feature_files, start=1):
        _p(f"[PROCESS][FILE {file_idx}/{file_count}] Start: {src}")
        parsed = _parse_feature_filename(src)
        if parsed is None:
            _p(f"[PROCESS][SKIP] Unparsed filename, skipping file: {src}")
            continue
        benchmark, variant, model = parsed
        parsed_file_count += 1
        category = _category_from_variant(variant)
        split_map = split_maps.get(benchmark, {})
        split_source = split_sources.get(benchmark, "missing")
        _p(
            "[PROCESS][META] "
            f"file={src.name}, benchmark={benchmark}, model={model}, variant={variant}, "
            f"category={category}, split_source={split_source}, mapped_ids={len(split_map)}"
        )

        file_rows = 0
        file_unmapped = 0

        with open(src, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                episode_id = str(rec["episode_id"])
                split = split_map.get(episode_id)
                if split is None:
                    unmapped_rows += 1
                    file_unmapped += 1
                    unmapped_by_file[str(src)] += 1
                    if args.allow_unmapped:
                        split = "unmapped"
                        _p(
                            "[PROCESS][UNMAPPED][ALLOW] "
                            f"file={src.name}, line={line_idx}, episode_id={episode_id}"
                        )
                    else:
                        _p(
                            "[PROCESS][UNMAPPED][ERROR] "
                            f"file={src.name}, line={line_idx}, episode_id={episode_id}"
                        )
                        raise RuntimeError(
                            "Found feature row without split mapping for episode_id="
                            f"{episode_id} in {src}. "
                            "Add trajectory split files or rerun with --allow-unmapped."
                        )

                row = {
                    "benchmark": benchmark,
                    "model": model,
                    "variant": variant,
                    "category": category,
                    "source_file": str(src),
                    "episode_id": episode_id,
                    "step_idx": int(rec.get("step_idx", 0)),
                    "split": split,
                    "split_source": split_source,
                    "slm_success": float(rec.get("slm_success", 0.0)),
                    "cost": float(rec.get("cost", 1.0)),
                    "perturbation_seed": int(rec.get("perturbation_seed", 0)),
                    "verifier_scores": (
                        [float(x) for x in rec.get("verifier_scores", [])]
                        if rec.get("verifier_scores") is not None
                        else None
                    ),
                    "features": [float(x) for x in rec["features"]],
                }

                _append_row(chunk, row)
                rows_written += 1
                file_rows += 1
                rows_by_file[str(src)] += 1
                counts_by_split[split] += 1
                counts_by_group[(benchmark, model, variant)] += 1

                if file_rows % ROW_PROGRESS_INTERVAL == 0:
                    _p(
                        "[PROCESS][PROGRESS] "
                        f"file={src.name}, rows_in_file={file_rows}, total_rows_written={rows_written}, "
                        f"current_chunk_size={len(chunk['episode_id'])}"
                    )

                if len(chunk["episode_id"]) >= args.chunk_size:
                    table = _chunk_to_table(chunk)
                    if writer is None:
                        _p(
                            "[WRITE] Initializing ParquetWriter: "
                            f"output={output_path}, compression={args.compression}"
                        )
                        writer = pq.ParquetWriter(
                            output_path,
                            table.schema,
                            compression=args.compression,
                        )
                    _p(
                        "[WRITE] Writing full chunk: "
                        f"chunk_rows={len(chunk['episode_id'])}, total_rows_written={rows_written}"
                    )
                    writer.write_table(table)
                    chunk = _empty_chunk()

        _p(
            "[PROCESS][FILE DONE] "
            f"file={src.name}, rows={file_rows}, unmapped={file_unmapped}, total_rows_written={rows_written}"
        )

    if len(chunk["episode_id"]) > 0:
        _p(
            "[WRITE] Flushing final partial chunk: "
            f"chunk_rows={len(chunk['episode_id'])}, total_rows_written={rows_written}"
        )
        table = _chunk_to_table(chunk)
        if writer is None:
            _p(
                "[WRITE] Initializing ParquetWriter for final chunk: "
                f"output={output_path}, compression={args.compression}"
            )
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression=args.compression,
            )
        writer.write_table(table)

    if writer is not None:
        _p("[WRITE] Closing ParquetWriter")
        writer.close()

    if rows_written == 0:
        raise RuntimeError("No rows were written to the unified dataset.")

    _p(f"[SUMMARY] parsed_file_count={parsed_file_count}")
    _p(f"[SUMMARY] total_rows_written={rows_written}")
    _p(f"[SUMMARY] total_unmapped_rows={unmapped_rows}")
    _p(f"[SUMMARY] counts_by_split={dict(counts_by_split)}")
    _p(f"[SUMMARY] benchmark split sources={split_sources}")

    _p("[SUMMARY] rows written per file:")
    for k in sorted(rows_by_file.keys()):
        _p(f"  - {k}: {rows_by_file[k]}")

    if unmapped_by_file:
        _p("[SUMMARY] unmapped rows per file:")
        for k in sorted(unmapped_by_file.keys()):
            _p(f"  - {k}: {unmapped_by_file[k]}")

    _p("[SUMMARY] rows by (benchmark|model|variant):")
    for (b, m, v), c in sorted(counts_by_group.items()):
        _p(f"  - {b}|{m}|{v}: {c}")

    summary = {
        "output": str(output_path),
        "rows_written": rows_written,
        "unmapped_rows": unmapped_rows,
        "counts_by_split": dict(counts_by_split),
        "counts_by_group": {
            f"{b}|{m}|{v}": c for (b, m, v), c in sorted(counts_by_group.items())
        },
        "split_sources": split_sources,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
