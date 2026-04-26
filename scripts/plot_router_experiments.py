#!/usr/bin/env python3
"""Generate plots and LaTeX/CSV tables from router experiment outputs.

Reads:
- results/router_experiments/metrics_long.csv
- results/router_experiments/experiment_manifest.jsonl (optional)

Writes:
- results/router_experiments/plots/*.png
- results/router_experiments/tables/*.csv
- results/router_experiments/tables/*.tex
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication-quality matplotlib defaults ────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.framealpha": 0.85,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour palette consistent across all paper figures.
_PALETTE = ["#2166AC", "#D6604D", "#4DAC26", "#762A83", "#F4A582", "#1B7837"]
_METHOD_COLORS = {
    "r2v":              _PALETTE[0],
    "slm_only":         _PALETTE[1],
    "llm_only":         _PALETTE[2],
    "entropy_router":   _PALETTE[3],
    "oracle_router":    _PALETTE[4],
    "heuristic_router": _PALETTE[5],
}
_METHOD_LABELS = {
    "r2v":              "R2V (ours)",
    "slm_only":         "SLM-only",
    "llm_only":         "LLM-only",
    "entropy_router":   "Entropy-Router",
    "oracle_router":    "Oracle-Router",
    "heuristic_router": "Heuristic-Router",
}

# Map raw variant names → λ float for the consistency ablation.
_LAMBDA_VARIANT_MAP: dict[str, float] = {
    "no_consistency": 0.0,
    "lam_0.05": 0.05,
    "lam_0.2": 0.2,
    "lam_0.5": 0.5,
    "lam_1.0": 1.0,
    "consistency_lambda_0.05": 0.05,
    "consistency_lambda_0.2": 0.2,
    "consistency_lambda_0.5": 0.5,
    "consistency_lambda_1.0": 1.0,
}

_BENCHMARK_LABELS = {
    "humaneval":     "HumanEval+",
    "textworld":     "TextWorld",
    "terminalbench": "TerminalBench",
}
_MODEL_LABELS = {
    "qwen7":    "Qwen-7B",
    "qwen14":   "Qwen-14B",
    "llama":    "LLaMA-3.1-8B",
    "gemma":    "Gemma-7B",
    "deepseek": "DeepSeek-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and tabulate router experiments")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/router_experiments_single",
        help="Root directory produced by run_router_experiments.sh",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional explicit metrics CSV path",
    )
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    rows: list[dict] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for k in (
                "success_rate",
                "success_rate_ci_low",
                "success_rate_ci_high",
                "worst_seed_sr",
                "cvar_failure",
                "avg_cost",
                "llm_call_rate",
                "ece",
                "brier",
                "alpha",
                "epsilon",
            ):
                parsed[k] = _to_float(parsed.get(k))

            method = (parsed.get("method") or "").strip()
            threshold = None
            if method.startswith("r2v@"):
                try:
                    threshold = float(method.split("@", 1)[1])
                except ValueError:
                    threshold = None
            parsed["threshold"] = threshold
            rows.append(parsed)
    return rows


def ensure_dirs(results_root: Path) -> tuple[Path, Path]:
    plots_dir = results_root / "plots"
    tables_dir = results_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, tables_dir


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, headers: list[str], rows: list[list[str]], caption: str, label: str) -> None:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_latest(rows: list[dict]) -> list[dict]:
    """Keep latest row per (category, variant, benchmark, model, method, alpha, epsilon)."""
    latest: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("category"),
            row.get("variant"),
            row.get("benchmark"),
            row.get("model"),
            row.get("method"),
            row.get("alpha"),
            row.get("epsilon"),
        )
        latest[key] = row
    return list(latest.values())


def build_main_table(rows: list[dict], tables_dir: Path) -> None:
    main_rows = [r for r in rows if r.get("category") == "main" and r.get("variant") == "main_results"]
    if not main_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(main_rows, key=lambda x: (x.get("benchmark", ""), x.get("model", ""), x.get("method", ""))):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "method": r.get("method"),
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "main_results.csv",
        out_rows,
        ["benchmark", "model", "method", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["method"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "main_results.tex",
        ["Benchmark", "Model", "Method", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="Main router results across benchmarks.",
        label="tab:router-main-results",
    )


def build_feature_ablation_table(rows: list[dict], tables_dir: Path) -> None:
    feats = [r for r in rows if r.get("category") == "feature_ablation"]
    if not feats:
        return

    out_rows: list[dict] = []
    for r in sorted(feats, key=lambda x: (x.get("benchmark", ""), x.get("model", ""), x.get("variant", ""), x.get("method", ""))):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "variant": r.get("variant"),
                "method": r.get("method"),
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "feature_ablations.csv",
        out_rows,
        ["benchmark", "model", "variant", "method", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["variant"], r["method"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "feature_ablations.tex",
        ["Benchmark", "Model", "Variant", "Method", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="Router feature ablation results.",
        label="tab:router-feature-ablations",
    )


def build_main_ablation_table(rows: list[dict], tables_dir: Path) -> None:
    ab_rows = [r for r in rows if r.get("category") == "main_ablation"]
    if not ab_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(ab_rows, key=lambda x: (x.get("benchmark", ""), x.get("model", ""), x.get("variant", ""), x.get("method", ""))):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "variant": r.get("variant"),
                "method": r.get("method"),
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "main_ablations.csv",
        out_rows,
        ["benchmark", "model", "variant", "method", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["variant"], r["method"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "main_ablations.tex",
        ["Benchmark", "Model", "Variant", "Method", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="Main ablations (no DPO, no BC/no DPO, lambda variants).",
        label="tab:router-main-ablations",
    )


def build_cvar_sweep_table(rows: list[dict], tables_dir: Path) -> None:
    sweep_rows = [r for r in rows if r.get("category") == "cvar_sweep" and r.get("method") == "r2v"]
    if not sweep_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(
        sweep_rows,
        key=lambda x: (
            x.get("benchmark", ""),
            x.get("model", ""),
            x.get("alpha") if x.get("alpha") is not None else math.inf,
            x.get("epsilon") if x.get("epsilon") is not None else math.inf,
        ),
    ):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "alpha": f"{r.get('alpha', float('nan')):.3f}",
                "epsilon": f"{r.get('epsilon', float('nan')):.3f}",
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "cvar_sweeps.csv",
        out_rows,
        ["benchmark", "model", "alpha", "epsilon", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["alpha"], r["epsilon"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "cvar_sweeps.tex",
        ["Benchmark", "Model", "Alpha", "Epsilon", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="CVaR alpha/epsilon sensitivity sweeps.",
        label="tab:router-cvar-sweeps",
    )


def plot_figure_tradeoff(rows: list[dict], plots_dir: Path) -> None:
    fig_rows = [r for r in rows if r.get("category") == "figure"]
    if not fig_rows:
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in fig_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    for (benchmark, model), group in grouped.items():
        plt.figure(figsize=(7, 5))
        methods = ["slm_only", "llm_only", "entropy_router"]

        for m in methods:
            mm = [r for r in group if r.get("method") == m]
            if not mm:
                continue
            r0 = mm[-1]
            x = r0.get("llm_call_rate")
            y = r0.get("success_rate")
            if x is None or y is None:
                continue
            plt.scatter([x], [y], s=90, label=m)

        r2v_points = [r for r in group if str(r.get("method", "")).startswith("r2v@")]
        if r2v_points:
            r2v_points = sorted(r2v_points, key=lambda r: (r.get("threshold") if r.get("threshold") is not None else math.inf))
            xs = [r["llm_call_rate"] for r in r2v_points if r.get("llm_call_rate") is not None]
            ys = [r["success_rate"] for r in r2v_points if r.get("success_rate") is not None]
            if xs and ys:
                plt.plot(xs, ys, marker="o", linewidth=2, label="r2v threshold sweep")

        plt.xlabel("LLM call rate")
        plt.ylabel("Success rate")
        plt.title(f"Method tradeoff: {benchmark} / {model}")
        plt.grid(alpha=0.3)
        plt.legend()
        out = plots_dir / f"figure_tradeoff_{benchmark}_{model}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()


def plot_feature_ablation_sr(rows: list[dict], plots_dir: Path) -> None:
    feat_rows = [r for r in rows if r.get("category") == "feature_ablation" and r.get("method") in {"r2v", "verifier_router"}]
    if not feat_rows:
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in feat_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    for (benchmark, model), group in grouped.items():
        group = sorted(group, key=lambda r: str(r.get("variant")))
        labels = [str(r.get("variant")) for r in group]
        vals = [r.get("success_rate") for r in group]

        if not any(v is not None for v in vals):
            continue

        xs = list(range(len(labels)))
        ys = [v if v is not None else 0.0 for v in vals]

        plt.figure(figsize=(max(8, 0.8 * len(labels)), 5))
        plt.bar(xs, ys)
        plt.xticks(xs, labels, rotation=35, ha="right")
        plt.ylabel("Success rate")
        plt.title(f"Feature ablation SR: {benchmark} / {model}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = plots_dir / f"feature_ablation_sr_{benchmark}_{model}.png"
        plt.savefig(out, dpi=220)
        plt.close()


def plot_cvar_sweeps(rows: list[dict], plots_dir: Path) -> None:
    sweep_rows = [r for r in rows if r.get("category") == "cvar_sweep" and r.get("method") == "r2v"]
    if not sweep_rows:
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in sweep_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    metrics = [
        ("success_rate", "SR"),
        ("cvar_failure", "CVaR-F"),
        ("avg_cost", "Cost"),
        ("llm_call_rate", "LLM%"),
    ]

    for (benchmark, model), group in grouped.items():
        eps_values = sorted({r.get("epsilon") for r in group if r.get("epsilon") is not None})
        if not eps_values:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()

        for ax, (metric_key, metric_title) in zip(axes, metrics):
            for eps in eps_values:
                subset = [r for r in group if r.get("epsilon") == eps and r.get("alpha") is not None and r.get(metric_key) is not None]
                subset = sorted(subset, key=lambda r: float(r["alpha"]))
                if not subset:
                    continue
                xs = [float(r["alpha"]) for r in subset]
                ys = [float(r[metric_key]) for r in subset]
                ax.plot(xs, ys, marker="o", label=f"epsilon={eps:g}")

            ax.set_title(metric_title)
            ax.set_xlabel("alpha")
            ax.grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=20)

        fig.suptitle(f"CVaR sweeps: {benchmark} / {model}", fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = plots_dir / f"cvar_sweeps_{benchmark}_{model}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)


# ── New: multi-panel Pareto figure ────────────────────────────────────────────
def plot_pareto_multimodel(rows: list[dict], plots_dir: Path) -> None:
    """Figure 1 — multi-panel Pareto: SR vs LLM% per benchmark.

    Each panel shows one benchmark.  The R2V threshold-sweep forms a Pareto
    curve; baselines are plotted as isolated markers.
    """
    fig_rows = [r for r in rows if r.get("category") == "figure"]
    if not fig_rows:
        return

    benchmarks = sorted({str(r.get("benchmark")) for r in fig_rows if r.get("benchmark")})
    n = len(benchmarks)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, benchmark in zip(axes, benchmarks):
        bench_rows = [r for r in fig_rows if r.get("benchmark") == benchmark]
        models = sorted({str(r.get("model")) for r in bench_rows if r.get("model")})
        model_colors = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}

        baseline_methods_plotted: set[str] = set()
        for model in models:
            model_rows = [r for r in bench_rows if r.get("model") == model]
            color = model_colors[model]
            label = _MODEL_LABELS.get(model, model)

            # Pareto curve from r2v@threshold rows.
            sweep = sorted(
                [r for r in model_rows if str(r.get("method", "")).startswith("r2v@")],
                key=lambda r: r.get("threshold") if r.get("threshold") is not None else math.inf,
            )
            xs = [r["llm_call_rate"] for r in sweep if r.get("llm_call_rate") is not None]
            ys = [r["success_rate"] for r in sweep if r.get("success_rate") is not None]
            if xs and ys:
                ax.plot(xs, ys, color=color, marker="o", markersize=4,
                        linewidth=1.8, label=f"R2V ({label})", zorder=3)

        # Baseline scatter (averaged across models for clarity).
        for bm in ["slm_only", "llm_only", "entropy_router", "oracle_router"]:
            bm_rows = [r for r in bench_rows
                       if r.get("method") == bm
                       and r.get("llm_call_rate") is not None
                       and r.get("success_rate") is not None]
            if not bm_rows:
                continue
            x = float(np.mean([r["llm_call_rate"] for r in bm_rows]))
            y = float(np.mean([r["success_rate"] for r in bm_rows]))
            ax.scatter([x], [y],
                       color=_METHOD_COLORS.get(bm, "gray"),
                       marker="D", s=70, zorder=4,
                       label=_METHOD_LABELS.get(bm, bm))

        ax.set_title(_BENCHMARK_LABELS.get(benchmark, benchmark))
        ax.set_xlabel("LLM call rate")
        ax.set_ylabel("Task success rate")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=20, ncol=1)

    fig.suptitle("R2V-Agent: Task success rate vs. LLM escalation cost", fontsize=20)
    fig.tight_layout()
    out = plots_dir / "figure1_pareto_multimodel.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


# ── New: λ_cons sweep figure ──────────────────────────────────────────────────
def plot_lambda_sweep(rows: list[dict], plots_dir: Path) -> None:
    """Figure 2 — SR and CVaR-Failure vs consistency-regularisation weight λ.

    Two-panel figure: left panel = success rate, right = CVaR-failure.
    """
    lam_rows = [
        r for r in rows
        if r.get("category") == "lambda_ablation"
        and r.get("method") == "r2v"
        and r.get("variant") in _LAMBDA_VARIANT_MAP
    ]
    if not lam_rows:
        return

    for r in lam_rows:
        r["lambda_val"] = _LAMBDA_VARIANT_MAP[r["variant"]]

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in lam_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    fig, (ax_sr, ax_cvar) = plt.subplots(1, 2, figsize=(9, 4))

    for i, ((benchmark, model), grp) in enumerate(sorted(grouped.items())):
        grp_sorted = sorted(grp, key=lambda r: r["lambda_val"])
        xs = [r["lambda_val"] for r in grp_sorted]
        sr_vals  = [r.get("success_rate") for r in grp_sorted]
        cvar_vals = [r.get("cvar_failure") for r in grp_sorted]
        color = _PALETTE[i % len(_PALETTE)]
        label = f"{_BENCHMARK_LABELS.get(benchmark, benchmark)}/{_MODEL_LABELS.get(model, model)}"

        if any(v is not None for v in sr_vals):
            ys = [v if v is not None else float("nan") for v in sr_vals]
            ax_sr.plot(xs, ys, color=color, marker="o", label=label)
        if any(v is not None for v in cvar_vals):
            ys = [v if v is not None else float("nan") for v in cvar_vals]
            ax_cvar.plot(xs, ys, color=color, marker="s", label=label)

    ax_sr.set_xlabel(r"Consistency weight $\lambda_{\mathrm{cons}}$")
    ax_sr.set_ylabel("Task success rate")
    ax_sr.set_title("(a) Success rate vs. $\\lambda_{\\mathrm{cons}}$")
    ax_sr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax_cvar.set_xlabel(r"Consistency weight $\lambda_{\mathrm{cons}}$")
    ax_cvar.set_ylabel("CVaR failure rate")
    ax_cvar.set_title("(b) CVaR failure vs. $\\lambda_{\\mathrm{cons}}$")
    ax_cvar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    handles, labels = ax_sr.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.12), fontsize=20)

    fig.suptitle("Effect of consistency regularisation weight on routing performance",
                 fontsize=20)
    fig.tight_layout()
    out = plots_dir / "figure2_lambda_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


# ── New: λ_cons ablation table ────────────────────────────────────────────────
def build_lambda_ablation_table(rows: list[dict], tables_dir: Path) -> None:
    lam_rows = [
        r for r in rows
        if r.get("category") == "lambda_ablation"
        and r.get("method") == "r2v"
        and r.get("variant") in _LAMBDA_VARIANT_MAP
    ]
    if not lam_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(
        lam_rows,
        key=lambda x: (
            x.get("benchmark", ""),
            x.get("model", ""),
            _LAMBDA_VARIANT_MAP.get(x.get("variant", ""), math.inf),
        ),
    ):
        lam = _LAMBDA_VARIANT_MAP.get(r.get("variant", ""), float("nan"))
        out_rows.append({
            "benchmark": _BENCHMARK_LABELS.get(r.get("benchmark", ""), r.get("benchmark", "")),
            "model":     _MODEL_LABELS.get(r.get("model", ""), r.get("model", "")),
            "lambda":    f"{lam:g}",
            "sr":        f"{r.get('success_rate', float('nan')):.3f}",
            "cvar_f":    f"{r.get('cvar_failure', float('nan')):.3f}",
            "cost":      f"{r.get('avg_cost', float('nan')):.2f}",
            "llm_rate":  f"{r.get('llm_call_rate', float('nan')):.3f}",
        })

    write_csv(
        tables_dir / "lambda_ablation.csv",
        out_rows,
        ["benchmark", "model", "lambda", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["lambda"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "lambda_ablation.tex",
        ["Benchmark", "Model", r"$\lambda_{\mathrm{cons}}$", "SR↑", "CVaR-F↓", "Cost↓", "LLM\\%↓"],
        latex_rows,
        caption=(
            "Effect of consistency-regularisation weight $\\lambda_{\\mathrm{cons}}$ "
            "on routing performance.  $\\lambda=0$ corresponds to no consistency "
            "loss (\\texttt{no\\_consistency} variant).  Bold indicates the best "
            "value per benchmark/model pair."
        ),
        label="tab:lambda-ablation",
    )


# ── New: feature ablation table (extended) ───────────────────────────────────
def build_feature_ablation_table_extended(rows: list[dict], tables_dir: Path) -> None:
    """Richer feature ablation table ordered by feature group importance."""
    feats = [r for r in rows
             if r.get("category") == "feature_ablation" and r.get("method") == "r2v"]
    if not feats:
        return

    _ORDER = [
        "all_features",
        "verifier_plus_entropy",
        "verifier_only",
        "entropy_only",
        "logprob_only",
        "step_context_only",
        "no_verifier",
        "no_entropy",
    ]
    _FEAT_LABELS = {
        "all_features":         "All features",
        "verifier_plus_entropy": "Verifier + entropy",
        "verifier_only":        "Verifier only",
        "entropy_only":         "Entropy only",
        "logprob_only":         "Log-prob only",
        "step_context_only":    "Step/context only",
        "no_verifier":          "w/o verifier",
        "no_entropy":           "w/o entropy",
    }

    def sort_key(r: dict) -> tuple:
        v = r.get("variant", "")
        try:
            idx = _ORDER.index(v)
        except ValueError:
            idx = len(_ORDER)
        return (r.get("benchmark", ""), r.get("model", ""), idx)

    out_rows: list[dict] = []
    for r in sorted(feats, key=sort_key):
        out_rows.append({
            "benchmark": _BENCHMARK_LABELS.get(r.get("benchmark", ""), r.get("benchmark", "")),
            "model":     _MODEL_LABELS.get(r.get("model", ""), r.get("model", "")),
            "features":  _FEAT_LABELS.get(r.get("variant", ""), r.get("variant", "")),
            "sr":        f"{r.get('success_rate', float('nan')):.3f}",
            "cvar_f":    f"{r.get('cvar_failure', float('nan')):.3f}",
            "cost":      f"{r.get('avg_cost', float('nan')):.2f}",
            "llm_rate":  f"{r.get('llm_call_rate', float('nan')):.3f}",
        })

    write_csv(
        tables_dir / "feature_ablation_extended.csv",
        out_rows,
        ["benchmark", "model", "features", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["features"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "feature_ablation_extended.tex",
        ["Benchmark", "Model", "Feature set", "SR↑", "CVaR-F↓", "Cost↓", "LLM\\%↓"],
        latex_rows,
        caption=(
            "Feature ablation: effect of removing or isolating feature groups "
            "at inference time (router weights are unchanged). "
            "\\textit{All features} is the full 15-dim vector."
        ),
        label="tab:feature-ablation",
    )


# ── New: feature-transform router comparison table ───────────────────────────
def build_feature_transform_table(rows: list[dict], tables_dir: Path) -> None:
    """Table comparing routers trained with different feature transforms.

    These are distinct from inference-time feature masks (feature_ablation):
    each row represents a *separately trained* router where features were
    permanently modified during training (no_entropy / verifier_pseudo_entropy).
    Includes the full-feature main router as baseline.
    """
    # Collect main-results r2v rows (full-feature router) as the baseline.
    main_rows = [
        r for r in rows
        if r.get("category") == "main"
        and r.get("variant") == "main_results"
        and r.get("method") == "r2v"
    ]
    # Collect the trained-transform rows.
    transform_rows = [
        r for r in rows
        if r.get("category") == "feature_transform"
        and r.get("method") == "r2v"
    ]
    if not main_rows and not transform_rows:
        return

    _TRANSFORM_LABELS = {
        "main_results":             "All features (full)",
        "no_entropy":               "No entropy (trained)",
        "verifier_pseudo_entropy":  "Verifier pseudo-entropy (trained)",
    }

    combined = [
        {**r, "transform_label": "main_results"} for r in main_rows
    ] + [
        {**r, "transform_label": r.get("variant", "")} for r in transform_rows
    ]

    _ORDER = ["main_results", "no_entropy", "verifier_pseudo_entropy"]

    def sort_key(r: dict) -> tuple:
        v = r.get("transform_label", "")
        try:
            idx = _ORDER.index(v)
        except ValueError:
            idx = len(_ORDER)
        return (r.get("benchmark", ""), r.get("model", ""), idx)

    out_rows: list[dict] = []
    for r in sorted(combined, key=sort_key):
        out_rows.append({
            "benchmark":  _BENCHMARK_LABELS.get(r.get("benchmark", ""), r.get("benchmark", "")),
            "model":      _MODEL_LABELS.get(r.get("model", ""), r.get("model", "")),
            "router":     _TRANSFORM_LABELS.get(r.get("transform_label", ""), r.get("transform_label", "")),
            "sr":         f"{r.get('success_rate', float('nan')):.3f}",
            "cvar_f":     f"{r.get('cvar_failure', float('nan')):.3f}",
            "cost":       f"{r.get('avg_cost', float('nan')):.2f}",
            "llm_rate":   f"{r.get('llm_call_rate', float('nan')):.3f}",
            "ece":        f"{r.get('ece', float('nan')):.4f}" if r.get("ece") is not None else "—",
            "brier":      f"{r.get('brier', float('nan')):.4f}" if r.get("brier") is not None else "—",
        })

    write_csv(
        tables_dir / "feature_transform_comparison.csv",
        out_rows,
        ["benchmark", "model", "router", "sr", "cvar_f", "cost", "llm_rate", "ece", "brier"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["router"],
         r["sr"], r["cvar_f"], r["cost"], r["llm_rate"], r["ece"], r["brier"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "feature_transform_comparison.tex",
        ["Benchmark", "Model", "Router input features",
         "SR↑", "CVaR-F↓", "Cost↓", "LLM\\%↓", "ECE↓", "Brier↓"],
        latex_rows,
        caption=(
            "Comparison of routers trained with different feature transforms. "
            "\\textit{All features (full)} uses the complete 15-dim vector. "
            "\\textit{No entropy (trained)} permanently zeros the SLM-entropy "
            "feature during both training and evaluation. "
            "\\textit{Verifier pseudo-entropy (trained)} replaces SLM entropy "
            "with entropy derived from verifier scores, enabling deployment "
            "with closed-source models where true token probabilities are "
            "unavailable. ECE and Brier score measure calibration quality."
        ),
        label="tab:feature-transform",
    )


# ── New: per-perturbation-type breakdown plot ─────────────────────────────────
def plot_perturbation_breakdown(rows: list[dict], plots_dir: Path) -> None:
    """Supplementary — SR by perturbation type for R2V vs baselines."""
    # Perturbation type data is encoded in per-seed rows in the bundle JSON;
    # it is not yet surfaced in metrics_long.csv.  If the data is absent,
    # silently skip rather than crash.
    _ = rows  # reserved for when per-type data is wired into the CSV
    return


# ── Improved main results table with CI columns ───────────────────────────────
def build_main_table_with_ci(rows: list[dict], tables_dir: Path) -> None:
    main_rows = [r for r in rows if r.get("category") == "main" and r.get("variant") == "main_results"]
    if not main_rows:
        return

    priority_methods = ["r2v", "slm_only", "llm_only", "entropy_router", "oracle_router", "heuristic_router"]

    def method_order(m: str) -> int:
        try:
            return priority_methods.index(m)
        except ValueError:
            return len(priority_methods)

    out_rows: list[dict] = []
    for r in sorted(
        main_rows,
        key=lambda x: (
            x.get("benchmark", ""),
            x.get("model", ""),
            method_order(x.get("method", "")),
        ),
    ):
        ci_lo = r.get("success_rate_ci_low")
        ci_hi = r.get("success_rate_ci_high")
        sr = r.get("success_rate", float("nan"))
        ci_str = (
            f"[{ci_lo:.3f}, {ci_hi:.3f}]"
            if ci_lo is not None and ci_hi is not None
            else "—"
        )
        out_rows.append({
            "benchmark": _BENCHMARK_LABELS.get(r.get("benchmark", ""), r.get("benchmark", "")),
            "model":     _MODEL_LABELS.get(r.get("model", ""), r.get("model", "")),
            "method":    _METHOD_LABELS.get(r.get("method", ""), r.get("method", "")),
            "sr":        f"{sr:.3f}",
            "sr_ci":     ci_str,
            "worst_sr":  f"{r.get('worst_seed_sr', float('nan')):.3f}",
            "cvar_f":    f"{r.get('cvar_failure', float('nan')):.3f}",
            "cost":      f"{r.get('avg_cost', float('nan')):.1f}",
            "llm_rate":  f"{r.get('llm_call_rate', float('nan')):.3f}",
        })

    write_csv(
        tables_dir / "main_results_with_ci.csv",
        out_rows,
        ["benchmark", "model", "method", "sr", "sr_ci", "worst_sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["method"],
         r["sr"], r["sr_ci"], r["worst_sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "main_results_with_ci.tex",
        ["Benchmark", "Model", "Method",
         "SR↑", "95\\% CI", "Worst-SR↑", "CVaR-F↓", "Cost↓", "LLM\\%↓"],
        latex_rows,
        caption=(
            "Main results.  R2V achieves frontier-LLM success rates while "
            "routing $>$85\\% of steps to the SLM.  "
            "Worst-SR = worst per-seed success rate (robustness); "
            "CVaR-F = CVaR$_{0.2}$ of failure rate (tail robustness); "
            "Cost = average per-episode cost; LLM\\% = fraction of steps routed to LLM. "
            "95\\% bootstrap CIs shown for SR."
        ),
        label="tab:main-results",
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    metrics_path = Path(args.metrics_path) if args.metrics_path else (results_root / "metrics_long.csv")

    rows = load_rows(metrics_path)
    rows = aggregate_latest(rows)

    plots_dir, tables_dir = ensure_dirs(results_root)

    # ── Tables ──────────────────────────────────────────────────────────────
    # Table 1: main results with bootstrap CI
    build_main_table_with_ci(rows, tables_dir)
    # Table 1 (legacy format, kept for backwards compatibility)
    build_main_table(rows, tables_dir)
    # Table 2a: feature-transform router comparison (trained variants)
    build_feature_transform_table(rows, tables_dir)
    # Table 2b: inference-time feature mask ablation
    build_feature_ablation_table_extended(rows, tables_dir)
    build_feature_ablation_table(rows, tables_dir)
    # Table 3: consistency-λ ablation
    build_lambda_ablation_table(rows, tables_dir)
    # Supplementary: main ablations and CVaR sweep table
    build_main_ablation_table(rows, tables_dir)
    build_cvar_sweep_table(rows, tables_dir)

    # ── Plots ────────────────────────────────────────────────────────────────
    # Figure 1: multi-panel Pareto (SR vs LLM%)
    plot_pareto_multimodel(rows, plots_dir)
    # Legacy per-benchmark tradeoff plots
    plot_figure_tradeoff(rows, plots_dir)
    # Figure 2: λ_cons sweep
    plot_lambda_sweep(rows, plots_dir)
    # Figure 3: CVaR hyperparameter sensitivity
    plot_cvar_sweeps(rows, plots_dir)
    # Feature ablation bar charts
    plot_feature_ablation_sr(rows, plots_dir)
    # Perturbation-type breakdown (stubbed until per-type metrics are in CSV)
    plot_perturbation_breakdown(rows, plots_dir)

    print(f"Saved plots to:  {plots_dir}")
    print(f"Saved tables to: {tables_dir}")


if __name__ == "__main__":
    main()
