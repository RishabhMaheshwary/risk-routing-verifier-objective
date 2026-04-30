#!/usr/bin/env python3
"""
Convert yoonholee/terminalbench-trajectories (HuggingFace) to r2v Episode JSONL.

Usage:
    # Default: Factory Droid + terminus-3-3 scaffolds, reward=1 only
    python scripts/convert_hf_trajectories.py \
        --output data/trajectories/terminalbench_clean/trajectories.jsonl

    # Custom scaffold/model filter
    python scripts/convert_hf_trajectories.py \
        --scaffolds "Factory Droid" "terminus-3-3" \
        --output data/trajectories/terminalbench_clean/trajectories.jsonl \
        --min-reward 1

HF dataset step schema (from yoonholee/terminalbench-trajectories):
    src   : 'user' | 'agent' | 'system'
    msg   : agent reasoning text
    tools : list of {fn, cmd} dicts, or null
    obs   : tool output string (truncated to 5000 chars), or null

Conversion logic:
    - Only agent steps with tool calls become r2v Steps.
    - Observation = obs of the current step (what the tool returned).
    - Action      = run [{cmd}] for bash-like tools, fn [{cmd}] otherwise.
    - Episode reward=1 → success=True.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.trajectory import (
    Action,
    ActionSource,
    Episode,
    EpisodeMetadata,
    Observation,
    PerturbationType,
    Step,
)

# Default scaffolds to include (best pass rates with sufficient data)
DEFAULT_SCAFFOLDS = ["Factory Droid", "terminus-3-3"]

# Tool function names that map to shell commands → use run [cmd] format
_BASH_TOOL_NAMES = {
    "bash", "bash_command", "shell", "terminal",
    "execute", "run", "command", "cmd",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_tools(tools_raw) -> list[dict]:
    """Parse tools field — list, JSON string, or null."""
    if tools_raw is None:
        return []
    if isinstance(tools_raw, list):
        return tools_raw
    if isinstance(tools_raw, str):
        if tools_raw in ("None", "null", ""):
            return []
        try:
            parsed = json.loads(tools_raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            import ast
            parsed = ast.literal_eval(tools_raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def _obs_text(obs_raw) -> str:
    """Return observation text. Null / None → empty string."""
    if obs_raw is None:
        return ""
    s = str(obs_raw)
    if s in ("None", "null"):
        return ""
    return s


def _action_text(tools: list[dict]) -> tuple[str, str]:
    """Return (action_raw_text, action_type) from tools list."""
    if not tools:
        return "", "unknown"
    tool = tools[0]
    fn = str(tool.get("fn", "")).strip()
    cmd = str(tool.get("cmd", "")).strip()
    if fn.lower() in _BASH_TOOL_NAMES:
        return f"run [{cmd}]", "run"
    return f"{fn} [{cmd}]", fn.lower()


# ── conversion ────────────────────────────────────────────────────────────────

def convert_row(row: dict, run_id: str) -> "Episode | None":
    """Convert one HF dataset row to an r2v Episode. Returns None to skip."""
    steps_raw = row.get("steps")
    if not steps_raw:
        return None

    if isinstance(steps_raw, str):
        try:
            hf_steps = json.loads(steps_raw)
        except Exception:
            return None
    elif isinstance(steps_raw, list):
        hf_steps = steps_raw
    else:
        return None

    # json.loads("null") → None, json.loads("[]") → []
    if not hf_steps:
        return None

    r2v_steps: list[Step] = []
    step_idx = 0

    for hf_step in hf_steps:
        if not isinstance(hf_step, dict):
            continue
        if hf_step.get("src") != "agent":
            continue
        tools = _parse_tools(hf_step.get("tools"))
        if not tools:
            continue  # pure reasoning/thinking step

        obs_text = _obs_text(hf_step.get("obs"))
        action_raw, action_type = _action_text(tools)
        if not action_raw:
            continue

        r2v_steps.append(Step(
            step_idx=step_idx,
            observation=Observation(raw_text=obs_text),
            action=Action(raw_text=action_raw, action_type=action_type),
            action_source=ActionSource.TEACHER,
            reward=0.0,
            perturbation_type=PerturbationType.NONE,
        ))
        step_idx += 1

    if not r2v_steps:
        return None

    reward = float(row.get("reward") or 0.0)
    r2v_steps[-1].reward = reward  # assign episode reward to final step

    model = str(row.get("model", ""))
    if "@" in model:
        model_name, provider = model.rsplit("@", 1)
    else:
        model_name, provider = model, ""

    task_id = str(row.get("task_name", "unknown"))
    trial_id = str(row.get("trial_id") or row.get("trial_name") or "")
    agent = str(row.get("agent", ""))

    metadata = EpisodeMetadata(
        task_id=task_id,
        goal=task_id,  # HF dataset doesn't store goal text
        benchmark="terminalbench",
        run_id=run_id,
        teacher_model=model_name,
        teacher_provider=provider,
        extra={"agent": agent, "trial_id": trial_id},
    )

    episode_id = f"{task_id}_{agent}_{trial_id}" if trial_id else f"{task_id}_{agent}"

    return Episode(
        episode_id=episode_id,
        metadata=metadata,
        steps=r2v_steps,
        success=reward >= 1.0,
        partial_score=reward,
        total_cost=(row.get("cost_cents") or 0.0) / 100.0,
        wall_time_seconds=float(row.get("duration_seconds") or 0.0),
        perturbation_type=PerturbationType.NONE,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert HF terminalbench trajectories to r2v JSONL")
    parser.add_argument(
        "--scaffolds", nargs="+", default=DEFAULT_SCAFFOLDS,
        help=f"Scaffold(s) to include (default: {DEFAULT_SCAFFOLDS})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model(s) to filter. Default: all models for the selected scaffolds.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--min-reward", type=float, default=1.0,
        help="Minimum episode reward to include (default: 1.0 = successes only)",
    )
    parser.add_argument(
        "--all-rewards", action="store_true",
        help="Include all episodes regardless of reward",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--max-failures", type=int, default=None,
        help="Cap failure episodes (reward < min-reward) to this count",
    )
    args = parser.parse_args()

    from datasets import load_dataset  # type: ignore
    print("Loading yoonholee/terminalbench-trajectories …")
    ds = load_dataset("yoonholee/terminalbench-trajectories", split="train")
    print(f"  {len(ds)} total rows")

    # Filter by scaffold
    rows = [r for r in ds if r.get("agent") in args.scaffolds]
    print(f"  {len(rows)} rows matching scaffolds={args.scaffolds}")

    # Filter by model if specified
    if args.models:
        rows = [r for r in rows if r.get("model") in args.models]
        print(f"  {len(rows)} rows matching models={args.models}")

    # Filter by reward
    if not args.all_rewards:
        rows = [r for r in rows if (r.get("reward") or 0) >= args.min_reward]
        print(f"  {len(rows)} rows with reward >= {args.min_reward}")

    if args.max_episodes:
        rows = rows[: args.max_episodes]
        print(f"  capped at {len(rows)}")

    # Stats breakdown
    from collections import Counter
    counts = Counter(f"{r['agent']}/{r['model']}" for r in rows)
    for combo, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {combo}: {n}")

    scaffolds_tag = "_".join(s.replace(" ", "-") for s in args.scaffolds)
    run_id = f"terminalbench_hf_{scaffolds_tag}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with out_path.open("w") as f:
        for row in rows:
            ep = convert_row(row, run_id=run_id)
            if ep is None:
                skipped += 1
                continue
            f.write(json.dumps(ep.to_dict()) + "\n")
            written += 1

    print(f"\nDone: {written} written, {skipped} skipped → {out_path}")


if __name__ == "__main__":
    main()
