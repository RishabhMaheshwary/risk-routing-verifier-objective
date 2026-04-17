"""
Core data structures for agent trajectories and episodes.

Defines the canonical representation used throughout the pipeline:
teacher collection → perturbation → labeling → training → evaluation.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import jsonlines


class ActionSource(str, Enum):
    """Which model produced the action."""
    TEACHER = "teacher"
    SLM = "slm"
    LLM_FALLBACK = "llm_fallback"
    SELF_CORRECTED = "self_corrected"


class PerturbationType(str, Enum):
    """Perturbation categories applied to trajectories."""
    NONE = "none"
    TOOL_FLAKINESS = "tool_flakiness"
    PARTIAL_OBSERVABILITY = "partial_observability"
    PROMPT_INJECTION = "prompt_injection"
    DISTRACTOR = "distractor"
    COMPOSITE = "composite"  # Multiple perturbations combined


@dataclass
class Observation:
    """Agent observation at a single timestep."""
    raw_text: str                     # Full observation text (accessibility tree / logs)
    url: Optional[str] = None         # Current URL (WebArena)
    dom_snapshot: Optional[str] = None  # Raw HTML/DOM (if available)
    metadata: dict[str, Any] = field(default_factory=dict)

    def tokenize_length(self, tokenizer) -> int:
        """Estimate token count for this observation."""
        return len(tokenizer.encode(self.raw_text))


@dataclass
class Action:
    """Agent action at a single timestep."""
    raw_text: str                     # Full action string (WebArena grammar / patch)
    action_type: Optional[str] = None  # e.g., "click", "type", "scroll", "stop"
    arguments: dict[str, Any] = field(default_factory=dict)
    plan_tag: Optional[str] = None     # Optional structured plan string

    @property
    def is_stop(self) -> bool:
        return self.action_type in ("stop", "finish", "submit")


@dataclass
class StepLabel:
    """Labels for a single step, used for verifier training."""
    is_correct: Optional[bool] = None       # Step-level correctness
    is_progress: Optional[bool] = None      # Makes progress toward goal
    verifier_score: Optional[float] = None  # V_φ(x, a) score
    teacher_agreement: Optional[bool] = None  # Matches teacher action
    safety_violation: Optional[bool] = None   # Executed injected instruction
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single step in an agent trajectory."""
    step_idx: int
    observation: Observation
    action: Action
    action_source: ActionSource
    reward: float = 0.0
    label: Optional[StepLabel] = None

    # Context = (goal, o_<=t, a_<t, y_<t) — built during training
    context: Optional[str] = None

    # Perturbation applied to this step's observation
    perturbation_type: PerturbationType = PerturbationType.NONE
    perturbation_seed: Optional[int] = None
    perturbation_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeMetadata:
    """Metadata about a full episode."""
    task_id: str
    template_id: Optional[str] = None
    goal: str = ""
    benchmark: str = ""  # e.g. "webarena", "gaia", "humaneval", "alfworld"
    site: Optional[str] = None  # WebArena site
    repo: Optional[str] = None  # repository (if applicable)
    difficulty: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Data versioning ──────────────────────────────────────────
    # Set by collect_trajectories.py; used by DataRegistry to group and
    # query episodes across multi-model collection runs.
    run_id: str = ""                # e.g. "webarena_openai_gpt-4o_20260218T210012"
    teacher_model: str = ""         # e.g. "gpt-4o", "claude-3-5-sonnet-20241022"
    teacher_provider: str = ""      # e.g. "openai", "anthropic", "google"


@dataclass
class Episode:
    """A complete agent episode (trajectory)."""
    episode_id: str
    metadata: EpisodeMetadata
    steps: list[Step]

    # Episode-level outcomes
    success: bool = False           # Final success S(τ)
    partial_score: float = 0.0      # Partial credit if applicable
    total_cost: float = 0.0         # Token/API cost
    num_llm_fallbacks: int = 0
    num_self_corrections: int = 0
    wall_time_seconds: float = 0.0

    # Perturbation info
    perturbation_type: PerturbationType = PerturbationType.NONE
    perturbation_seed: Optional[int] = None

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def trajectory_hash(self) -> str:
        """Deterministic hash for deduplication."""
        content = json.dumps(
            [s.action.raw_text for s in self.steps], sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        """Deserialize from dictionary."""
        meta = EpisodeMetadata(**data.pop("metadata"))
        steps = []
        for s in data.pop("steps"):
            obs = Observation(**s.pop("observation"))
            act = Action(**s.pop("action"))
            label = StepLabel(**s.pop("label")) if s.get("label") else None
            _step_fields = {f.name for f in fields(Step)}
            steps.append(Step(
                observation=obs, action=act, label=label,
                action_source=ActionSource(s.pop("action_source")),
                perturbation_type=PerturbationType(s.pop("perturbation_type", "none")),
                **{k: v for k, v in s.items() if k not in ("observation", "action", "label") and k in _step_fields}
            ))
        return cls(metadata=meta, steps=steps, **data)


# ============================================================
# Trajectory I/O
# ============================================================

class TrajectoryStore:
    """Persistent storage for trajectory datasets using JSONL.

    Parameters
    ----------
    path : str | Path
        Destination JSONL file.
    async_write : bool
        When *True* a background thread handles disk I/O so callers
        (e.g. GPU workers) are never blocked on ``write()`` / ``flush()``.
    """

    def __init__(self, path: str | Path, *, async_write: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._async = async_write
        self._writer = None
        if async_write:
            from r2v.utils.async_writer import AsyncJSONLWriter
            self._writer = AsyncJSONLWriter(self.path, mode="a")
            self._writer.start()

    def save_episode(self, episode: Episode) -> None:
        """Append a single episode to the JSONL file."""
        if self._writer is not None:
            self._writer.write(episode.to_dict())
        else:
            with jsonlines.open(self.path, mode="a") as writer:
                writer.write(episode.to_dict())

    def save_episodes(self, episodes: list[Episode]) -> None:
        """Append episodes to the JSONL file."""
        if self._writer is not None:
            self._writer.write_many([ep.to_dict() for ep in episodes])
        else:
            with jsonlines.open(self.path, mode="a") as writer:
                for ep in episodes:
                    writer.write(ep.to_dict())

    def flush(self) -> None:
        """Block until all queued writes are flushed (async mode only)."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Drain outstanding writes and shut down the background thread."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def load_episodes(self, max_count: int | None = None) -> list[Episode]:
        """Load episodes from the JSONL file.

        Flushes the async writer first so reads are consistent.
        """
        self.flush()
        episodes = []
        if not self.path.exists():
            return episodes
        with jsonlines.open(self.path, mode="r") as reader:
            for i, obj in enumerate(reader):
                if max_count and i >= max_count:
                    break
                episodes.append(Episode.from_dict(obj))
        return episodes

    def count(self) -> int:
        """Count episodes without loading all into memory."""
        self.flush()
        if not self.path.exists():
            return 0
        count = 0
        with jsonlines.open(self.path, mode="r") as reader:
            for _ in reader:
                count += 1
        return count

    def iter_episodes(self):
        """Iterate over episodes without loading all into memory."""
        if not self.path.exists():
            return
        with jsonlines.open(self.path, mode="r") as reader:
            for obj in reader:
                yield Episode.from_dict(obj)
