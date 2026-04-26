"""
PyTorch datasets for R2V-Agent training.

Provides:
- BCDataset: behavior cloning from teacher trajectories
- PreferenceDataset: DPO pairs from verifier-scored candidates
- VerifierDataset: (context, action, label) for verifier training
- RouterDataset: features + routing labels for router training
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from r2v.data.trajectory import (
    Episode, Step, TrajectoryStore,
)


def build_context_string(goal: str, steps: list[Step], current_step_idx: int) -> str:
    """Build the agent context x_t = (G, o_<=t, a_<t, y_<t).

    Constructs a linearized context string from the goal and trajectory history.
    """
    parts = [f"Goal: {goal}\n"]
    for i in range(current_step_idx + 1):
        s = steps[i]
        parts.append(f"--- Step {i + 1} ---")
        parts.append(f"Observation: {s.observation.raw_text}")
        if s.observation.url:
            parts.append(f"URL: {s.observation.url}")
        if i < current_step_idx:
            parts.append(f"Action: {s.action.raw_text}")
            if s.action.plan_tag:
                parts.append(f"Plan: {s.action.plan_tag}")
    return "\n".join(parts)


# ============================================================
# Behavior Cloning Dataset
# ============================================================

@dataclass
class BCExample:
    context: str
    target_action: str
    episode_id: str
    step_idx: int


class BCDataset(Dataset):
    """Dataset for behavior cloning on teacher trajectories."""

    def __init__(
        self,
        trajectory_path: str | None = None,
        tokenizer=None,
        max_seq_len: int = 4096,
        max_episodes: int | None = None,
        *,
        episodes: list | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[BCExample] = []

        if episodes is None:
            store = TrajectoryStore(trajectory_path)
            episodes = store.load_episodes(max_count=max_episodes)

        for ep in episodes:
            if not ep.success:
                continue  # Only learn from successful teacher trajectories
            for i, step in enumerate(ep.steps):
                ctx = build_context_string(ep.metadata.goal, ep.steps, i)
                self.examples.append(BCExample(
                    context=ctx,
                    target_action=step.action.raw_text,
                    episode_id=ep.episode_id,
                    step_idx=i,
                ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        # Format as instruction-following: context → action
        prompt = f"{ex.context}\nAction:"
        full_text = f"{prompt} {ex.target_action}"

        # Return plain lists — collate_fn handles dynamic padding per batch.
        # This avoids wasting ~58% of FLOPs on padding tokens (median 1723 vs max 4096).
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
        )

        # Get prompt length for label masking (no tensor overhead)
        prompt_len = len(self.tokenizer(
            prompt, max_length=self.max_seq_len, truncation=True,
        )["input_ids"])

        # Labels: -100 for prompt tokens (no loss), actual ids for target
        labels = list(encoding["input_ids"])
        # Guard against fully-masked labels when long prompts consume the
        # entire sequence budget (can produce NaN loss in CE reduction).
        num_mask = min(prompt_len, max(0, len(labels) - 1))
        labels[:num_mask] = [-100] * num_mask

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Dynamic padding: pad to the longest sequence in the batch.

        Instead of padding every sample to max_seq_len (4096), pad only to
        the longest in the current batch. This gives ~2-3× throughput boost.
        """
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [0] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# Preference Dataset (DPO)
# ============================================================

@dataclass
class PreferenceExample:
    context: str
    chosen_action: str        # a+ (verifier-preferred)
    rejected_action: str      # a- (verifier-rejected)
    chosen_score: float
    rejected_score: float
    context_alt: str | None = None  # same step, different perturbation seed


class PreferenceDataset(Dataset):
    """Dataset for DPO-style preference distillation using verifier scores."""

    def __init__(
        self,
        candidates_path: str,
        tokenizer,
        max_seq_len: int = 4096,
        min_score_gap: float = 0.1,
        allowed_episode_ids: set[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[PreferenceExample] = []
        self.stats = {
            "total_rows": 0,
            "kept": 0,
            "skipped_malformed_json": 0,
            "skipped_small_gap": 0,
            "skipped_same_action": 0,
            "skipped_invalid_schema": 0,
            "skipped_wrong_split": 0,
        }

        # Load candidate action data.
        # Supports two JSONL schemas:
        #   1. Pre-paired: {context, chosen, rejected, chosen_score, rejected_score}
        #   2. Raw candidates: {context, candidates, verifier_scores}
        import json

        logger = logging.getLogger(__name__)
        malformed_examples: list[int] = []
        with open(candidates_path, "rb") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    self.stats["skipped_malformed_json"] += 1
                    if len(malformed_examples) < 5:
                        malformed_examples.append(line_no)
                    continue

                self.stats["total_rows"] += 1
                if allowed_episode_ids is not None:
                    eid = obj.get("episode_id")
                    if eid is not None and eid not in allowed_episode_ids:
                        self.stats["skipped_wrong_split"] += 1
                        continue
                if "chosen" in obj and "rejected" in obj:
                    # Schema 1: already paired by generate_candidates
                    score_gap = obj.get("chosen_score", 1.0) - obj.get("rejected_score", 0.0)
                    if score_gap < min_score_gap:
                        self.stats["skipped_small_gap"] += 1
                        continue

                    chosen_action = obj["chosen"]
                    rejected_action = obj["rejected"]
                    if chosen_action == rejected_action:
                        self.stats["skipped_same_action"] += 1
                        continue

                    self.examples.append(PreferenceExample(
                        context=obj["context"],
                        chosen_action=chosen_action,
                        rejected_action=rejected_action,
                        chosen_score=obj.get("chosen_score", 1.0),
                        rejected_score=obj.get("rejected_score", 0.0),
                        context_alt=obj.get("context_alt"),
                    ))
                elif "verifier_scores" in obj and "candidates" in obj:
                    # Schema 2: raw candidates — pick best/worst pair
                    best_idx = max(range(len(obj["verifier_scores"])),
                                   key=lambda i: obj["verifier_scores"][i])
                    worst_idx = min(range(len(obj["verifier_scores"])),
                                    key=lambda i: obj["verifier_scores"][i])
                    score_gap = obj["verifier_scores"][best_idx] - obj["verifier_scores"][worst_idx]
                    if score_gap < min_score_gap:
                        self.stats["skipped_small_gap"] += 1
                        continue

                    chosen_action = obj["candidates"][best_idx]
                    rejected_action = obj["candidates"][worst_idx]
                    if chosen_action == rejected_action:
                        self.stats["skipped_same_action"] += 1
                        continue

                    self.examples.append(PreferenceExample(
                        context=obj["context"],
                        chosen_action=chosen_action,
                        rejected_action=rejected_action,
                        chosen_score=obj["verifier_scores"][best_idx],
                        rejected_score=obj["verifier_scores"][worst_idx],
                        context_alt=obj.get("context_alt"),
                    ))
                else:
                    self.stats["skipped_invalid_schema"] += 1

        if malformed_examples:
            logger.warning(
                "PreferenceDataset skipped %d malformed JSONL rows in %s (first bad lines: %s)",
                self.stats["skipped_malformed_json"],
                candidates_path,
                ", ".join(str(n) for n in malformed_examples),
            )

        self.stats["kept"] = len(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt = f"{ex.context}\nAction:"

        chosen_text = f"{prompt} {ex.chosen_action}"
        rejected_text = f"{prompt} {ex.rejected_action}"

        # Return plain lists — collate_fn pads to max-in-batch, not max_seq_len.
        chosen_enc = self.tokenizer(
            chosen_text, max_length=self.max_seq_len,
            truncation=True, padding=False,
        )
        rejected_enc = self.tokenizer(
            rejected_text, max_length=self.max_seq_len,
            truncation=True, padding=False,
        )

        # Tokenize prompt once for label masking (no tensor overhead)
        prompt_len = len(self.tokenizer(
            prompt, max_length=self.max_seq_len, truncation=True,
        )["input_ids"])

        chosen_labels = list(chosen_enc["input_ids"])
        num_mask_chosen = min(prompt_len, max(0, len(chosen_labels) - 1))
        chosen_labels[:num_mask_chosen] = [-100] * num_mask_chosen
        rejected_labels = list(rejected_enc["input_ids"])
        num_mask_rejected = min(prompt_len, max(0, len(rejected_labels) - 1))
        rejected_labels[:num_mask_rejected] = [-100] * num_mask_rejected

        out = {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels": rejected_labels,
        }

        if ex.context_alt is not None:
            # Tokenize alt context + same chosen action to get alt_labels with correct offsets.
            alt_prompt = f"{ex.context_alt}\nAction:"
            alt_text = f"{alt_prompt} {ex.chosen_action}"
            alt_enc = self.tokenizer(
                alt_text, max_length=self.max_seq_len, truncation=True, padding=False,
            )
            alt_prompt_len = len(self.tokenizer(
                alt_prompt, max_length=self.max_seq_len, truncation=True,
            )["input_ids"])
            alt_labels = list(alt_enc["input_ids"])
            num_mask_alt = min(alt_prompt_len, max(0, len(alt_labels) - 1))
            alt_labels[:num_mask_alt] = [-100] * num_mask_alt
            out["alt_input_ids"] = alt_enc["input_ids"]
            out["alt_attention_mask"] = alt_enc["attention_mask"]
            out["alt_labels"] = alt_labels

        return out

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Dynamic padding for DPO pairs.

        Pads chosen and rejected to the SAME max length so they can be
        concatenated into a single forward pass (2× fewer kernel launches).

        If any example has alt_input_ids (perturbed context pair from
        generate_candidates), those are padded separately and included as
        alt_input_ids / alt_attention_mask / alt_labels.
        """
        max_len = max(
            max(len(b["chosen_input_ids"]) for b in batch),
            max(len(b["rejected_input_ids"]) for b in batch),
        )

        result = {k: [] for k in [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ]}

        for b in batch:
            ch_pad = max_len - len(b["chosen_input_ids"])
            result["chosen_input_ids"].append(b["chosen_input_ids"] + [0] * ch_pad)
            result["chosen_attention_mask"].append(b["chosen_attention_mask"] + [0] * ch_pad)
            result["chosen_labels"].append(b["chosen_labels"] + [-100] * ch_pad)

            rj_pad = max_len - len(b["rejected_input_ids"])
            result["rejected_input_ids"].append(b["rejected_input_ids"] + [0] * rj_pad)
            result["rejected_attention_mask"].append(b["rejected_attention_mask"] + [0] * rj_pad)
            result["rejected_labels"].append(b["rejected_labels"] + [-100] * rj_pad)

        tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        # Include alt context tensors when any example has them.
        if any("alt_input_ids" in b for b in batch):
            alt_max_len = max(
                len(b["alt_input_ids"]) if "alt_input_ids" in b else len(b["chosen_input_ids"])
                for b in batch
            )
            alt_ids, alt_mask, alt_labs = [], [], []
            for b in batch:
                if "alt_input_ids" in b:
                    pad = alt_max_len - len(b["alt_input_ids"])
                    alt_ids.append(b["alt_input_ids"] + [0] * pad)
                    alt_mask.append(b["alt_attention_mask"] + [0] * pad)
                    alt_labs.append(b["alt_labels"] + [-100] * pad)
                else:
                    # No alt for this example — reuse chosen (R-Drop fallback per sample)
                    pad = alt_max_len - len(b["chosen_input_ids"])
                    alt_ids.append(b["chosen_input_ids"] + [0] * pad)
                    alt_mask.append(b["chosen_attention_mask"] + [0] * pad)
                    alt_labs.append(b["chosen_labels"] + [-100] * pad)
            tensors["alt_input_ids"] = torch.tensor(alt_ids, dtype=torch.long)
            tensors["alt_attention_mask"] = torch.tensor(alt_mask, dtype=torch.long)
            tensors["alt_labels"] = torch.tensor(alt_labs, dtype=torch.long)

        return tensors


# ============================================================
# Verifier Dataset
# ============================================================

class VerifierDataset(Dataset):
    """Dataset for training the step/outcome verifier V_φ."""

    def __init__(
        self,
        trajectory_path: str | None = None,
        tokenizer=None,
        max_seq_len: int = 4096,
        use_step_labels: bool = True,
        max_episodes: int | None = None,
        *,
        episodes: list | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []

        if episodes is None:
            store = TrajectoryStore(trajectory_path)
            episodes = store.load_episodes(max_count=max_episodes)

        for ep in episodes:
            for i, step in enumerate(ep.steps):
                ctx = build_context_string(ep.metadata.goal, ep.steps, i)
                text = f"{ctx}\nAction: {step.action.raw_text}"

                # Final outcome label (always available)
                entry = {
                    "text": text,
                    "final_label": float(ep.success),
                    "episode_id": ep.episode_id,
                    "step_idx": i,
                    "perturbation_type": ep.perturbation_type.value if hasattr(ep.perturbation_type, 'value') else str(ep.perturbation_type),
                }

                # Step-level label (if available)
                if use_step_labels and step.label is not None:
                    if step.label.is_correct is not None:
                        entry["step_label"] = float(step.label.is_correct)
                    elif step.label.is_progress is not None:
                        entry["step_label"] = float(step.label.is_progress)
                    else:
                        # Leave ambiguous steps unlabeled rather than leaking
                        # the final episode outcome into every step target.
                        entry["step_label"] = None
                else:
                    entry["step_label"] = None

                self.examples.append(entry)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        # Do NOT pad here — use the collate_fn for dynamic padding per batch.
        # This avoids wasting compute on padding tokens in the backbone forward pass.
        enc = self.tokenizer(
            ex["text"], max_length=self.max_seq_len,
            truncation=True, padding=False,
        )

        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "final_label": ex["final_label"],
        }
        if ex["step_label"] is not None:
            item["step_label"] = ex["step_label"]
            item["has_step_label"] = 1.0
        else:
            item["step_label"] = 0.0
            item["has_step_label"] = 0.0

        return item

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Dynamic padding: pad to the longest sequence in the batch."""
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids = []
        attention_mask = []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [0] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "final_label": torch.tensor([b["final_label"] for b in batch], dtype=torch.float32),
            "step_label": torch.tensor([b["step_label"] for b in batch], dtype=torch.float32),
            "has_step_label": torch.tensor([b["has_step_label"] for b in batch], dtype=torch.float32),
        }


# ============================================================
# Router Dataset
# ============================================================

@dataclass
class RouterExample:
    """Feature vector + label for router training."""
    features: list[float]   # [verifier_score, entropy, step_pct, token_pct, ...]
    label: float            # 1.0 = should fallback to LLM
    success: float          # Episode success (for CVaR computation)
    perturbation_seed: int
    cost: float


class RouterDataset(Dataset):
    """Dataset for training the risk-calibrated router."""

    def __init__(
        self,
        examples: list[RouterExample],
        *,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        feats = np.array([e.features for e in examples], dtype=np.float32)
        self.mean: np.ndarray = mean if mean is not None else feats.mean(axis=0)
        self.std: np.ndarray = std if std is not None else feats.std(axis=0).clip(min=1e-6)

        # Pre-normalise the whole matrix once at construction time so that
        # __getitem__ is a zero-cost tensor slice instead of per-sample arithmetic.
        normed = (feats - self.mean) / self.std
        self._features = torch.from_numpy(normed)
        self._labels   = torch.tensor([e.label for e in examples],              dtype=torch.float32)
        self._success  = torch.tensor([e.success for e in examples],            dtype=torch.float32)
        self._seeds    = torch.tensor([e.perturbation_seed for e in examples],  dtype=torch.long)
        self._costs    = torch.tensor([e.cost for e in examples],               dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features":          self._features[idx],
            "label":             self._labels[idx],
            "success":           self._success[idx],
            "perturbation_seed": self._seeds[idx],
            "cost":              self._costs[idx],
        }

    def to(self, device: torch.device) -> "RouterDataset":
        """Move all pre-computed tensors to *device* (e.g. GPU) in-place."""
        self._features = self._features.to(device)
        self._labels   = self._labels.to(device)
        self._success  = self._success.to(device)
        self._seeds    = self._seeds.to(device)
        self._costs    = self._costs.to(device)
        return self

    @classmethod
    def from_episodes(
        cls,
        episodes: list[Episode],
        verifier_scores: dict[str, list[float]],
        entropies: dict[str, list[float]],
        cost_slm: float = 1.0,
        cost_llm: float = 50.0,
        risk_threshold: float = 0.5,
    ) -> "RouterDataset":
        """Build router dataset from evaluated episodes with verifier scores."""
        examples = []
        for ep in episodes:
            ep_scores = verifier_scores.get(ep.episode_id, [])
            ep_entropies = entropies.get(ep.episode_id, [])

            for i, step in enumerate(ep.steps):
                v_score = ep_scores[i] if i < len(ep_scores) else 0.5
                entropy = ep_entropies[i] if i < len(ep_entropies) else 1.0
                step_pct = (i + 1) / max(len(ep.steps), 1)
                risk = (1.0 - v_score) * 0.7 + entropy * 0.3  # risk estimate ρ(x_t)

                features = [v_score, entropy, step_pct, risk]
                should_fallback = float(risk > risk_threshold)
                cost = cost_llm if should_fallback else cost_slm

                examples.append(RouterExample(
                    features=features,
                    label=should_fallback,
                    success=float(ep.success),
                    perturbation_seed=ep.perturbation_seed or 0,
                    cost=cost,
                ))
        return cls(examples)
