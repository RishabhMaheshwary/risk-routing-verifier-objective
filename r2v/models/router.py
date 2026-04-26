"""
Router r_ψ(x_t) → [0, 1]: probability of using LLM fallback at step t.

The router is a lightweight classifier/regressor on:
- Verifier score V_φ(x_t, â_SLM)
- Policy entropy H(π_θ(·|x_t))
- Step number / episode progress
- Optional: policy hidden state summary

Key innovation: trained with a **risk-calibrated** objective that
optimizes cost subject to robust (CVaR/worst-case) success constraints
over perturbation seeds.

Training uses a Lagrangian formulation:
  min_ψ max_λ  E[cost(d)] + λ(CVaR_α(1 - S_z) - ε)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class Router(nn.Module):
    """Risk-calibrated routing network.

    Input features:
    - verifier_score: V_φ(x, â_SLM) ∈ [0, 1]
    - entropy: H(π_θ(·|x)) ∈ R+
    - step_pct: step_t / T ∈ [0, 1]
    - risk_score: ρ(x_t) = f(1 - V, H) ∈ [0, 1]

    Output: calibrated probability of fallback to LLM ∈ [0, 1]
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

        # Feature dimensions
        input_dim = self._compute_input_dim(config.get("input_features", {}))
        hidden_dims = config.get("hidden_dims", [128, 64])
        dropout = config.get("dropout", 0.2)

        # Build MLP
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Temperature scaling for calibration (post-hoc)
        self.temperature = nn.Parameter(torch.ones(1))

        # Lagrange multiplier for CVaR constraint (learned during training)
        # Start higher so the constraint is competitive with cost gradient from epoch 1.
        log_lambda_init = float(config.get("log_lambda_init", 1.5))
        self.log_lambda = nn.Parameter(torch.full((1,), log_lambda_init))

        logger.info(
            f"Router initialized: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _compute_input_dim(self, feature_config: dict) -> int:
        """Compute input dimension from feature configuration."""
        dim = 0
        if feature_config.get("verifier_score", True):
            dim += 1
        if feature_config.get("entropy", True):
            dim += 1
        if feature_config.get("step_number", True):
            dim += 1
        if feature_config.get("token_count", False):
            dim += 1
        # Hidden state projection dimension
        dim += feature_config.get("policy_hidden_dim", 0)
        # Risk score (derived feature)
        dim += 1  # Always include risk score
        return max(dim, 4)  # Minimum 4 features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → fallback probability.

        Args:
            features: [batch_size, input_dim] feature vectors

        Returns:
            [batch_size] fallback probabilities in [0, 1]
        """
        logits = self.mlp(features).squeeze(-1)
        # Apply temperature scaling for calibration
        scaled_logits = logits / self.temperature.clamp(min=0.01)
        return torch.sigmoid(scaled_logits)

    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary routing decision."""
        probs = self.forward(features)
        return (probs > threshold).long()

    @property
    def lagrange_multiplier(self) -> torch.Tensor:
        """Non-negative Lagrange multiplier for CVaR constraint."""
        return F.softplus(self.log_lambda)


class RouterLoss(nn.Module):
    """Risk-calibrated routing loss with CVaR constraint.

    Implements the Lagrangian:
        L(ψ, λ) = E[cost(d)] + λ(CVaR_α(1 - S_z) - ε)

    Where:
    - cost(d) = c_SLM * (1-d) + c_LLM * d  (per-step expected cost)
    - CVaR_α(1-S_z) = tail average failure rate over perturbation seeds
    - λ is learned via dual ascent
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.objective = config.get("robust_objective", "cvar")
        self.cvar_alpha = config.get("cvar_alpha", 0.2)
        self.cvar_epsilon = config.get("cvar_epsilon", 0.3)
        self.cost_slm = config.get("cost_slm", 1.0)
        self.cost_llm = config.get("cost_llm", 50.0)
        self.lagrangian_lr = config.get("lagrangian_lr", 0.01)

        # Calibration loss weights (brier_weight=0.0 disables calibration loss)
        self.brier_weight = float(config.get("brier_weight", 1.0))
        self.ece_weight = 0.5

    def forward(
        self,
        fallback_probs: torch.Tensor,     # Router outputs [batch]
        labels: torch.Tensor,              # Ground-truth should-fallback [batch]
        success: torch.Tensor,             # Episode success [batch]
        perturbation_seeds: torch.Tensor,  # Perturbation seed IDs [batch]
        lagrange_multiplier: torch.Tensor, # λ from router
    ) -> dict[str, torch.Tensor]:
        """Compute routing loss.

        Returns dict with:
        - total_loss: combined loss for backprop
        - cost_loss: expected cost term
        - robustness_loss: CVaR constraint violation
        - calibration_loss: Brier score
        - lagrangian_loss: for λ update (negate for dual ascent)
        """
        # 1. Cost loss: expected per-step cost
        expected_cost = (
            fallback_probs * self.cost_llm
            + (1 - fallback_probs) * self.cost_slm
        )
        cost_loss = expected_cost.mean()

        # 2. Robustness loss (CVaR or worst-case)
        if self.objective == "cvar":
            robustness_loss = self._compute_cvar_loss(
                fallback_probs, success, perturbation_seeds
            )
        elif self.objective == "worst_case":
            robustness_loss = self._compute_worst_case_loss(
                fallback_probs, success, perturbation_seeds
            )
        else:  # expected
            robustness_loss = F.binary_cross_entropy(
                fallback_probs, labels, reduction="mean"
            )

        # 3. Calibration loss (Brier score)
        brier_loss = F.mse_loss(fallback_probs, labels)

        # 4. Lagrangian: router minimizes, λ maximizes constraint violation
        constraint_violation = robustness_loss - self.cvar_epsilon
        lagrangian_term = lagrange_multiplier * constraint_violation

        # Total loss for router parameters (minimize)
        total_loss = cost_loss + lagrangian_term + self.brier_weight * brier_loss

        # Dual loss for λ (maximize constraint violation → minimize negative)
        dual_loss = -lagrange_multiplier * constraint_violation.detach()

        return {
            "total_loss": total_loss,
            "cost_loss": cost_loss,
            "robustness_loss": robustness_loss,
            "calibration_loss": brier_loss,
            "lagrangian_term": lagrangian_term,
            "dual_loss": dual_loss,
            "constraint_violation": constraint_violation.detach(),
        }

    def _compute_cvar_loss(
        self,
        fallback_probs: torch.Tensor,
        success: torch.Tensor,
        perturbation_seeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CVaR_α(1 - S_z): tail average failure rate.

        Groups samples by perturbation seed, computes per-seed failure rate,
        then takes CVaR (average of worst α-fraction).
        """
        # Effective failure after routing decisions.
        # If the router falls back (d≈1), failure risk should decrease.
        # If it keeps SLM (d≈0), failure risk remains close to (1-success).
        failure = (1.0 - success) * (1.0 - fallback_probs)
        unique_seeds = perturbation_seeds.unique()

        if len(unique_seeds) <= 1:
            return failure.mean()

        # Compute per-seed failure rate
        seed_failures = []
        for seed in unique_seeds:
            mask = perturbation_seeds == seed
            if mask.sum() > 0:
                # Weight failure by routing decision quality
                weighted_failure = failure[mask].mean()
                seed_failures.append(weighted_failure)

        seed_failures = torch.stack(seed_failures)

        # CVaR: average of worst α-fraction of seeds
        k = max(1, int(len(seed_failures) * self.cvar_alpha))
        topk_failures, _ = torch.topk(seed_failures, k)
        cvar = topk_failures.mean()

        return cvar

    def _compute_worst_case_loss(
        self,
        fallback_probs: torch.Tensor,
        success: torch.Tensor,
        perturbation_seeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute max_z (1 - S_z): worst-case failure rate."""
        # Use post-routing effective failure, not static episode failure.
        failure = (1.0 - success) * (1.0 - fallback_probs)
        unique_seeds = perturbation_seeds.unique()

        if len(unique_seeds) <= 1:
            return failure.mean()

        worst_failure = torch.tensor(0.0, device=failure.device)
        for seed in unique_seeds:
            mask = perturbation_seeds == seed
            if mask.sum() > 0:
                seed_failure = failure[mask].mean()
                worst_failure = torch.max(worst_failure, seed_failure)

        return worst_failure


class TemperatureScaling:
    """Post-hoc temperature scaling for router calibration.

    Fits a single temperature parameter T on a held-out validation set
    to minimize negative log-likelihood. Applied after training.
    """

    def __init__(self, num_bins: int = 15):
        self.temperature = 1.0
        self.num_bins = num_bins

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """Fit temperature on validation set."""
        from scipy.optimize import minimize_scalar

        def nll(T):
            scaled = logits / T
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            )

        result = minimize_scalar(nll, bounds=(0.01, 5.0), method="bounded")
        self.temperature = result.x
        logger.info(f"Calibration temperature: {self.temperature:.4f}")

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    def compute_ece(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        for i in range(self.num_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
        return ece
