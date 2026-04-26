"""
Router Trainer: risk-calibrated cost-robustness optimization.

Trains the router r_ψ using the Lagrangian formulation:
  min_ψ max_λ  E[cost(d)] + λ(CVaR_α(1 - S_z) - ε)

The router learns to fall back to the LLM teacher when:
1. The verifier score is low (SLM action likely wrong)
2. The policy entropy is high (SLM is uncertain)
3. The combined risk exceeds the learned threshold

Calibration is enforced via Brier score and post-hoc temperature scaling.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from r2v.models.router import Router, RouterLoss, TemperatureScaling

logger = logging.getLogger(__name__)


class RouterTrainer:
    """Trainer for the risk-calibrated router."""

    def __init__(
        self,
        router: Router,
        train_dataset,
        eval_dataset=None,
        config: dict[str, Any] = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        output_dir: str = "experiments/checkpoints/router",
    ):
        self.router = router
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl_path = self.output_dir / "metrics_history.jsonl"
        self.metrics_json_path = self.output_dir / "metrics_history.json"
        self.metrics_csv_path = self.output_dir / "metrics_history.csv"
        self.checkpoint_manifest_path = self.output_dir / "checkpoint_manifest.jsonl"

        self.epochs = self.config.get("epochs", 20)
        self.batch_size = self.config.get("batch_size", 64)
        self.lr = self.config.get("learning_rate", 1e-3)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.lagrangian_lr = self.config.get("lagrangian_lr", 0.01)
        self.eval_every_epochs = int(self.config.get("eval_every_epochs", 1))
        self.checkpoint_every_epochs = int(self.config.get("checkpoint_every_epochs", 1))
        self.save_optimizer_state = bool(self.config.get("save_optimizer_state", True))

        self.loss_fn = RouterLoss(self.config)
        self.temp_scaling = TemperatureScaling(
            num_bins=self.config.get("num_bins", 15)
        )

    @staticmethod
    def _iter_batches(
        dataset,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
    ):
        """Yield batches by slicing pre-normalised tensors directly.

        Avoids the per-sample Python overhead of DataLoader/__getitem__ when
        the entire dataset is already in memory as contiguous tensors.
        The first call moves the dataset tensors to *device* in-place if they
        are not already there.
        """
        n = len(dataset)
        dataset.to(device)
        idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
        for start in range(0, n, batch_size):
            sl = idx[start : start + batch_size]
            yield {
                "features":          dataset._features[sl],
                "label":             dataset._labels[sl],
                "success":           dataset._success[sl],
                "perturbation_seed": dataset._seeds[sl],
                "cost":              dataset._costs[sl],
            }

    def train(self) -> dict[str, Any]:
        """Run router training with Lagrangian dual optimization."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.router = self.router.to(device)
        self.router.train()

        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size * 4,
                shuffle=False,
                num_workers=0,
            )

        # Separate optimizers for primal (router params) and dual (λ)
        router_params = [
            p for name, p in self.router.named_parameters()
            if p.requires_grad and name != "log_lambda"
        ]
        primal_optimizer = torch.optim.AdamW(
            router_params, lr=self.lr, weight_decay=self.weight_decay
        )
        dual_optimizer = torch.optim.Adam(
            [self.router.log_lambda], lr=self.lagrangian_lr
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            primal_optimizer, T_max=self.epochs
        )

        metrics_history: list[dict[str, Any]] = []
        best_eval_metric = float("inf")
        best_epoch = -1
        best_checkpoint_path: str | None = None

        for epoch in range(self.epochs):
            epoch_metrics = {}
            num_batches = 0

            for batch in self._iter_batches(self.train_dataset, self.batch_size, True, device):
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                success = batch["success"].to(device)
                seeds = batch["perturbation_seed"].to(device)

                # Forward pass
                fallback_probs = self.router(features)

                # Compute loss
                losses = self.loss_fn(
                    fallback_probs, labels, success, seeds,
                    self.router.lagrange_multiplier,
                )

                # Primal step (minimize router loss)
                primal_optimizer.zero_grad()
                losses["total_loss"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(router_params, 1.0)
                primal_optimizer.step()

                # Dual step (maximize λ * constraint violation)
                dual_optimizer.zero_grad()
                losses["dual_loss"].backward()
                dual_optimizer.step()

                # Accumulate metrics
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        epoch_metrics[k] = epoch_metrics.get(k, 0) + v.item()
                num_batches += 1

            scheduler.step()

            # Epoch summary
            avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
            epoch_record: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_cost_loss": avg_metrics.get("cost_loss", 0.0),
                "train_robustness_loss": avg_metrics.get("robustness_loss", 0.0),
                "train_calibration_loss": avg_metrics.get("calibration_loss", 0.0),
                "train_total_loss": avg_metrics.get("total_loss", 0.0),
                "train_dual_loss": avg_metrics.get("dual_loss", 0.0),
                "train_constraint_violation": avg_metrics.get("constraint_violation", 0.0),
                "lambda": float(self.router.lagrange_multiplier.item()),
                "temperature": float(self.router.temperature.item()),
                "lr": float(primal_optimizer.param_groups[0]["lr"]),
            }

            logger.info(
                f"[Router] Epoch {epoch+1}/{self.epochs} "
                f"Cost={epoch_record['train_cost_loss']:.4f} "
                f"Robust={epoch_record['train_robustness_loss']:.4f} "
                f"Calib={epoch_record['train_calibration_loss']:.4f} "
                f"λ={epoch_record['lambda']:.4f} "
                f"T={epoch_record['temperature']:.4f}"
            )

            # Evaluation
            if eval_loader and (epoch + 1) % self.eval_every_epochs == 0:
                eval_metrics = self._evaluate(eval_loader, device)
                epoch_record.update(
                    {
                        "eval_brier": float(eval_metrics["brier"]),
                        "eval_ece": float(eval_metrics["ece"]),
                        "eval_accuracy": float(eval_metrics["accuracy"]),
                    }
                )
                logger.info(
                    f"[Router] Eval: ECE={eval_metrics['ece']:.4f} "
                    f"Brier={eval_metrics['brier']:.4f} "
                    f"Acc={eval_metrics['accuracy']:.4f}"
                )
                if eval_metrics["brier"] < best_eval_metric:
                    best_eval_metric = eval_metrics["brier"]
                    best_epoch = epoch + 1
                    best_checkpoint_path = str(self.checkpoints_dir / f"checkpoint_epoch_{epoch+1:03d}.pt")
                    self._save(
                        "best",
                        epoch=epoch + 1,
                        epoch_metrics=epoch_record,
                        primal_optimizer=primal_optimizer,
                        dual_optimizer=dual_optimizer,
                        scheduler=scheduler,
                    )

            metrics_history.append(epoch_record)
            self._append_metrics_jsonl(epoch_record)
            self._write_metrics_history(metrics_history)

            if (epoch + 1) % self.checkpoint_every_epochs == 0:
                self._save(
                    f"checkpoint_epoch_{epoch+1:03d}",
                    epoch=epoch + 1,
                    epoch_metrics=epoch_record,
                    primal_optimizer=primal_optimizer,
                    dual_optimizer=dual_optimizer,
                    scheduler=scheduler,
                )

        # Post-hoc temperature scaling on eval set
        # Skip when router.temperature_scaling=false (no_temp_scaling ablation)
        if eval_loader and self.config.get("temperature_scaling", True):
            self._calibrate(eval_loader, device)

        final_epoch_metrics = metrics_history[-1] if metrics_history else None
        self._save(
            "final",
            epoch=self.epochs,
            epoch_metrics=final_epoch_metrics,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
            scheduler=scheduler,
        )

        return {
            "history": metrics_history,
            "best_eval_brier": best_eval_metric,
            "best_epoch": best_epoch,
            "best_checkpoint_path": best_checkpoint_path,
            "metrics_jsonl": str(self.metrics_jsonl_path),
            "metrics_json": str(self.metrics_json_path),
            "metrics_csv": str(self.metrics_csv_path),
            "checkpoint_manifest": str(self.checkpoint_manifest_path),
        }

    @torch.no_grad()
    def _evaluate(self, eval_loader, device) -> dict[str, float]:
        """Evaluate router on held-out data."""
        self.router.eval()
        all_probs = []
        all_labels = []

        for batch in eval_loader:
            features = batch["features"].to(device)
            probs = self.router(features)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        brier = np.mean((all_probs - all_labels) ** 2)
        ece = self.temp_scaling.compute_ece(all_probs, all_labels)
        accuracy = np.mean((all_probs > 0.5) == all_labels)

        self.router.train()
        return {"brier": brier, "ece": ece, "accuracy": accuracy}

    def _calibrate(self, eval_loader, device):
        """Apply post-hoc temperature scaling."""
        self.router.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                features = batch["features"].to(device)
                logits = self.router.mlp(features).squeeze(-1)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["label"].numpy())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        self.temp_scaling.fit(all_logits, all_labels)

        # Update router temperature
        with torch.no_grad():
            self.router.temperature.fill_(self.temp_scaling.temperature)

        logger.info(f"[Router] Post-hoc calibration temperature: {self.temp_scaling.temperature:.4f}")

    def _append_metrics_jsonl(self, epoch_record: dict[str, Any]) -> None:
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record) + "\n")

    def _write_metrics_history(self, metrics_history: list[dict[str, Any]]) -> None:
        with open(self.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, indent=2)

        if not metrics_history:
            return

        fieldnames: list[str] = sorted(
            {k for rec in metrics_history for k in rec.keys()}
        )
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in metrics_history:
                writer.writerow(rec)

    def _save(
        self,
        name: str,
        *,
        epoch: int,
        epoch_metrics: dict[str, Any] | None,
        primal_optimizer: torch.optim.Optimizer | None = None,
        dual_optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        """Save router checkpoint and append to checkpoint manifest."""
        if name.startswith("checkpoint_epoch_"):
            save_path = self.checkpoints_dir / f"{name}.pt"
        else:
            save_path = self.output_dir / f"{name}.pt"

        payload: dict[str, Any] = {
            "epoch": epoch,
            "router_state_dict": self.router.state_dict(),
            "temperature": float(self.temp_scaling.temperature),
            "router_temperature_parameter": float(self.router.temperature.item()),
            "lagrange_multiplier": float(self.router.lagrange_multiplier.item()),
            "config": self.config,
            "epoch_metrics": epoch_metrics,
        }
        payload.update(self.checkpoint_metadata)

        if self.save_optimizer_state:
            if primal_optimizer is not None:
                payload["primal_optimizer_state_dict"] = primal_optimizer.state_dict()
            if dual_optimizer is not None:
                payload["dual_optimizer_state_dict"] = dual_optimizer.state_dict()
            if scheduler is not None:
                payload["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(payload, save_path)
        logger.info(f"[Router] Saved checkpoint: {save_path}")

        manifest_row = {
            "name": name,
            "path": str(save_path),
            "epoch": epoch,
            "eval_brier": None if epoch_metrics is None else epoch_metrics.get("eval_brier"),
            "eval_ece": None if epoch_metrics is None else epoch_metrics.get("eval_ece"),
            "eval_accuracy": None if epoch_metrics is None else epoch_metrics.get("eval_accuracy"),
            "train_total_loss": None if epoch_metrics is None else epoch_metrics.get("train_total_loss"),
        }
        with open(self.checkpoint_manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_row) + "\n")
