"""Physics-informed trainer for Fourier-optics field regression.

Fixes vs. the previous version:

  * ``energy_conservation_loss`` compared ``pred[0]`` to ``pred[-1]`` — two
    *unrelated samples in the batch*. It is replaced by a per-sample Parseval
    power-consistency term: predicted field power should match the input field
    power (lossless propagation conserves power).
  * ``complex_loss`` operated on real tensors, so its phase term was degenerate.
    Fields are now consistently represented as 2-channel ``(real, imag)`` tensors
    and the loss reconstructs the complex field before computing amplitude and
    wrap-invariant phase errors.
  * removed the ``config`` used-before-assignment bug, the module-level
    ``load_config`` crash path, the duplicate imports, and the stray test/yaml
    text appended to the bottom of the file.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

try:  # optuna is optional; auto-tuning degrades gracefully without it
    import optuna

    _HAS_OPTUNA = True
except Exception:  # pragma: no cover
    _HAS_OPTUNA = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert a 2-channel ``[B, 2, H, W]`` real tensor to complex ``[B, H, W]``."""
    if torch.is_complex(x):
        return x
    if x.shape[1] == 2:
        return torch.complex(x[:, 0], x[:, 1])
    return x[:, 0].to(torch.complex64)  # single-channel amplitude, zero phase


class OpticsTrainer:
    """Trainer with early stopping, optional Optuna tuning, and a physics term."""

    def __init__(
        self,
        model: torch.nn.Module,
        wavelength: Optional[float] = None,
        pixel_size: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.best_weights = None
        self.study = None

    # --- data ---------------------------------------------------------------
    def create_dataloader(
        self, data: torch.Tensor, targets: torch.Tensor, batch_size: int = 8, shuffle: bool = True
    ) -> DataLoader:
        return DataLoader(TensorDataset(data, targets), batch_size=batch_size, shuffle=shuffle)

    # --- losses -------------------------------------------------------------
    def complex_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Amplitude MSE + wrap-invariant phase loss on complex fields."""
        p, t = to_complex(pred), to_complex(target)
        amp_loss = torch.nn.functional.mse_loss(torch.abs(p), torch.abs(t))
        # 1 - cos(dphi) is smooth and invariant to 2*pi wraps.
        phase_loss = (1.0 - torch.cos(torch.angle(p) - torch.angle(t))).mean()
        return amp_loss + 0.5 * phase_loss

    def energy_conservation_loss(self, pred: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        """Per-sample Parseval power consistency between input and prediction."""
        p, i = to_complex(pred), to_complex(inp)
        power_out = torch.sum(torch.abs(p) ** 2, dim=(-2, -1))
        power_in = torch.sum(torch.abs(i) ** 2, dim=(-2, -1))
        # relative error keeps the term scale-free across batches
        return torch.mean(((power_out - power_in) / (power_in + 1e-8)) ** 2)

    # --- train / validate ---------------------------------------------------
    def train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer, lambda_physics: float = 0.1
    ) -> float:
        self.model.train()
        total = 0.0
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.complex_loss(outputs, targets)
            if lambda_physics > 0 and data.shape[1] == outputs.shape[1]:
                loss = loss + lambda_physics * self.energy_conservation_loss(outputs, data)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total, phase_rmse = 0.0, 0.0
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                total += self.complex_loss(outputs, targets).item()
                p, t = to_complex(outputs), to_complex(targets)
                dphi = torch.angle(p) - torch.angle(t)
                # wrap the difference into (-pi, pi] before RMSE
                dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                phase_rmse += torch.sqrt(torch.mean(dphi**2)).item()
        return total / len(loader), phase_rmse / len(loader)

    # --- training loops -----------------------------------------------------
    def manual_train(self, train_loader, val_loader, config: Dict[str, Any]):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get("lr", 1e-3))
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(config["epochs"]):
            train_loss = self.train_epoch(train_loader, optimizer, config.get("lambda_physics", 0.1))
            val_loss, phase_rmse = self.validate(val_loader)
            logger.info(
                "Epoch %d: train %.4f | val %.4f | phase RMSE %.4f",
                epoch + 1, train_loss, val_loss, phase_rmse,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    logger.info("Early stopping triggered")
                    break
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
        return best_loss

    def objective(self, trial, train_loader, val_loader) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        lambda_physics = trial.suggest_float("lambda_physics", 0.0, 1.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val = float("inf")
        patience = 0
        for _ in range(20):
            self.train_epoch(train_loader, optimizer, lambda_physics)
            val_loss, _ = self.validate(val_loader)
            if val_loss < best_val:
                best_val, patience = val_loss, 0
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= 5:
                    break
        return best_val

    def auto_tune(self, train_loader, val_loader, n_trials: int = 20):
        if not _HAS_OPTUNA:
            logger.warning("optuna not installed; falling back to a single manual run")
            return self.manual_train(
                train_loader, val_loader, {"epochs": 20, "patience": 5, "lambda_physics": 0.1}
            )
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader), n_trials=n_trials
        )
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
        return self.study.best_value

    # --- io -----------------------------------------------------------------
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info("Model weights saved to %s", path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("Model weights loaded from %s", path)
