"""Headless end-to-end smoke test of the full pipeline.

The previous version blocked on ``input()``/``plt.show()``, used ``np`` without
importing it, imported a non-existent ``preprocessing.fourier_preprocessing``
path, and executed a broken deployment block at import time. This version runs
non-interactively and can be executed directly or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")  # headless

import torch

from models.architecture import ModelSelector
from preprocessing.fourier_preprocessing import FourierPreprocessing
from training.trainer import OpticsTrainer
from utils.metrics import FourierOpticsMetrics


def test_pipeline():
    torch.manual_seed(0)
    grid = 64

    # Step 1-2: ingest + preprocess a complex field
    pre = FourierPreprocessing(pixel_size=5.0, wavelength=632.8)
    data = torch.randn(grid, grid) + 1j * torch.randn(grid, grid)
    amplitude, phase = pre.preprocess(data, remove_dc=True, window_type="tukey", unwrap_phase=True)
    assert amplitude.shape == (grid, grid)

    # Step 3: model selection (2-channel real/imag convention)
    selector = ModelSelector(data_shape=(grid, grid), data_type="complex_field")
    selector.model_choice = "fno"
    model = selector.build_model()

    # Step 4: brief training on a tiny synthetic set
    trainer = OpticsTrainer(model, wavelength=632.8e-9, pixel_size=5e-6)
    x = torch.randn(8, 2, grid, grid)
    y = torch.randn(8, 2, grid, grid)
    loader = trainer.create_dataloader(x, y, batch_size=4)
    trainer.manual_train(loader, loader, {"epochs": 2, "patience": 2, "lambda_physics": 0.0})

    # Step 5: metrics on a held-out pair
    metrics = FourierOpticsMetrics(632.8e-9, 5e-6)
    pred = torch.exp(1j * torch.randn(grid, grid))
    target = torch.exp(1j * torch.randn(grid, grid))
    results = metrics.calculate_all_metrics(pred, target)
    assert "strehl_ratio" in results
    assert all(v == v for v in results.values())  # no NaNs


if __name__ == "__main__":
    test_pipeline()
    print("pipeline smoke test passed.")
