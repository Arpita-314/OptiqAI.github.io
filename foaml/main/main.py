"""End-to-end demo of the Fourier-Optics AutoML framework.

Rewritten to actually run. The previous ``main.py`` had a syntax error (a
dedented block mid-file), imported modules that did not exist
(``preprocessing.fourier_preprocessing`` when the file was ``preprocessing.py``,
``data.ingestion`` when the file was ``data/data/ingestion.py``), used ``np``
without importing it, and blocked on ``input()`` everywhere.

This version is non-interactive by default and demonstrates a *meaningful*
surrogate task: an FNO learns to emulate free-space angular-spectrum
propagation. Run:

    python main/main.py            # from the foaml/ directory
    python main/main.py --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys

# make the flat intra-package imports resolve when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.architecture import ModelSelector
from optics.propagator import AngularSpectrum2d
from training.trainer import OpticsTrainer, to_complex
from utils.metrics import FourierOpticsMetrics


def make_dataset(n_samples: int, grid: int, propagator: AngularSpectrum2d):
    """Random smooth complex fields and their true propagated counterparts.

    Represented as 2-channel (real, imag) tensors so a single convention flows
    through the whole pipeline.
    """
    # low-pass random fields (smooth, band-limited, physically reasonable inputs)
    spec = torch.randn(n_samples, grid, grid, dtype=torch.complex64)
    kk = torch.fft.fftfreq(grid)
    mask = (kk[:, None] ** 2 + kk[None, :] ** 2) < 0.05**2
    fields = torch.fft.ifft2(torch.fft.fft2(spec) * mask)
    fields = fields / fields.abs().amax(dim=(-2, -1), keepdim=True)

    with torch.no_grad():
        targets = propagator(fields.to(torch.complex128)).to(torch.complex64)

    def to2ch(z):
        return torch.stack([z.real, z.imag], dim=1).float()

    return to2ch(fields), to2ch(targets)


def main():
    parser = argparse.ArgumentParser(description="Fourier-Optics AutoML demo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--wavelength", type=float, default=1550e-9)
    parser.add_argument("--pixel-size", type=float, default=0.5e-6)
    parser.add_argument("--distance", type=float, default=20e-6)
    args = parser.parse_args()

    print("== Fourier-Optics AutoML: learning the propagation operator ==")

    propagator = AngularSpectrum2d(
        wavelength=args.wavelength, dx=args.pixel_size, distance=args.distance
    )
    train_x, train_y = make_dataset(args.n_train, args.grid, propagator)
    val_x, val_y = make_dataset(args.n_val, args.grid, propagator)
    print(f"dataset: train {tuple(train_x.shape)} -> {tuple(train_y.shape)}")

    selector = ModelSelector(data_shape=(args.grid, args.grid), data_type="complex_field")
    print(f"auto-recommended model: {selector.available_models[selector.auto_recommend()]}")
    selector.model_choice = "fno"
    model = selector.build_model()

    trainer = OpticsTrainer(model, wavelength=args.wavelength, pixel_size=args.pixel_size)
    train_loader = trainer.create_dataloader(train_x, train_y, batch_size=8)
    val_loader = trainer.create_dataloader(val_x, val_y, batch_size=8, shuffle=False)

    start_loss, _ = trainer.validate(val_loader)
    trainer.manual_train(
        train_loader, val_loader,
        {"epochs": args.epochs, "patience": args.epochs, "lambda_physics": 0.05, "lr": 3e-3},
    )
    end_loss, phase_rmse = trainer.validate(val_loader)
    print(f"val loss: {start_loss:.4f} -> {end_loss:.4f} | phase RMSE {phase_rmse:.4f}")

    # honest metric read on one held-out example
    metrics = FourierOpticsMetrics(args.wavelength, args.pixel_size)
    with torch.no_grad():
        pred = to_complex(model(val_x[:1]))[0]
        target = to_complex(val_y[:1])[0]
    results = metrics.calculate_all_metrics(pred, target)
    print("held-out metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("done.")


if __name__ == "__main__":
    main()
