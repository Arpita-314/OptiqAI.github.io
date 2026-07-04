"""Model zoo for the Fourier-Optics AutoML framework.

Rewritten to fix three defects in the previous version:

  * ``FourierBlock`` claimed to be a Fourier Neural Operator but was a 3x3 spatial
    convolution applied to raw FFT coefficients (mixes adjacent frequency bins;
    none of the operator-theoretic properties of an FNO). It is replaced by a
    real :class:`SpectralConv2d` (learned complex weights on a truncated set of
    low modes, per Li et al. 2020), which is resolution-invariant.
  * ``_build_unet`` referenced ``self.enforce_physics`` from inside a nested class
    (wrong ``self``) and had no ``forward`` -> crashed on instantiation. Fixed.
  * ``_build_diffopt`` put an ``nn.Parameter`` inside an ``nn.Sequential``
    (``TypeError``). Replaced by a genuine differentiable optical model built on
    the angular-spectrum propagator with a learnable propagation distance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from optics.propagator import AngularSpectrum2d


# --------------------------------------------------------------------------- #
# Real Fourier Neural Operator
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """Canonical FNO spectral convolution (Li et al., 2020).

    Applies a learned *complex* linear transform to a truncated set of low
    Fourier modes, then transforms back. This is a global convolution and is
    resolution-invariant: the number of learned modes is fixed regardless of the
    spatial resolution of the input.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # kept Fourier modes along dim -2
        self.modes2 = modes2  # kept Fourier modes along dim -1 (rfft axis)

        scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(inp: Tensor, weights: Tensor) -> Tensor:
        # (batch, in, x, y), (in, out, x, y) -> (batch, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft = torch.zeros(
            batch, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        # low modes: top-left corner and bottom-left corner (negative freqs)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weight2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D field-to-field regression."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        width: int = 32,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.project1 = nn.Conv2d(width, 128, kernel_size=1)
        self.project2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = F.gelu(spec(x) + pw(x))
        x = F.gelu(self.project1(x))
        return self.project2(x)


# --------------------------------------------------------------------------- #
# U-Net with a spectral bottleneck
# --------------------------------------------------------------------------- #
def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SpectralUNet(nn.Module):
    """Compact U-Net whose bottleneck is an FNO spectral layer."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base: int = 32):
        super().__init__()
        self.enc1 = _double_conv(in_channels, base)
        self.enc2 = _double_conv(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            _double_conv(base * 2, base * 4),
            SpectralConv2d(base * 4, base * 4, modes1=12, modes2=12),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _double_conv(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# --------------------------------------------------------------------------- #
# Differentiable optical model
# --------------------------------------------------------------------------- #
class DiffOptModel(nn.Module):
    """A learnable optical element followed by angular-spectrum propagation.

    Input/output are 2-channel real tensors ``[B, 2, H, W]`` encoding
    ``(real, imag)`` of the complex field. The propagation distance and a
    per-pixel phase mask are learnable, so the whole model is differentiable
    end-to-end — the basis for gradient-based (adjoint) inverse design.
    """

    def __init__(
        self,
        wavelength: float = 1550e-9,
        dx: float = 0.5e-6,
        distance: float = 100e-6,
        grid: int = 256,
    ):
        super().__init__()
        self.propagator = AngularSpectrum2d(
            wavelength=wavelength, dx=dx, distance=distance, learnable_distance=True
        )
        # Learnable phase mask (a designable optical element / metasurface proxy).
        self.phase_mask = nn.Parameter(torch.zeros(grid, grid, dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != 2:
            raise ValueError("DiffOptModel expects 2-channel (real, imag) input")
        field = torch.complex(x[:, 0].double(), x[:, 1].double())
        h, w = field.shape[-2], field.shape[-1]
        mask = self.phase_mask[:h, :w]
        field = field * torch.exp(1j * mask)
        out = self.propagator(field)
        return torch.stack([out.real, out.imag], dim=1).to(x.dtype)


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
class ModelSelector:
    def __init__(self, data_shape: tuple, data_type: str):
        self.data_shape = data_shape
        self.data_type = data_type
        self.model_choice: Optional[str] = None
        self.priority = "accuracy"
        self.enforce_physics = True
        self.available_models = {
            "fno": "Fourier Neural Operator",
            "unet": "U-Net with spectral bottleneck",
            "diffopt": "Differentiable Optical Model",
            "convnet": "Traditional CNN",
        }

    # --- channel bookkeeping -------------------------------------------------
    def _channels(self) -> tuple[int, int]:
        """Return (in_channels, out_channels) for the current data type.

        Complex fields are represented as 2 real channels (real, imag).
        """
        if self.data_type == "complex_field":
            return 2, 2
        if self.data_type in ("psf", "intensity"):
            return 1, 1
        return 1, 2

    def get_user_preferences(self):
        """Interactive selection (CLI only — never called from library code)."""
        print("\n[Model Selection]")
        for key, value in self.available_models.items():
            print(f"{key}: {value}")
        self.model_choice = input("Select model (fno/unet/diffopt/convnet): ").lower()
        self.priority = input("Optimize for (accuracy/speed/interpretability): ").lower()
        self.enforce_physics = input("Enforce physical constraints? (y/n): ").lower() == "y"

    def build_model(self) -> nn.Module:
        choice = self.model_choice or self.auto_recommend()
        self.model_choice = choice
        if choice == "fno":
            return self._build_fno()
        if choice == "unet":
            return self._build_unet()
        if choice == "diffopt":
            return self._build_diffopt()
        return self._build_convnet()

    def _build_fno(self) -> nn.Module:
        in_ch, out_ch = self._channels()
        return FNO2d(in_channels=in_ch, out_channels=out_ch)

    def _build_unet(self) -> nn.Module:
        in_ch, out_ch = self._channels()
        return SpectralUNet(in_channels=in_ch, out_channels=out_ch)

    def _build_diffopt(self) -> nn.Module:
        grid = int(self.data_shape[-1]) if self.data_shape else 256
        return DiffOptModel(grid=grid)

    def _build_convnet(self) -> nn.Module:
        in_ch, out_ch = self._channels()
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def auto_recommend(self) -> str:
        if self.data_type == "complex_field":
            return "fno" if self.data_shape[0] > 256 else "unet"
        if self.data_type == "psf":
            return "diffopt"
        return "convnet"


if __name__ == "__main__":
    selector = ModelSelector(data_shape=(256, 256), data_type="complex_field")
    print("Recommended:", selector.available_models[selector.auto_recommend()])
    model = selector.build_model()
    x = torch.randn(2, 2, 128, 128)  # note: trained-mode-count is resolution-invariant
    print("output shape:", tuple(model(x).shape))
