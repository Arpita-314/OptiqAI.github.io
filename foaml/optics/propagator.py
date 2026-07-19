"""Differentiable 2D Angular Spectrum propagator (PyTorch).

Why this exists: a surrogate/design tool is only useful for *inverse design* if
you can backprop through the forward model. This module is a physically correct,
autograd-friendly propagator. Because it is differentiable w.r.t. both the input
field and the propagation distance, the adjoint gradient used by gradient-based
inverse design comes for free from ``loss.backward()`` — no separate adjoint
solve, no finite differences.

The math matches :func:`foaml.foaml_2.propagator_1d.angular_spectrum_1d`, lifted
to 2D:  ``H(fx, fy) = exp(i * 2*pi/lambda * sqrt(1 - (lambda fx)^2 - (lambda fy)^2) * z)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _transfer_function(
    ny: int,
    nx: int,
    wavelength: float,
    dx: float,
    dy: float,
    distance: torch.Tensor,
    *,
    bandlimit: bool = True,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Build the complex ASM transfer function on the given grid.

    ``distance`` is a tensor so gradients can flow to it (learnable focus).
    """
    fx = torch.fft.fftfreq(nx, d=dx, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(ny, d=dy, device=device, dtype=dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    alpha2 = (wavelength * FX) ** 2 + (wavelength * FY) ** 2
    # sqrt(1 - alpha2) in the complex plane -> evanescent decay for alpha2 > 1.
    kz = (2.0 * math.pi / wavelength) * torch.sqrt(
        torch.clamp(1.0 - alpha2, min=-1e30).to(torch.complex128)
    )
    H = torch.exp(1j * kz * distance.to(torch.complex128))

    if bandlimit:
        du = 1.0 / (nx * dx)
        dv = 1.0 / (ny * dy)
        z = float(distance.detach().abs().item()) if distance.numel() == 1 else float(distance.detach().abs().max())
        fx_lim = 1.0 / (wavelength * math.sqrt((2.0 * du * z) ** 2 + 1.0))
        fy_lim = 1.0 / (wavelength * math.sqrt((2.0 * dv * z) ** 2 + 1.0))
        mask = (FX.abs() <= fx_lim) & (FY.abs() <= fy_lim)
        H = H * mask.to(H.dtype)
    return H


class AngularSpectrum2d(nn.Module):
    """Differentiable free-space propagator over a fixed distance.

    Parameters
    ----------
    wavelength, dx, dy : float
        Wavelength and pixel pitch, all in metres.
    distance : float
        Initial propagation distance in metres.
    learnable_distance : bool
        If True, ``distance`` becomes an optimisable ``nn.Parameter`` (a simple
        differentiable "autofocus" / learnable optical element).
    """

    def __init__(
        self,
        wavelength: float,
        dx: float,
        distance: float,
        dy: float | None = None,
        *,
        learnable_distance: bool = False,
        bandlimit: bool = True,
    ):
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy if dy is not None else dx)
        self.bandlimit = bandlimit
        dist = torch.tensor(float(distance), dtype=torch.float64)
        if learnable_distance:
            self.distance = nn.Parameter(dist)
        else:
            self.register_buffer("distance", dist)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Propagate ``field`` (complex, shape ``[..., H, W]``)."""
        if not torch.is_complex(field):
            field = field.to(torch.complex128)
        ny, nx = field.shape[-2], field.shape[-1]
        H = _transfer_function(
            ny, nx, self.wavelength, self.dx, self.dy, self.distance,
            bandlimit=self.bandlimit, device=field.device,
        )
        spectrum = torch.fft.fft2(field)
        return torch.fft.ifft2(spectrum * H)
