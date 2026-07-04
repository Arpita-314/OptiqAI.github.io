"""Fourier-optics quality metrics.

The previous ``strehl_ratio`` computed ``max(intensity) / sum(intensity)`` — a
peak-to-total energy fraction, which is *not* the Strehl ratio. It is replaced
by the correct definition:

    S = | (1/A) integral_pupil A(x) exp(i phi(x)) dx |^2  /  | (1/A) integral_pupil A(x) dx |^2

i.e. the normalised on-axis PSF intensity of the aberrated pupil relative to the
unaberrated one. The Maréchal approximation ``S ~= exp(-sigma_phi^2)`` is
provided separately for small aberrations.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class FourierOpticsMetrics:
    def __init__(self, wavelength: float, pixel_size: float):
        self.wavelength = wavelength  # metres
        self.pixel_size = pixel_size  # metres

    def strehl_ratio(self, wavefront: torch.Tensor) -> float:
        """Diffraction Strehl ratio of a (possibly aberrated) pupil field."""
        amp = torch.abs(wavefront)
        phase = torch.angle(wavefront)
        coherent = torch.sum(amp * torch.exp(1j * phase))
        incoherent = torch.sum(amp)
        s = (torch.abs(coherent) ** 2) / (incoherent**2 + 1e-12)
        return float(s.item())

    def strehl_marechal(self, wavefront: torch.Tensor) -> float:
        """Maréchal small-aberration estimate ``exp(-sigma_phi^2)``."""
        phase = torch.angle(wavefront)
        # remove piston; wrap residual into (-pi, pi]
        residual = phase - torch.mean(phase)
        residual = torch.atan2(torch.sin(residual), torch.cos(residual))
        sigma2 = torch.var(residual, unbiased=False)
        return float(torch.exp(-sigma2).item())

    def mtf(self, psf: torch.Tensor) -> np.ndarray:
        """Modulation Transfer Function, normalised by the DC (zero-frequency) term."""
        otf = torch.fft.fftshift(torch.fft.fft2(psf))
        mtf = torch.abs(otf)
        center = mtf[mtf.shape[-2] // 2, mtf.shape[-1] // 2]
        return (mtf / (center + 1e-12)).cpu().numpy()

    def wavefront_error_stats(self, wavefront: torch.Tensor) -> Dict[str, float]:
        phase = torch.angle(wavefront)
        phase = phase - torch.mean(phase)
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))  # unwrap into (-pi, pi]
        return {
            "RMS_error": float(torch.std(phase).item()),
            "PV_error": float((torch.max(phase) - torch.min(phase)).item()),
        }

    def calculate_all_metrics(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["strehl_ratio"] = self.strehl_ratio(predicted)
        metrics["strehl_marechal"] = self.strehl_marechal(predicted)

        pred_mtf = self.mtf(torch.abs(predicted) ** 2)
        target_mtf = self.mtf(torch.abs(target) ** 2)
        metrics["mtf_correlation"] = float(
            np.corrcoef(pred_mtf.flatten(), target_mtf.flatten())[0, 1]
        )

        metrics.update(self.wavefront_error_stats(predicted / (target + 1e-12)))

        dphi = torch.angle(predicted) - torch.angle(target)
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        metrics["phase_rmse"] = float(torch.sqrt(torch.mean(dphi**2)).item())
        return metrics


if __name__ == "__main__":
    wavelength, pixel_size = 632.8e-9, 5e-6
    # An unaberrated flat wavefront must have Strehl ~ 1.
    flat = torch.ones(256, 256, dtype=torch.complex64)
    calc = FourierOpticsMetrics(wavelength, pixel_size)
    print("Strehl (flat):", calc.strehl_ratio(flat))
    aberrated = torch.exp(1j * 0.5 * torch.randn(256, 256))
    print("Strehl (aberrated):", calc.strehl_ratio(aberrated))
    print("Maréchal (aberrated):", calc.strehl_marechal(aberrated))
