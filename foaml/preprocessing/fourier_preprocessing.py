"""Preprocessing for Fourier-optics fields.

Fixes vs. the previous ``preprocessing.py``:

  * ``scipy.signal.tukey`` moved to ``scipy.signal.windows.tukey`` in modern
    SciPy (the old path raises ``AttributeError``). Imported from the correct
    location.
  * units are now consistently SI internally (metres); the public constructor
    still accepts pixel size in micrometres and wavelength in nanometres.
  * the 2D window is a proper separable outer product over *both* axes rather
    than assuming a square grid.
  * phase unwrapping moves through NumPy safely and returns a torch tensor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.fft as fft
from scipy.signal.windows import hamming, tukey


class FourierPreprocessing:
    def __init__(self, pixel_size: float, wavelength: float):
        """
        Parameters
        ----------
        pixel_size : float
            Pixel pitch in micrometres.
        wavelength : float
            Wavelength in nanometres.
        """
        self.pixel_size_m = pixel_size * 1e-6
        self.wavelength_m = wavelength * 1e-9

    def dc_removal(self, data: torch.Tensor) -> torch.Tensor:
        return data - torch.mean(data)

    def apply_window(self, data: torch.Tensor, window_type: str = "tukey", alpha: float = 0.25) -> torch.Tensor:
        ny, nx = data.shape[-2], data.shape[-1]
        if window_type == "tukey":
            wy = tukey(ny, alpha=alpha)
            wx = tukey(nx, alpha=alpha)
        elif window_type == "hamming":
            wy, wx = hamming(ny), hamming(nx)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
        window = np.outer(wy, wx)
        w = torch.from_numpy(window).to(device=data.device, dtype=torch.float64)
        return data * w.to(data.dtype)

    def fft2(self, data: torch.Tensor) -> torch.Tensor:
        return fft.fftshift(fft.fft2(fft.ifftshift(data)))

    def ifft2(self, data: torch.Tensor) -> torch.Tensor:
        return fft.fftshift(fft.ifft2(fft.ifftshift(data)))

    def calculate_spatial_frequencies(self, shape: tuple):
        ny, nx = shape
        dx = self.pixel_size_m
        fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
        return np.meshgrid(fx, fy)

    def phase_unwrap(self, wrapped_phase: torch.Tensor) -> torch.Tensor:
        arr = wrapped_phase.detach().cpu().numpy()
        unwrapped = np.unwrap(np.unwrap(arr, axis=0), axis=1)
        return torch.from_numpy(unwrapped).to(device=wrapped_phase.device, dtype=wrapped_phase.dtype)

    def preprocess(self, data: torch.Tensor, remove_dc: bool = True, window_type: str = "tukey", unwrap_phase: bool = False):
        if remove_dc:
            data = self.dc_removal(data)
        data = self.apply_window(data, window_type)
        if torch.is_complex(data):
            amplitude = torch.abs(data)
            phase = torch.angle(data)
            if unwrap_phase:
                phase = self.phase_unwrap(phase)
            return amplitude, phase
        return data


if __name__ == "__main__":
    sample = torch.randn(256, 256) + 1j * torch.randn(256, 256)
    pre = FourierPreprocessing(pixel_size=5, wavelength=632.8)
    amp, phase = pre.preprocess(sample, unwrap_phase=True)
    print("amplitude", tuple(amp.shape), "phase", tuple(phase.shape))
