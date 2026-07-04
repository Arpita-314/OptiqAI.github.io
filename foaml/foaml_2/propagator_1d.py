"""Band-limited Angular Spectrum Method (ASM) propagator — 1D.

This replaces the previous implementation, which had three physics bugs that
made it a no-op:

  1. ``np.fft.fftfreq(n)`` was called without the sample spacing ``d=dx``, so the
     spatial frequency axis was in *cycles/sample* (max 0.5) instead of
     *cycles/metre*. Against ``(1/lambda)**2 ~ 4e12`` the ``kx**2`` term was ~13
     orders of magnitude too small, collapsing the transfer function to a
     constant phase. The field never diffracted.
  2. ``np.sqrt`` of a negative *float* returns ``nan``. Evanescent components
     (``|lambda * fx| > 1``) therefore became NaN instead of decaying. Fixed by
     evaluating the square root in the complex plane.
  3. No band-limiting. The ASM aliases at long propagation distances. We apply
     the Matsushima & Shimobaba (2009) band-limit.

References
----------
Goodman, *Introduction to Fourier Optics*, ch. 3-4 (angular spectrum).
Matsushima & Shimobaba, "Band-Limited Angular Spectrum Method...",
Opt. Express 17, 19662 (2009).
"""

from __future__ import annotations

import numpy as np


def angular_spectrum_1d(
    field: np.ndarray,
    wavelength: float,
    distance: float,
    dx: float,
    *,
    bandlimit: bool = True,
) -> np.ndarray:
    """Propagate a 1D scalar wavefield a distance ``z`` via the angular spectrum.

    Parameters
    ----------
    field : np.ndarray
        Complex (or real) input field sampled on a uniform grid.
    wavelength : float
        Wavelength in metres.
    distance : float
        Propagation distance in metres (may be negative to back-propagate).
    dx : float
        Physical sample spacing in metres. **Required** — this is the parameter
        whose absence broke the previous version.
    bandlimit : bool, optional
        Apply the Matsushima-Shimobaba anti-aliasing band limit (default True).

    Returns
    -------
    np.ndarray
        Complex propagated field, same shape as ``field``.

    Notes
    -----
    The transfer function ``H = exp(i k_z z)`` has ``|H| = 1`` for propagating
    waves and ``|H| < 1`` for evanescent waves, so energy is conserved for a
    field whose angular spectrum lies inside the propagating cone
    (``|lambda * fx| < 1``, i.e. ``dx > lambda/2``) and decays physically for
    evanescent content. That is the correct behaviour, not a bug.
    """
    field = np.asarray(field, dtype=np.complex128)
    n = field.shape[-1]

    fx = np.fft.fftfreq(n, d=dx)  # cycles / metre  <-- d=dx is the fix
    alpha = wavelength * fx  # direction cosine along x

    # k_z = (2*pi/lambda) * sqrt(1 - alpha^2). Complex sqrt => evanescent decay.
    kz = (2.0 * np.pi / wavelength) * np.sqrt(1.0 - alpha**2 + 0j)
    H = np.exp(1j * kz * distance)

    if bandlimit and distance != 0.0:
        # Local band limit that keeps the sampled transfer function faithful.
        du = 1.0 / (n * dx)  # frequency sample spacing
        f_limit = 1.0 / (wavelength * np.sqrt((2.0 * du * distance) ** 2 + 1.0))
        H[np.abs(fx) > f_limit] = 0.0

    return np.fft.ifft(np.fft.fft(field) * H)


def fourier_propagate_1d(
    field: np.ndarray,
    wavelength: float,
    distance: float,
    dx: float | None = None,
) -> np.ndarray:
    """Backwards-compatible wrapper around :func:`angular_spectrum_1d`.

    If ``dx`` is omitted we default to ``wavelength`` (a coarse but physically
    valid grid where every spatial frequency propagates). Real callers should
    pass the true sample spacing.
    """
    if dx is None:
        dx = wavelength
    return angular_spectrum_1d(field, wavelength, distance, dx)


def gaussian_beam_waist(w0: float, wavelength: float, z: float) -> float:
    """Analytic 1/e^2 amplitude waist of a Gaussian beam after distance ``z``."""
    zr = np.pi * w0**2 / wavelength  # Rayleigh range
    return w0 * np.sqrt(1.0 + (z / zr) ** 2)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = 500e-9
    dx = 0.5e-6
    n = 1024
    w0 = 5e-6
    x = (np.arange(n) - n / 2) * dx
    field = np.exp(-(x**2) / w0**2)  # Gaussian amplitude, waist w0

    z = 200e-6
    out = angular_spectrum_1d(field, wavelength, z, dx)

    e_in = np.sum(np.abs(field) ** 2)
    e_out = np.sum(np.abs(out) ** 2)
    print(f"energy in/out: {e_in:.4f} / {e_out:.4f} (ratio {e_out / e_in:.4f})")
    print(
        f"analytic waist at z={z * 1e6:.0f}um: "
        f"{gaussian_beam_waist(w0, wavelength, z) * 1e6:.2f}um"
    )

    plt.plot(x * 1e6, np.abs(field) ** 2, label="input")
    plt.plot(x * 1e6, np.abs(out) ** 2, label=f"z={z * 1e6:.0f}um")
    plt.xlabel("x (um)")
    plt.ylabel("intensity")
    plt.legend()
    plt.title("Angular-spectrum propagation of a Gaussian beam")
    plt.savefig("propagation_1d.png", dpi=120)
