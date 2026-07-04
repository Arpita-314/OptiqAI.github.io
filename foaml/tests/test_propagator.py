import numpy as np
import pytest
import torch

from foaml_2.propagator_1d import angular_spectrum_1d, gaussian_beam_waist
from optics.propagator import AngularSpectrum2d


def _gaussian_1d(n, dx, w0):
    x = (np.arange(n) - n / 2) * dx
    return x, np.exp(-(x**2) / w0**2)


def test_energy_conserved_for_propagating_field():
    """A well-sampled band-limited beam conserves power under ASM (Parseval)."""
    wavelength, dx, n, w0 = 500e-9, 0.5e-6, 1024, 5e-6
    _, field = _gaussian_1d(n, dx, w0)
    out = angular_spectrum_1d(field, wavelength, 200e-6, dx)
    e_in = np.sum(np.abs(field) ** 2)
    e_out = np.sum(np.abs(out) ** 2)
    assert np.isclose(e_in, e_out, rtol=2e-3)


def test_no_nans_with_evanescent_content():
    """Oversampled grid produces evanescent modes; result must decay, not NaN."""
    wavelength, dx, n = 500e-9, 20e-9, 512  # dx << lambda/2 -> evanescent content
    field = np.zeros(n, dtype=complex)
    field[n // 2] = 1.0  # point source: broad spectrum incl. evanescent
    out = angular_spectrum_1d(field, wavelength, 1e-3, dx)
    assert np.all(np.isfinite(out))


def test_gaussian_beam_spreads_by_analytic_law():
    """Propagated 1/e^2 waist must match the paraxial Gaussian-beam formula."""
    wavelength, dx, n, w0 = 633e-9, 0.5e-6, 4096, 8e-6
    x, field = _gaussian_1d(n, dx, w0)
    z = 300e-6
    out = angular_spectrum_1d(field, wavelength, z, dx)
    intensity = np.abs(out) ** 2
    # measured 1/e^2 amplitude waist == 2*sigma of the intensity Gaussian
    sigma = np.sqrt(np.sum(intensity * x**2) / np.sum(intensity))
    measured_w = 2.0 * sigma
    analytic_w = gaussian_beam_waist(w0, wavelength, z)
    assert measured_w == pytest.approx(analytic_w, rel=0.05)


def test_round_trip_back_propagation():
    """Propagating +z then -z returns (approximately) the original field."""
    wavelength, dx, n, w0 = 633e-9, 0.5e-6, 2048, 8e-6
    _, field = _gaussian_1d(n, dx, w0)
    fwd = angular_spectrum_1d(field, wavelength, 150e-6, dx, bandlimit=False)
    back = angular_spectrum_1d(fwd, wavelength, -150e-6, dx, bandlimit=False)
    assert np.allclose(field, back.real, atol=1e-6)


def test_torch_propagator_is_differentiable_in_distance():
    """Gradient must flow to the propagation distance (the adjoint hook)."""
    prop = AngularSpectrum2d(wavelength=1550e-9, dx=0.5e-6, distance=50e-6, learnable_distance=True)
    field = torch.zeros(1, 64, 64, dtype=torch.complex128)
    field[0, 32, 32] = 1.0
    out = prop(field)
    loss = (out.abs() ** 2).sum()
    loss.backward()
    assert prop.distance.grad is not None
    assert torch.isfinite(prop.distance.grad).all()
