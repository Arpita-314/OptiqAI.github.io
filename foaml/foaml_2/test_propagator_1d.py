import numpy as np

from foaml_2.propagator_1d import angular_spectrum_1d


def test_propagation_conserves_energy():
    """Energy is conserved for a band-limited (propagating-only) field.

    The original test propagated an evanescent-heavy field and only "passed"
    because the buggy propagator was ~identity. A physically correct ASM
    conserves power only for content inside the propagating cone, so we use a
    well-sampled Gaussian (dx > lambda/2).
    """
    wavelength, dx, n, w0 = 500e-9, 0.5e-6, 1024, 5e-6
    x = (np.arange(n) - n / 2) * dx
    field = np.exp(-(x**2) / w0**2)
    propagated = angular_spectrum_1d(field, wavelength, 1e-4, dx)
    assert np.allclose(
        np.sum(np.abs(field) ** 2), np.sum(np.abs(propagated) ** 2), rtol=2e-3
    )
