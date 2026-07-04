# FOAML — physics & pipeline fixes

This branch fixes the correctness gaps in the Fourier-Optics AutoML framework and
makes the pipeline runnable and tested (`python -m pytest` → 18 passing).

## Physics core

- **Angular-spectrum propagator (`foaml_2/propagator_1d.py`)** was effectively a
  no-op. Fixes:
  - `np.fft.fftfreq(n)` → `np.fft.fftfreq(n, d=dx)` (frequencies were in
    cycles/sample, ~13 orders of magnitude too small → transfer function was a
    constant phase; no diffraction).
  - complex square root so evanescent waves **decay** instead of becoming `NaN`.
  - Matsushima–Shimobaba band-limit to suppress aliasing.
  - validated against the analytic Gaussian-beam spreading law and by round-trip
    back-propagation.
- **New differentiable 2D propagator (`optics/propagator.py`)** — autograd-friendly
  angular spectrum with a learnable distance, so the adjoint gradient for inverse
  design comes free from `.backward()`.

## Models (`models/architecture.py`)

- Replaced the fake "Fourier Neural Operator" (a 3×3 conv on raw FFT
  coefficients) with a **real `SpectralConv2d`** (learned complex weights on
  truncated low modes) and a resolution-invariant `FNO2d`.
- Removed the misnamed "energy conservation" normalisation (it rescaled every
  output to unit power regardless of input).
- Fixed `_build_unet` (referenced the wrong `self`, had no `forward`) → a working
  `SpectralUNet`.
- Fixed `_build_diffopt` (`nn.Parameter` inside `nn.Sequential` → `TypeError`) →
  a genuine differentiable optical model with a learnable phase mask + distance.

## Training (`training/trainer.py`)

- `energy_conservation_loss` compared `pred[0]` vs `pred[-1]` — two unrelated
  batch samples. Replaced with per-sample Parseval power consistency.
- `complex_loss` operated on real tensors (degenerate phase term). Fields now use
  a consistent 2-channel `(real, imag)` convention; loss reconstructs the complex
  field and uses a wrap-invariant phase term.
- Removed the used-before-assignment `config` bug, the module-level `load_config`
  crash path, duplicate imports, and stray test/yaml text.

## Metrics (`utils/metrics.py`)

- `strehl_ratio` computed `max/sum` intensity (not the Strehl ratio). Replaced
  with the correct diffraction Strehl `|<A e^{iφ}>|² / <A>²`, plus a Maréchal
  estimate. A flat wavefront now returns S ≈ 1 (regression-tested).

## Preprocessing (`preprocessing/fourier_preprocessing.py`)

- `scipy.signal.tukey` → `scipy.signal.windows.tukey` (moved in modern SciPy).
- consistent SI units internally; proper separable 2D window over both axes.

## Packaging / runnability

- moved `data/data/ingestion.py` → `data/ingestion.py`; renamed
  `preprocessing.py` → `fourier_preprocessing.py` to match imports.
- rewrote `main/main.py`: removed the syntax error and `input()` blocking; it now
  runs non-interactively and demonstrates an FNO **learning the propagation
  operator** (val loss 0.60 → 0.03 in ~8 epochs).
- `conftest.py`, `pytest.ini`, `requirements.txt`, and a physics-validating test
  suite under `tests/`.

## Not fixed here (flagged for the team)

The benchmark numbers in the pitch PDFs (19,000× vs Meep, R²=0.988) are not
reproduced by any code in this repo and differ across documents. They should be
replaced with measured results against a GPU baseline (e.g. Tidy3D) before being
shown to a technical audience.
