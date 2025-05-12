# FourierLab: GPU and CPU Modes

FourierLab provides both GPU-accelerated and CPU-based wave propagation. This tutorial explains how to use both modes effectively.

## Device Modes

FourierLab supports three device modes:

1. **Auto Mode** (Default)
   - Automatically selects GPU if available, falls back to CPU if not
   - Best for most users
   - Usage: `mode='auto'`

2. **GPU Mode**
   - Uses NVIDIA GPU for acceleration
   - Provides advanced features and better performance
   - Usage: `mode='gpu'`

3. **CPU Mode**
   - Uses CPU for computation
   - More accessible, works on any computer
   - Usage: `mode='cpu'`

## GPU Mode Features

### Prerequisites

1. **NVIDIA GPU with CUDA Support**
   - Check if you have an NVIDIA GPU:
     ```bash
     nvidia-smi
     ```
   - If not installed, download and install NVIDIA drivers from:
     https://www.nvidia.com/Download/index.aspx

2. **CUDA Toolkit 12.x**
   - Download and install from:
     https://developer.nvidia.com/cuda-downloads
   - After installation, set environment variables:
     - CUDA_PATH: Path to CUDA installation
     - Add to PATH:
       - %CUDA_PATH%\bin
       - %CUDA_PATH%\libnvvp

3. **Python Packages**
   ```bash
   pip install cupy-cuda12x pycuda
   ```

### GPU Features

- Maximum field size: 8192x8192
- Batch processing support
- Mixed precision computation
- Multi-GPU support
- Advanced features (real-time visualization, etc.)
- High optimization level

### Recommended GPU Settings

- Field size: 2048x2048
- Batch size: 32
- Precision: float32
- Memory limit: 80% of available GPU memory
- Optimization level: high

## CPU Mode Features

### Prerequisites

1. **Python Packages**
   ```bash
   pip install mkl openblas
   ```

### CPU Features

- Maximum field size: 2048x2048
- Single-field processing
- Double precision computation
- Basic features
- Medium optimization level

### Recommended CPU Settings

- Field size: 1024x1024
- Batch size: 1
- Precision: float64
- Memory limit: 50% of available RAM
- Optimization level: medium

## Using the Propagator

### Basic Usage

```python
from fourierlab.core.propagator_factory import PropagatorFactory

# Create propagator (auto mode)
propagator = PropagatorFactory.create_propagator()

# Or specify mode
propagator = PropagatorFactory.create_propagator(mode='gpu')  # or 'cpu'

# Get available features
features = PropagatorFactory.get_available_features()

# Get recommended settings
settings = PropagatorFactory.get_recommended_settings()
```

### Available Methods

1. **Angular Spectrum Method**
   ```python
   result = propagator.angular_spectrum(field, wavelength, distance)
   ```

2. **Rayleigh-Sommerfeld Diffraction**
   ```python
   result = propagator.rayleigh_sommerfeld(field, wavelength, distance)
   ```

3. **Fresnel Diffraction**
   ```python
   result = propagator.fresnel(field, wavelength, distance, dx, dy)
   ```

## Performance Optimization

### GPU Mode

1. **Memory Management**
   ```python
   with propagator:  # Context manager for automatic cleanup
       result = propagator.angular_spectrum(field, wavelength, distance)
   ```

2. **Mixed Precision**
   ```python
   field = field.astype(np.float32)  # Use float32 for faster computation
   result = propagator.angular_spectrum(field, wavelength, distance)
   ```

### CPU Mode

1. **Memory Management**
   ```python
   # CPU mode automatically manages memory
   result = propagator.angular_spectrum(field, wavelength, distance)
   ```

2. **Precision Control**
   ```python
   field = field.astype(np.float64)  # Use float64 for higher accuracy
   result = propagator.angular_spectrum(field, wavelength, distance)
   ```

## Troubleshooting

### GPU Mode Issues

1. **CUDA Driver Version Insufficient**
   - Update NVIDIA drivers
   - Download from: https://www.nvidia.com/Download/index.aspx

2. **Out of Memory**
   - Reduce field size
   - Use mixed precision
   - Clear GPU memory:
     ```python
     import cupy as cp
     cp.get_default_memory_pool().free_all_blocks()
     ```

### CPU Mode Issues

1. **Slow Performance**
   - Reduce field size
   - Use recommended settings
   - Ensure MKL/OpenBLAS is installed

2. **Memory Issues**
   - Reduce field size
   - Close other applications
   - Use recommended memory limit

## Benchmarking

Run the benchmark script to compare modes:
```bash
python tests/benchmark_gpu.py
```

Expected output:
```
Propagation Benchmark Results:
Field size: 2048x2048
GPU time: 0.21s
CPU time: 1.31s
Speedup: 6.2x
```

## Best Practices

1. Use auto mode for best compatibility
2. Monitor memory usage
3. Choose appropriate field size
4. Use recommended settings
5. Keep drivers and libraries updated
6. Profile your code to identify bottlenecks 