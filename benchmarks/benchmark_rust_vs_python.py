import numpy as np
import time
from fourierlab.core.rust_bindings import WavePropagator, PatternGenerator, PhaseMaskOptimizer

def benchmark_wave_propagation():
    """Benchmark wave propagation methods."""
    print("\nWave Propagation Benchmark")
    print("-" * 50)
    
    # Initialize propagator
    propagator = WavePropagator(wavelength=632.8e-9, pixel_size=10e-6)
    
    # Test sizes
    sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024)]
    batch_size = 4
    
    print(f"{'Size':<10} {'Method':<20} {'Single (ms)':<15} {'Batch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        # Create test field
        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        X, Y = np.meshgrid(x, y)
        field = np.exp(-(X**2 + Y**2) / 0.1)
        fields = np.stack([field] * batch_size)
        
        # Test each method
        methods = [
            ("Angular Spectrum", propagator.angular_spectrum, propagator.angular_spectrum_batch),
            ("Rayleigh-Sommerfeld", propagator.rayleigh_sommerfeld, propagator.rayleigh_sommerfeld_batch),
            ("Fresnel", propagator.fresnel, propagator.fresnel_batch),
        ]
        
        for name, single_fn, batch_fn in methods:
            # Single field
            start = time.time()
            single_fn(field, 0.1)
            single_time = (time.time() - start) * 1000
            
            # Batch processing
            start = time.time()
            batch_fn(fields, 0.1)
            batch_time = (time.time() - start) * 1000
            
            # Calculate speedup
            speedup = (single_time * batch_size) / batch_time
            
            print(f"{size[0]:<10} {name:<20} {single_time:>8.2f} ms    {batch_time:>8.2f} ms    {speedup:>8.2f}x")

def benchmark_pattern_generation():
    """Benchmark pattern generation methods."""
    print("\nPattern Generation Benchmark")
    print("-" * 50)
    
    # Initialize generator
    generator = PatternGenerator(wavelength=632.8e-9, pixel_size=10e-6)
    
    # Test sizes
    sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024)]
    batch_size = 4
    
    print(f"{'Size':<10} {'Pattern Type':<20} {'Single (ms)':<15} {'Batch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        # Test each pattern type
        pattern_types = [
            ("Cross", "cross", None),
            ("Circle", "circle", None),
            ("Square", "square", None),
            ("Grating", "grating", None),
            ("Spiral", "spiral", None),
            ("Vortex", "vortex", 1),
            ("Bessel", "bessel", None),
            ("Hermite", "hermite", 1),
            ("Laguerre", "laguerre", 1),
        ]
        
        for name, pattern_type, order in pattern_types:
            # Single pattern
            start = time.time()
            generator.generate_pattern(pattern_type, size, 0.1, 10.0, order)
            single_time = (time.time() - start) * 1000
            
            # Batch processing
            start = time.time()
            generator.generate_pattern_batch(pattern_type, size, 0.1, 10.0, order, batch_size)
            batch_time = (time.time() - start) * 1000
            
            # Calculate speedup
            speedup = (single_time * batch_size) / batch_time
            
            print(f"{size[0]:<10} {name:<20} {single_time:>8.2f} ms    {batch_time:>8.2f} ms    {speedup:>8.2f}x")

def benchmark_phase_mask_optimization():
    """Benchmark phase mask optimization methods."""
    print("\nPhase Mask Optimization Benchmark")
    print("-" * 50)
    
    # Initialize optimizer
    optimizer = PhaseMaskOptimizer(wavelength=632.8e-9, pixel_size=10e-6)
    
    # Test sizes
    sizes = [(64, 64), (256, 256), (512, 512)]
    batch_size = 4
    iterations = [50, 100, 200]
    
    print(f"{'Size':<10} {'Iterations':<10} {'Single (ms)':<15} {'Batch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        # Create test target
        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        X, Y = np.meshgrid(x, y)
        target = np.exp(-(X**2 + Y**2) / 0.1)
        targets = np.stack([target] * batch_size)
        
        for max_iter in iterations:
            # Update optimizer
            optimizer = PhaseMaskOptimizer(wavelength=632.8e-9, pixel_size=10e-6, max_iterations=max_iter)
            
            # Single optimization
            start = time.time()
            optimizer.optimize_phase_mask(target, distance=0.1)
            single_time = (time.time() - start) * 1000
            
            # Batch optimization
            start = time.time()
            optimizer.optimize_phase_mask_batch(targets, distance=0.1)
            batch_time = (time.time() - start) * 1000
            
            # Calculate speedup
            speedup = (single_time * batch_size) / batch_time
            
            print(f"{size[0]:<10} {max_iter:<10} {single_time:>8.2f} ms    {batch_time:>8.2f} ms    {speedup:>8.2f}x")

def main():
    """Run all benchmarks."""
    print("FourierLab Performance Benchmarks")
    print("=" * 50)
    
    benchmark_wave_propagation()
    benchmark_pattern_generation()
    benchmark_phase_mask_optimization()

if __name__ == "__main__":
    main() 