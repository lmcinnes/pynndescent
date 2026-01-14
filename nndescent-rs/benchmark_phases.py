#!/usr/bin/env python
"""
Phase-level profiling to compare timing of each major phase between
PyNNDescent and Rust implementation.

This helps identify which phases are slower in Rust.
"""

import time
import numpy as np


def generate_data(n, dim, seed=42):
    """Generate random test data."""
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def profile_pynndescent(data, n_neighbors):
    """Profile PyNNDescent with verbose timing output."""
    from pynndescent import NNDescent

    print("PyNNDescent phases:")
    print("-" * 50)

    # Run with verbose=True to get internal timing
    start = time.perf_counter()
    index = NNDescent(data, n_neighbors=n_neighbors, verbose=True, low_memory=False)
    # Force graph computation
    _ = index.neighbor_graph
    total = time.perf_counter() - start

    print(f"\nTotal wall time: {total*1000:.1f}ms")
    return total


def profile_rust(data, n_neighbors):
    """Profile Rust implementation with verbose timing output."""
    from pynndescent_rs import NNDescent

    print("\nRust phases:")
    print("-" * 50)

    start = time.perf_counter()
    index = NNDescent(data, n_neighbors=n_neighbors, verbose=True)
    _ = index.neighbor_graph
    total = time.perf_counter() - start

    print(f"\nTotal wall time: {total*1000:.1f}ms")
    return total


def main():
    print("=" * 60)
    print("Phase Profiling: PyNNDescent vs Rust")
    print("=" * 60)

    # Test on different sizes
    configs = [
        (5000, 100, "5k x 100"),
        (10000, 128, "10k x 128"),
        (50000, 128, "50k x 128"),
    ]

    n_neighbors = 30

    for n, dim, name in configs:
        print(f"\n{'='*60}")
        print(f"Dataset: {name} (n_neighbors={n_neighbors})")
        print("=" * 60)

        data = generate_data(n, dim)

        # Warmup PyNNDescent JIT
        print("\nWarming up PyNNDescent JIT...")
        warmup = generate_data(500, 50)
        from pynndescent import NNDescent as PyNND

        _ = PyNND(warmup, n_neighbors=10, verbose=False, low_memory=False)
        print("Done.\n")

        py_time = profile_pynndescent(data, n_neighbors)
        rs_time = profile_rust(data, n_neighbors)

        print(f"\n{'='*60}")
        print(f"Summary for {name}:")
        print(f"  PyNNDescent: {py_time*1000:.1f}ms")
        print(f"  Rust:        {rs_time*1000:.1f}ms")
        print(f"  Speedup:     {py_time/rs_time:.2f}x")
        print("=" * 60)


if __name__ == "__main__":
    main()
