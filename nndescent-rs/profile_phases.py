#!/usr/bin/env python
"""
Profile NNDescent phase-by-phase to identify remaining bottlenecks.
"""

import numpy as np
import time
import pynndescent_rs
from pynndescent import NNDescent


def profile_pynndescent(data, n_neighbors=30, n_iters=8):
    """Profile PyNNDescent with verbose timing."""
    print("=== PyNNDescent Profiling ===")

    # JIT warmup with same-sized data
    print("JIT warmup...")
    warmup_data = np.random.rand(data.shape[0], data.shape[1]).astype(np.float32)
    _ = NNDescent(
        warmup_data, n_neighbors=n_neighbors, n_jobs=8, low_memory=False, verbose=False
    )
    print("Warmup complete.\n")

    start = time.perf_counter()
    index = NNDescent(
        data,
        n_neighbors=n_neighbors,
        n_jobs=8,
        low_memory=False,  # Fair comparison
        verbose=True,
    )
    total = time.perf_counter() - start
    print(f"Total: {total*1000:.2f}ms")
    return index


def profile_rust(data, n_neighbors=30):
    """Profile Rust implementation with verbose timing."""
    print("\n=== Rust Profiling ===")

    start = time.perf_counter()
    index = pynndescent_rs.NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        verbose=True,  # Enable verbose output
    )
    total = time.perf_counter() - start
    print(f"Total: {total*1000:.2f}ms")
    return index


def main():
    print("=" * 70)
    print("NNDescent Phase Profiling")
    print("=" * 70)
    print()

    # Test with 10k points
    n = 10000
    dim = 128
    np.random.seed(42)
    data = np.random.rand(n, dim).astype(np.float32)

    print(f"Dataset: {n} × {dim}")
    print()

    # Profile PyNNDescent
    py_index = profile_pynndescent(data)

    # Profile Rust
    rust_index = profile_rust(data)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
