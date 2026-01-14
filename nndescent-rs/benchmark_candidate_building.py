#!/usr/bin/env python
"""
Benchmark candidate building in isolation: Rust vs PyNNDescent (Numba).
"""

import numpy as np
import time
import numba
from pynndescent.utils import new_build_candidates, tau_rand_int, checked_heap_push


@numba.njit(cache=True)
def create_random_graph(n_vertices, k, seed):
    """Create a random graph for testing."""
    rng_state = np.array([seed, seed + 1, seed + 2], dtype=np.int64)

    indices = np.zeros((n_vertices, k), dtype=np.int32)
    distances = np.zeros((n_vertices, k), dtype=np.float32)
    flags = np.ones((n_vertices, k), dtype=np.uint8)  # All new initially

    for i in range(n_vertices):
        for j in range(k):
            # Random neighbor (not self)
            neighbor = tau_rand_int(rng_state) % n_vertices
            while neighbor == i:
                neighbor = tau_rand_int(rng_state) % n_vertices
            indices[i, j] = neighbor
            distances[i, j] = np.float32(tau_rand_int(rng_state) % 1000) / 1000.0
            # ~50% are new
            flags[i, j] = 1 if tau_rand_int(rng_state) % 2 == 0 else 0

    return (indices, distances, flags)


def benchmark_pynndescent_candidates(
    n_vertices, k, max_candidates, n_iters, n_threads=1
):
    """Benchmark PyNNDescent candidate building."""

    # Create random graph
    graph = create_random_graph(n_vertices, k, 42)

    # Warmup
    rng_state = np.array([1, 2, 3], dtype=np.int64)
    _ = new_build_candidates(graph, max_candidates, rng_state, n_threads)

    # Benchmark
    times = []
    for _ in range(n_iters):
        # Reset flags to simulate iteration
        graph[2][:] = np.where(np.random.random(graph[2].shape) < 0.5, 1, 0).astype(
            np.uint8
        )

        rng_state = np.array(
            [
                np.random.randint(0, 2**31),
                np.random.randint(0, 2**31),
                np.random.randint(0, 2**31),
            ],
            dtype=np.int64,
        )

        start = time.perf_counter()
        new_cands, old_cands = new_build_candidates(
            graph, max_candidates, rng_state, n_threads
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_rust_candidates(n_vertices, k, max_candidates, n_iters):
    """Benchmark Rust candidate building."""
    try:
        import pynndescent_rs

        # Create random graph
        graph = create_random_graph(n_vertices, k, 42)
        indices = graph[0]

        # Benchmark
        times = []
        for iter_idx in range(n_iters):
            # Reset flags to simulate iteration
            flags = np.where(np.random.random((n_vertices, k)) < 0.5, 1, 0).astype(
                np.uint8
            )

            elapsed = pynndescent_rs.benchmark_candidate_building(
                indices, flags, max_candidates, iter_idx
            )
            times.append(elapsed)

        return np.mean(times), np.std(times)
    except (ImportError, AttributeError) as e:
        print(f"Rust candidate benchmark not available: {e}")
        return None, None


def main():
    print("=" * 80)
    print("Candidate Building Benchmark: Rust vs Numba")
    print("=" * 80)
    print()

    # Test configurations
    configs = [
        {"n_vertices": 1000, "k": 30, "max_candidates": 60, "n_iters": 10},
        {"n_vertices": 5000, "k": 30, "max_candidates": 60, "n_iters": 10},
        {"n_vertices": 10000, "k": 30, "max_candidates": 60, "n_iters": 10},
        {"n_vertices": 50000, "k": 30, "max_candidates": 60, "n_iters": 5},
    ]

    print(
        f"{'Config':<20} {'Numba 1T (ms)':<15} {'Numba 8T (ms)':<15} {'Rust (ms)':<15} {'vs 1T':<10} {'vs 8T':<10}"
    )
    print("-" * 80)

    for cfg in configs:
        n_vertices = cfg["n_vertices"]
        k = cfg["k"]
        max_candidates = cfg["max_candidates"]
        n_iters = cfg["n_iters"]

        config_str = f"{n_vertices}×{k}"

        # Benchmark Numba single-threaded
        numba_1t_mean, _ = benchmark_pynndescent_candidates(
            n_vertices, k, max_candidates, n_iters, n_threads=1
        )

        # Benchmark Numba 8-threaded
        numba_8t_mean, _ = benchmark_pynndescent_candidates(
            n_vertices, k, max_candidates, n_iters, n_threads=8
        )

        # Benchmark Rust
        rust_mean, _ = benchmark_rust_candidates(n_vertices, k, max_candidates, n_iters)

        if rust_mean is not None:
            speedup_1t = numba_1t_mean / rust_mean
            speedup_8t = numba_8t_mean / rust_mean
            print(
                f"{config_str:<20} {numba_1t_mean*1000:>13.2f}ms {numba_8t_mean*1000:>13.2f}ms {rust_mean*1000:>13.2f}ms {speedup_1t:>8.2f}x {speedup_8t:>8.2f}x"
            )
        else:
            print(
                f"{config_str:<20} {numba_1t_mean*1000:>13.2f}ms {numba_8t_mean*1000:>13.2f}ms {'N/A':>15} {'N/A':>10} {'N/A':>10}"
            )

    print()
    print("Notes:")
    print("  - Numba 1T: Single-threaded Numba (baseline for comparison)")
    print("  - Numba 8T: 8-threaded Numba (default in PyNNDescent)")
    print("  - Rust: Sequential Rust implementation")
    print("  - vs 1T: Rust speedup vs single-threaded Numba")
    print("  - vs 8T: Rust speedup vs 8-threaded Numba")


if __name__ == "__main__":
    main()
