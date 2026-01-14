#!/usr/bin/env python
"""
Micro-benchmarks for comparing specific operations between Rust and PyNNDescent.

Focus on:
1. Distance computations
2. Heap operations
3. Candidate building
"""

import time
import numpy as np


def generate_data(n, dim, seed=42):
    """Generate random test data."""
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def benchmark_distance_computation(n_points=10000, dim=128, n_pairs=1000000):
    """Benchmark raw distance computation speed."""
    print(f"\n{'='*60}")
    print(f"Distance Computation: {n_pairs:,} pairs, dim={dim}")
    print("=" * 60)

    data = generate_data(n_points, dim)
    np.random.seed(123)
    pairs_i = np.random.randint(0, n_points, n_pairs).astype(np.int32)
    pairs_j = np.random.randint(0, n_points, n_pairs).astype(np.int32)

    # NumPy baseline (vectorized)
    start = time.perf_counter()
    dists_np = np.sum((data[pairs_i] - data[pairs_j]) ** 2, axis=1)
    np_time = time.perf_counter() - start
    print(f"NumPy (vectorized):      {np_time*1000:.2f}ms")

    # Numba (via pynndescent distance function)
    try:
        from pynndescent.distances import euclidean
        import numba

        @numba.njit(parallel=False, fastmath=True)
        def compute_distances_numba(data, pairs_i, pairs_j):
            n = len(pairs_i)
            result = np.empty(n, dtype=np.float32)
            for k in range(n):
                result[k] = euclidean(data[pairs_i[k]], data[pairs_j[k]])
            return result

        # Warmup
        _ = compute_distances_numba(data[:100], pairs_i[:100], pairs_j[:100])

        start = time.perf_counter()
        dists_numba = compute_distances_numba(data, pairs_i, pairs_j)
        numba_time = time.perf_counter() - start
        print(f"Numba (sequential):      {numba_time*1000:.2f}ms")

    except ImportError:
        numba_time = None
        print("Numba: NOT AVAILABLE")

    # Rust
    try:
        from pynndescent_rs import benchmark_distances

        # Warmup
        _ = benchmark_distances(data[:100], pairs_i[:100], pairs_j[:100])

        start = time.perf_counter()
        dists_rust = benchmark_distances(data, pairs_i, pairs_j)
        rust_time = time.perf_counter() - start
        print(f"Rust (SIMD):             {rust_time*1000:.2f}ms")

        if numba_time:
            print(f"\nRust vs Numba speedup: {numba_time/rust_time:.2f}x")
        print(f"Rust vs NumPy speedup: {np_time/rust_time:.2f}x")

    except (ImportError, AttributeError) as e:
        print(f"Rust: NOT AVAILABLE ({e})")


def benchmark_heap_operations(n_points=50000, k=30, n_pushes=5000000):
    """Benchmark heap push operations."""
    print(f"\n{'='*60}")
    print(f"Heap Operations: {n_pushes:,} pushes into {n_points} x {k} heap")
    print("=" * 60)

    np.random.seed(42)
    # Random pushes: (point, neighbor, distance)
    points = np.random.randint(0, n_points, n_pushes).astype(np.int32)
    neighbors = np.random.randint(0, n_points, n_pushes).astype(np.int32)
    distances = np.random.rand(n_pushes).astype(np.float32) * 10

    # Numba heap
    try:
        from pynndescent.utils import make_heap, checked_flagged_heap_push
        import numba

        @numba.njit(parallel=False)
        def bench_heap_numba(heap, points, neighbors, distances):
            count = 0
            for i in range(len(points)):
                p = points[i]
                n = neighbors[i]
                d = distances[i]
                if checked_flagged_heap_push(
                    heap[1][p], heap[0][p], heap[2][p], d, n, 1
                ):
                    count += 1
            return count

        heap = make_heap(n_points, k)
        # Warmup
        _ = bench_heap_numba(heap, points[:1000], neighbors[:1000], distances[:1000])

        heap = make_heap(n_points, k)
        start = time.perf_counter()
        numba_count = bench_heap_numba(heap, points, neighbors, distances)
        numba_time = time.perf_counter() - start
        print(
            f"Numba heap:              {numba_time*1000:.2f}ms ({numba_count:,} accepted)"
        )

    except ImportError:
        numba_time = None
        print("Numba: NOT AVAILABLE")

    # Rust heap
    try:
        from pynndescent_rs import benchmark_heap_push

        # Warmup
        _ = benchmark_heap_push(
            1000, k, points[:1000], neighbors[:1000], distances[:1000]
        )

        start = time.perf_counter()
        rust_count = benchmark_heap_push(n_points, k, points, neighbors, distances)
        rust_time = time.perf_counter() - start
        print(
            f"Rust heap:               {rust_time*1000:.2f}ms ({rust_count:,} accepted)"
        )

        if numba_time:
            print(f"\nRust vs Numba speedup: {numba_time/rust_time:.2f}x")

    except (ImportError, AttributeError) as e:
        print(f"Rust: NOT AVAILABLE ({e})")


def benchmark_candidate_building(n_points=50000, k=30, max_candidates=60):
    """Benchmark candidate set building from neighbor graph."""
    print(f"\n{'='*60}")
    print(f"Candidate Building: {n_points} points, k={k}, max_cand={max_candidates}")
    print("=" * 60)

    # Create a random neighbor graph
    np.random.seed(42)
    indices = np.random.randint(0, n_points, (n_points, k)).astype(np.int32)
    distances = np.random.rand(n_points, k).astype(np.float32) * 10
    # Sort by distance
    for i in range(n_points):
        order = np.argsort(distances[i])
        indices[i] = indices[i][order]
        distances[i] = distances[i][order]
    # Random flags (50% new)
    flags = (np.random.rand(n_points, k) > 0.5).astype(np.uint8)

    # PyNNDescent candidate building
    try:
        from pynndescent.utils import make_heap
        from pynndescent.pynndescent_ import new_build_candidates
        import numba

        heap = (indices.copy(), distances.copy(), flags.copy())

        # Warmup
        rng_state = np.array([1, 2, 3], dtype=np.int64)
        _ = new_build_candidates(
            heap, max_candidates, rng_state, numba.get_num_threads()
        )

        heap = (indices.copy(), distances.copy(), flags.copy())
        rng_state = np.array([1, 2, 3], dtype=np.int64)

        start = time.perf_counter()
        new_cands, old_cands = new_build_candidates(
            heap, max_candidates, rng_state, numba.get_num_threads()
        )
        numba_time = time.perf_counter() - start
        print(f"Numba (parallel):        {numba_time*1000:.2f}ms")

    except (ImportError, AttributeError) as e:
        numba_time = None
        print(f"Numba: NOT AVAILABLE ({e})")

    # Rust candidate building
    try:
        from pynndescent_rs import benchmark_candidate_building

        # Warmup
        _ = benchmark_candidate_building(
            indices[:1000].copy(),
            distances[:1000].copy(),
            flags[:1000].copy(),
            max_candidates,
        )

        start = time.perf_counter()
        new_cands_rs, old_cands_rs = benchmark_candidate_building(
            indices.copy(), distances.copy(), flags.copy(), max_candidates
        )
        rust_time = time.perf_counter() - start
        print(f"Rust (parallel):         {rust_time*1000:.2f}ms")

        if numba_time:
            print(f"\nRust vs Numba speedup: {numba_time/rust_time:.2f}x")

    except (ImportError, AttributeError) as e:
        print(f"Rust: NOT AVAILABLE ({e})")


def main():
    print("=" * 60)
    print("Micro-Benchmarks: PyNNDescent vs Rust")
    print("=" * 60)

    benchmark_distance_computation()
    benchmark_heap_operations()
    benchmark_candidate_building()


if __name__ == "__main__":
    main()
