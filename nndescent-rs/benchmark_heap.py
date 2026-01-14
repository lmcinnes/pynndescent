#!/usr/bin/env python
"""
Benchmark heap push operations: Rust vs PyNNDescent (Numba).

This isolates just the heap operations to understand if they are the bottleneck.
"""

import numpy as np
import time
import numba


# PyNNDescent's checked_heap_push (from pynndescent/utils.py)
@numba.njit(
    "i4(f4[::1],i4[::1],f4,i4)",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
)
def checked_heap_push(priorities, indices, p, n):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # break if we already have this element.
    for i in range(size):
        if n == indices[i]:
            return 0

    # insert val at position zero
    priorities[0] = p
    indices[0] = n

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n

    return 1


@numba.njit(cache=True)
def benchmark_heap_numba(
    n_vertices,
    k,
    max_candidates,
    n_iters,
    priorities_list,
    indices_list,
    test_indices,
    test_priorities,
):
    """Simulate candidate building heap operations."""
    total_pushes = 0

    for iter_idx in range(n_iters):
        # Reset heaps
        for v in range(n_vertices):
            for c in range(max_candidates):
                priorities_list[v, c] = np.float32(np.inf)
                indices_list[v, c] = np.int32(-1)

        # Simulate pushing edges (forward + reverse)
        for i in range(n_vertices):
            for j in range(k):
                neighbor = test_indices[i, j]
                if neighbor < 0:
                    continue

                # Forward edge
                priority = test_priorities[iter_idx, i, j, 0]
                checked_heap_push(
                    priorities_list[i], indices_list[i], priority, neighbor
                )
                total_pushes += 1

                # Reverse edge
                reverse_priority = test_priorities[iter_idx, i, j, 1]
                checked_heap_push(
                    priorities_list[neighbor],
                    indices_list[neighbor],
                    reverse_priority,
                    np.int32(i),
                )
                total_pushes += 1

    return total_pushes


def benchmark_pynndescent_heap(n_vertices, k, max_candidates, n_iters):
    """Benchmark PyNNDescent heap operations."""

    # Pre-generate random data
    np.random.seed(42)
    test_indices = np.random.randint(0, n_vertices, size=(n_vertices, k)).astype(
        np.int32
    )
    test_priorities = np.random.rand(n_iters, n_vertices, k, 2).astype(np.float32)

    # Allocate heaps
    priorities_list = np.full((n_vertices, max_candidates), np.inf, dtype=np.float32)
    indices_list = np.full((n_vertices, max_candidates), -1, dtype=np.int32)

    # JIT warmup
    _ = benchmark_heap_numba(
        min(100, n_vertices),
        k,
        max_candidates,
        1,
        priorities_list[: min(100, n_vertices)].copy(),
        indices_list[: min(100, n_vertices)].copy(),
        test_indices[: min(100, n_vertices)],
        test_priorities[:1, : min(100, n_vertices)],
    )

    # Reset
    priorities_list = np.full((n_vertices, max_candidates), np.inf, dtype=np.float32)
    indices_list = np.full((n_vertices, max_candidates), -1, dtype=np.int32)

    # Benchmark
    start = time.perf_counter()
    total_pushes = benchmark_heap_numba(
        n_vertices,
        k,
        max_candidates,
        n_iters,
        priorities_list,
        indices_list,
        test_indices,
        test_priorities,
    )
    elapsed = time.perf_counter() - start

    return elapsed, total_pushes


def benchmark_rust_heap(n_vertices, k, max_candidates, n_iters):
    """Benchmark Rust heap operations via Python binding."""
    try:
        import pynndescent_rs

        # Pre-generate random data (same seed as Python)
        np.random.seed(42)
        test_indices = np.random.randint(0, n_vertices, size=(n_vertices, k)).astype(
            np.int32
        )
        test_priorities = np.random.rand(n_iters, n_vertices, k, 2).astype(np.float32)

        # Call Rust benchmark function
        start = time.perf_counter()
        total_pushes = pynndescent_rs.benchmark_heap_push(
            n_vertices, k, max_candidates, n_iters, test_indices, test_priorities
        )
        elapsed = time.perf_counter() - start

        return elapsed, total_pushes
    except (ImportError, AttributeError) as e:
        print(f"Rust heap benchmark not available: {e}")
        return None, None


def main():
    print("=" * 70)
    print("Heap Push Benchmark: Rust vs Numba")
    print("=" * 70)
    print()

    # Test configurations
    configs = [
        {"n_vertices": 1000, "k": 30, "max_candidates": 60, "n_iters": 8},
        {"n_vertices": 5000, "k": 30, "max_candidates": 60, "n_iters": 8},
        {"n_vertices": 10000, "k": 30, "max_candidates": 60, "n_iters": 8},
        {"n_vertices": 50000, "k": 30, "max_candidates": 60, "n_iters": 8},
    ]

    print(
        f"{'Config':<25} {'Numba (ms)':<12} {'Rust (ms)':<12} {'Speedup':<10} {'Pushes':<12}"
    )
    print("-" * 70)

    for cfg in configs:
        n_vertices = cfg["n_vertices"]
        k = cfg["k"]
        max_candidates = cfg["max_candidates"]
        n_iters = cfg["n_iters"]

        config_str = f"{n_vertices}×{k}×{n_iters}"

        # Benchmark Numba
        numba_time, numba_pushes = benchmark_pynndescent_heap(
            n_vertices, k, max_candidates, n_iters
        )

        # Benchmark Rust
        rust_time, rust_pushes = benchmark_rust_heap(
            n_vertices, k, max_candidates, n_iters
        )

        if rust_time is not None:
            speedup = numba_time / rust_time
            print(
                f"{config_str:<25} {numba_time*1000:>10.2f}ms {rust_time*1000:>10.2f}ms {speedup:>8.2f}x {numba_pushes:>10}"
            )
        else:
            print(
                f"{config_str:<25} {numba_time*1000:>10.2f}ms {'N/A':>12} {'N/A':>10} {numba_pushes:>10}"
            )

    print()
    print(
        "Note: Each push attempt may or may not modify the heap (duplicates rejected)."
    )


if __name__ == "__main__":
    main()
