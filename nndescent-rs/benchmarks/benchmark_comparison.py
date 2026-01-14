"""
Benchmark comparing nndescent-rs against PyNNDescent.

This script benchmarks k-NN graph construction time and recall.
"""

import time
import numpy as np
import argparse
from typing import Tuple


def generate_test_data(n_samples: int, n_features: int, seed: int = 42) -> np.ndarray:
    """Generate random test data."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features)).astype(np.float32)


def compute_recall(indices_pred: np.ndarray, indices_true: np.ndarray) -> float:
    """Compute recall@k between predicted and true neighbors."""
    n_samples, k = indices_pred.shape
    recall = 0.0
    for i in range(n_samples):
        pred_set = set(indices_pred[i])
        true_set = set(indices_true[i])
        recall += len(pred_set & true_set) / k
    return recall / n_samples


def compute_exact_knn(data: np.ndarray, k: int) -> np.ndarray:
    """Compute exact k-NN using brute force (for small datasets)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(data)
    _, indices = nn.kneighbors(data)
    return indices


def benchmark_pynndescent(
    data: np.ndarray,
    n_neighbors: int,
    n_trees: int = 8,
    n_iters: int = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Benchmark PyNNDescent."""
    from pynndescent import NNDescent

    start = time.perf_counter()
    index = NNDescent(
        data,
        n_neighbors=n_neighbors,
        n_trees=n_trees,
        n_iters=n_iters,
        verbose=verbose,
        low_memory=False,
    )
    # Force graph computation
    indices, distances = index.neighbor_graph
    elapsed = time.perf_counter() - start

    return indices, elapsed


def benchmark_rust(
    data: np.ndarray,
    n_neighbors: int,
    n_trees: int = 8,
    n_iters: int = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Benchmark nndescent-rs."""
    try:
        import pynndescent_rs
    except ImportError:
        print("ERROR: pynndescent_rs not installed. Build with:")
        print("  cd crates/pynndescent-rs && maturin develop --release")
        return None, 0.0

    start = time.perf_counter()
    index = pynndescent_rs.NNDescent(
        data,
        n_neighbors=n_neighbors,
        n_trees=n_trees,
        n_iters=n_iters,
        verbose=verbose,
    )
    # Force graph computation
    indices, distances = index.neighbor_graph
    elapsed = time.perf_counter() - start

    return indices, elapsed


def run_benchmark(
    n_samples: int,
    n_features: int,
    n_neighbors: int,
    n_trees: int = 8,
    n_iters: int = None,
    compute_recall_flag: bool = True,
    seed: int = 42,
):
    """Run comparison benchmark."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_samples:,} samples × {n_features} dimensions")
    print(f"           k={n_neighbors}, n_trees={n_trees}, n_iters={n_iters}")
    print(f"{'='*60}")

    # Generate data
    print("\nGenerating test data...")
    data = generate_test_data(n_samples, n_features, seed)
    print(f"  Data shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Memory: {data.nbytes / 1024 / 1024:.1f} MB")

    # Compute exact k-NN for recall (only for smaller datasets)
    exact_indices = None
    if compute_recall_flag and n_samples <= 50000:
        print("\nComputing exact k-NN for recall measurement...")
        start = time.perf_counter()
        exact_indices = compute_exact_knn(data, n_neighbors)
        print(f"  Exact k-NN time: {time.perf_counter() - start:.2f}s")

    # Benchmark PyNNDescent
    print("\n--- PyNNDescent (Python) ---")
    py_indices, py_time = benchmark_pynndescent(
        data, n_neighbors, n_trees, n_iters, verbose=False
    )
    print(f"  Time: {py_time:.3f}s")
    if exact_indices is not None:
        py_recall = compute_recall(py_indices, exact_indices)
        print(f"  Recall@{n_neighbors}: {py_recall:.4f}")

    # Benchmark Rust
    print("\n--- nndescent-rs (Rust) ---")
    rs_indices, rs_time = benchmark_rust(
        data, n_neighbors, n_trees, n_iters, verbose=False
    )

    if rs_indices is not None:
        print(f"  Time: {rs_time:.3f}s")
        if exact_indices is not None:
            rs_recall = compute_recall(rs_indices, exact_indices)
            print(f"  Recall@{n_neighbors}: {rs_recall:.4f}")

        # Comparison
        print("\n--- Comparison ---")
        if rs_time > 0:
            speedup = py_time / rs_time
            print(f"  Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

        if exact_indices is not None:
            print(f"  PyNNDescent recall: {py_recall:.4f}")
            print(f"  Rust recall:        {rs_recall:.4f}")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_neighbors": n_neighbors,
        "py_time": py_time,
        "rs_time": rs_time,
        "speedup": py_time / rs_time if rs_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark nndescent-rs vs PyNNDescent"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmarks only"
    )
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument(
        "-n", "--n-samples", type=int, default=10000, help="Number of samples"
    )
    parser.add_argument(
        "-d", "--n-features", type=int, default=128, help="Number of features"
    )
    parser.add_argument(
        "-k", "--n-neighbors", type=int, default=30, help="Number of neighbors"
    )
    parser.add_argument("--n-trees", type=int, default=8, help="Number of RP trees")
    parser.add_argument(
        "--n-iters", type=int, default=None, help="Number of iterations"
    )
    parser.add_argument(
        "--no-recall", action="store_true", help="Skip recall computation"
    )
    args = parser.parse_args()

    if args.quick:
        # Quick benchmarks
        configs = [
            (1000, 64, 15),
            (5000, 128, 30),
            (10000, 128, 30),
        ]
    elif args.full:
        # Full benchmark suite
        configs = [
            (1000, 64, 15),
            (5000, 128, 30),
            (10000, 128, 30),
            (25000, 128, 30),
            (50000, 128, 30),
            (100000, 128, 30),
            (10000, 256, 30),
            (10000, 512, 30),
            (10000, 768, 30),  # BERT-like
        ]
    else:
        configs = [(args.n_samples, args.n_features, args.n_neighbors)]

    results = []
    for n_samples, n_features, n_neighbors in configs:
        result = run_benchmark(
            n_samples,
            n_features,
            n_neighbors,
            n_trees=args.n_trees,
            n_iters=args.n_iters,
            compute_recall_flag=not args.no_recall,
        )
        results.append(result)

    # Summary table
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Config':<25} {'PyNNDescent':>12} {'Rust':>12} {'Speedup':>10}")
        print("-" * 60)
        for r in results:
            config = f"{r['n_samples']:,}×{r['n_features']} k={r['n_neighbors']}"
            py_t = f"{r['py_time']:.3f}s"
            rs_t = f"{r['rs_time']:.3f}s" if r["rs_time"] > 0 else "N/A"
            speedup = f"{r['speedup']:.2f}x" if r["speedup"] > 0 else "N/A"
            print(f"{config:<25} {py_t:>12} {rs_t:>12} {speedup:>10}")


if __name__ == "__main__":
    main()
