#!/usr/bin/env python
"""
Benchmark comparison between pynndescent_rs (Rust) and pynndescent (Python).

This script compares k-NN graph construction performance.

Usage:
    python benchmark_comparison.py              # Quick benchmark (random data only)
    python benchmark_comparison.py --full       # Full benchmark including MNIST
    python benchmark_comparison.py --runs 5     # More runs for better statistics
"""

import argparse
import time
import numpy as np
import sys
import os

# Benchmark parameters
RANDOM_DATASETS = [
    {"n": 1000, "dim": 50, "name": "random-1k"},
    {"n": 5000, "dim": 100, "name": "random-5k"},
    {"n": 10000, "dim": 128, "name": "random-10k"},
    {"n": 50000, "dim": 128, "name": "random-50k"},
]

N_NEIGHBORS = 30
DEFAULT_N_RUNS = 3


def generate_data(n, dim, seed=42):
    """Generate random test data."""
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def load_mnist():
    """Load MNIST dataset (60k images, 784 dimensions)."""
    try:
        from sklearn.datasets import fetch_openml

        print("Loading MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        data = np.ascontiguousarray(mnist.data.astype(np.float32))
        # Normalize to [0, 1]
        data = np.ascontiguousarray(data / 255.0)
        print(f"  Loaded {data.shape[0]} samples, {data.shape[1]} dimensions")
        return data
    except Exception as e:
        print(f"Could not load MNIST: {e}")
        return None


def load_fashion_mnist():
    """Load Fashion-MNIST dataset (60k images, 784 dimensions)."""
    try:
        from sklearn.datasets import fetch_openml

        print("Loading Fashion-MNIST dataset...")
        fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
        data = np.ascontiguousarray(fmnist.data.astype(np.float32))
        # Normalize to [0, 1]
        data = np.ascontiguousarray(data / 255.0)
        print(f"  Loaded {data.shape[0]} samples, {data.shape[1]} dimensions")
        return data
    except Exception as e:
        print(f"Could not load Fashion-MNIST: {e}")
        return None


def load_glove_100():
    """Load GloVe-100-angular dataset from ann-benchmarks (1.2M vectors, 100d).

    Downloads the HDF5 file on first use and caches it locally.
    """
    import hashlib

    try:
        import h5py
    except ImportError:
        print("h5py not installed. Install with: pip install h5py")
        return None

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, "glove-100-angular.hdf5")

    if not os.path.exists(filepath):
        url = "http://vectors.erikbern.com/glove-100-angular.hdf5"
        print(f"Downloading GloVe-100-angular dataset from {url}...")
        print("  (This is ~463MB, may take a few minutes)")
        import urllib.request

        urllib.request.urlretrieve(url, filepath)
        print("  Download complete.")

    print("Loading GloVe-100-angular dataset...")
    with h5py.File(filepath, "r") as f:
        # ann-benchmarks stores train data under 'train' key
        data = np.array(f["train"], dtype=np.float32)
    data = np.ascontiguousarray(data)
    print(f"  Loaded {data.shape[0]} samples, {data.shape[1]} dimensions")
    return data


def benchmark_pynndescent(data, n_neighbors, n_runs=3, metric="euclidean"):
    """Benchmark the original PyNNDescent."""
    try:
        from pynndescent import NNDescent
    except ImportError:
        print("PyNNDescent not installed, skipping...")
        return None, None

    # Warmup with same size data to trigger JIT compilation
    warmup_data = np.random.randn(*data.shape).astype(np.float32)
    _ = NNDescent(
        warmup_data,
        n_neighbors=n_neighbors,
        metric=metric,
        verbose=False,
        low_memory=False,
    )

    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        index = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            verbose=False,
            low_memory=False,
        )
        # Force graph computation
        _ = index.neighbor_graph
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_pynndescent_rs(data, n_neighbors, n_runs=3, metric="euclidean"):
    """Benchmark the Rust implementation."""
    try:
        from pynndescent_rs import NNDescent
    except ImportError as e:
        print(f"pynndescent_rs not installed: {e}")
        return None, None

    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        index = NNDescent(data, n_neighbors=n_neighbors, metric=metric, verbose=False)
        # Force graph computation
        _ = index.neighbor_graph
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def compute_recall(indices_true, indices_test, k=None):
    """Compute recall@k between two sets of neighbor indices.

    Excludes self-loops (where index == row number) from both sets
    to handle the difference between PyNNDescent (includes self) and
    Rust (excludes self).
    """
    if k is None:
        k = indices_true.shape[1]

    n = indices_true.shape[0]
    recall = 0.0

    for i in range(n):
        true_set = set(indices_true[i, :k]) - {i}
        test_set = set(indices_test[i, :k]) - {i}
        if len(true_set) > 0:
            recall += len(true_set & test_set) / len(true_set)

    return recall / n


def warmup_jit():
    """Run a small warmup to trigger JIT compilation in PyNNDescent."""
    try:
        from pynndescent import NNDescent

        print("Warming up PyNNDescent JIT compilation...")
        warmup_data = np.random.randn(500, 50).astype(np.float32)
        _ = NNDescent(warmup_data, n_neighbors=10, verbose=False, low_memory=False)
        _ = NNDescent(
            warmup_data,
            n_neighbors=10,
            metric="cosine",
            verbose=False,
            low_memory=False,
        )
        print("JIT warmup complete.")
    except ImportError:
        pass


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark NNDescent: Rust vs Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_comparison.py              # Quick benchmark (random data only)
    python benchmark_comparison.py --full       # Full benchmark including MNIST
    python benchmark_comparison.py --runs 5     # More runs for better statistics
    python benchmark_comparison.py --quick      # Only small random datasets
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include MNIST and Fashion-MNIST datasets (slower, requires download)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run on small datasets (1k, 5k) for quick testing",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Number of runs per benchmark (default: {DEFAULT_N_RUNS})",
    )
    parser.add_argument(
        "--no-recall", action="store_true", help="Skip recall comparison"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.runs
    print("=" * 70)
    print("NNDescent Benchmark: Rust vs Python")
    print("=" * 70)
    print(f"n_neighbors = {N_NEIGHBORS}, n_runs = {n_runs}")
    if args.quick:
        print("Mode: QUICK (small datasets only)")
    elif args.full:
        print("Mode: FULL (including MNIST datasets)")
    else:
        print("Mode: STANDARD (random datasets only)")
    print()

    # Check available implementations
    py_available = True
    rs_available = True

    try:
        import pynndescent

        print(f"PyNNDescent version: {pynndescent.__version__}")
    except ImportError:
        print("PyNNDescent: NOT AVAILABLE")
        py_available = False

    try:
        import pynndescent_rs

        print(f"pynndescent_rs version: {pynndescent_rs.version()}")
        print(f"SIMD support: {pynndescent_rs.simd_info()}")
    except ImportError:
        print("pynndescent_rs: NOT AVAILABLE")
        rs_available = False

    # JIT warmup before benchmarks
    if py_available:
        print()
        warmup_jit()
        print()

    # Build datasets list
    datasets = []

    # Filter random datasets based on mode
    if args.quick:
        # Only small datasets for quick testing
        random_ds = [ds for ds in RANDOM_DATASETS if ds["n"] <= 5000]
    else:
        random_ds = RANDOM_DATASETS

    for ds in random_ds:
        data = generate_data(ds["n"], ds["dim"])
        datasets.append({"name": ds["name"], "data": data, "metric": "euclidean"})

    # Real-world datasets (only with --full)
    mnist_data = None
    glove_data = None
    if args.full:
        mnist_data = load_mnist()
        if mnist_data is not None:
            datasets.append(
                {"name": "MNIST", "data": mnist_data, "metric": "euclidean"}
            )

        fmnist_data = load_fashion_mnist()
        if fmnist_data is not None:
            datasets.append(
                {"name": "Fashion-MNIST", "data": fmnist_data, "metric": "euclidean"}
            )

        glove_data = load_glove_100()
        if glove_data is not None:
            datasets.append(
                {"name": "GloVe-100", "data": glove_data, "metric": "cosine"}
            )

    print()
    print("-" * 80)
    print(
        f"{'Dataset':<15} {'N':>8} {'Dim':>6} {'Metric':>8} {'PyNND (s)':>12} {'Rust (s)':>12} {'Speedup':>10}"
    )
    print("-" * 80)

    for dataset in datasets:
        name = dataset["name"]
        data = dataset["data"]
        metric = dataset.get("metric", "euclidean")
        n = data.shape[0]
        dim = data.shape[1]

        # Benchmark PyNNDescent
        if py_available:
            py_mean, py_std = benchmark_pynndescent(
                data, N_NEIGHBORS, n_runs, metric=metric
            )
            py_str = f"{py_mean:.3f}±{py_std:.3f}" if py_mean else "N/A"
        else:
            py_mean, py_std = None, None
            py_str = "N/A"

        # Benchmark Rust
        if rs_available:
            rs_mean, rs_std = benchmark_pynndescent_rs(
                data, N_NEIGHBORS, n_runs, metric=metric
            )
            rs_str = f"{rs_mean:.3f}±{rs_std:.3f}" if rs_mean else "N/A"
        else:
            rs_mean, rs_std = None, None
            rs_str = "N/A"

        # Compute speedup
        if py_mean and rs_mean:
            speedup = py_mean / rs_mean
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(
            f"{name:<15} {n:>8} {dim:>6} {metric:>8} {py_str:>12} {rs_str:>12} {speedup_str:>10}"
        )

    print("-" * 80)
    print()

    # Quality comparison (recall)
    if py_available and rs_available and not args.no_recall:
        print("Quality Comparison (Recall):")
        print("-" * 40)

        data = generate_data(5000, 100)

        from pynndescent import NNDescent as PyNND
        from pynndescent_rs import NNDescent as RsNND

        py_index = PyNND(data, n_neighbors=N_NEIGHBORS, verbose=False)
        py_indices, py_distances = py_index.neighbor_graph

        rs_index = RsNND(data, n_neighbors=N_NEIGHBORS, verbose=False)
        rs_indices, rs_distances = rs_index.neighbor_graph

        # Cross-compare
        recall_rs_vs_py = compute_recall(py_indices, rs_indices)
        print(f"Rust recall vs PyNNDescent: {recall_rs_vs_py:.4f}")

        # Test on MNIST if available
        if mnist_data is not None:
            print()
            print("MNIST recall comparison:")
            py_index = PyNND(mnist_data, n_neighbors=N_NEIGHBORS, verbose=False)
            py_indices, _ = py_index.neighbor_graph

            rs_index = RsNND(mnist_data, n_neighbors=N_NEIGHBORS, verbose=False)
            rs_indices, _ = rs_index.neighbor_graph

            recall = compute_recall(py_indices, rs_indices)
            print(f"  Rust recall vs PyNNDescent: {recall:.4f}")

        # Test on GloVe-100 if available
        if glove_data is not None:
            print()
            print("GloVe-100 recall comparison (cosine):")
            py_index = PyNND(
                glove_data,
                n_neighbors=N_NEIGHBORS,
                metric="cosine",
                verbose=False,
                low_memory=False,
            )
            py_indices, _ = py_index.neighbor_graph

            rs_index = RsNND(
                glove_data, n_neighbors=N_NEIGHBORS, metric="cosine", verbose=False
            )
            rs_indices, _ = rs_index.neighbor_graph

            recall = compute_recall(py_indices, rs_indices)
            print(f"  Rust recall vs PyNNDescent: {recall:.4f}")

            # Exact ground truth recall if available
            gt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".data",
                "glove-100-exact-knn-k30.npz",
            )
            if os.path.exists(gt_path):
                gt = np.load(gt_path)
                gt_indices = gt["indices"]
                print()
                print("GloVe-100 exact recall (vs brute-force ground truth):")

                def recall_vs_gt(test_indices, gt_indices, k):
                    n = gt_indices.shape[0]
                    correct = 0
                    total = 0
                    for i in range(n):
                        gt_set = set(gt_indices[i, :k])
                        test_set = set(test_indices[i, :k]) - {i}
                        correct += len(gt_set & test_set)
                        total += len(gt_set)
                    return correct / total

                rs_exact = recall_vs_gt(rs_indices, gt_indices, N_NEIGHBORS)
                py_exact = recall_vs_gt(py_indices, gt_indices, N_NEIGHBORS)
                print(f"  Rust exact recall@{N_NEIGHBORS}:  {rs_exact:.6f}")
                print(f"  PyNND exact recall@{N_NEIGHBORS}: {py_exact:.6f}")

        print()


if __name__ == "__main__":
    main()
