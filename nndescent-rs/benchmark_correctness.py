#!/usr/bin/env python
"""
Correctness benchmark for NNDescent implementations.

This script verifies k-NN graph accuracy by comparing against ground truth
computed using sklearn's NearestNeighbors (brute-force exact k-NN).

Usage:
    python benchmark_correctness.py              # Standard correctness test
    python benchmark_correctness.py --full       # Include MNIST datasets
    python benchmark_correctness.py --quick      # Quick test (small datasets only)
"""

import argparse
import time
import numpy as np


# Test datasets
DATASETS = [
    {"n": 500, "dim": 20, "name": "small-500"},
    {"n": 1000, "dim": 50, "name": "medium-1k"},
    {"n": 5000, "dim": 100, "name": "large-5k"},
]

N_NEIGHBORS = 30


def generate_data(n, dim, seed=42):
    """Generate random test data."""
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def load_mnist(max_samples=10000):
    """Load MNIST dataset (subsampled for speed)."""
    try:
        from sklearn.datasets import fetch_openml

        print("Loading MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        data = mnist.data.astype(np.float32)
        # Normalize to [0, 1]
        data = data / 255.0
        # Subsample for reasonable test time
        if max_samples and len(data) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]
        print(f"  Loaded {data.shape[0]} samples, {data.shape[1]} dimensions")
        return data
    except Exception as e:
        print(f"Could not load MNIST: {e}")
        return None


def compute_ground_truth(data, n_neighbors):
    """Compute exact k-NN using sklearn's brute-force search.

    Returns neighbors EXCLUDING self (the point itself).
    """
    from sklearn.neighbors import NearestNeighbors

    # Add 1 to n_neighbors since the point itself is always the nearest
    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1, algorithm="brute", metric="euclidean"
    )
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Return without self-neighbors (skip first column)
    return indices[:, 1:], distances[:, 1:]


def remove_self_neighbors(indices, distances=None):
    """Remove self-neighbors from results (where index[i] == i).

    PyNNDescent includes each point as its own first neighbor with distance 0.
    This function removes those for fair comparison with ground truth.
    """
    n_points, k = indices.shape
    new_indices = np.zeros((n_points, k - 1), dtype=indices.dtype)
    new_distances = (
        None
        if distances is None
        else np.zeros((n_points, k - 1), dtype=distances.dtype)
    )

    for i in range(n_points):
        # Find where self appears (usually position 0)
        mask = indices[i] != i
        # Take first k-1 non-self neighbors
        non_self = indices[i][mask]
        new_indices[i] = non_self[: k - 1]
        if distances is not None:
            new_distances[i] = distances[i][mask][: k - 1]

    return (new_indices, new_distances) if distances is not None else new_indices


def compute_recall(indices_true, indices_test, k=None):
    """Compute recall@k: fraction of true neighbors found in test results."""
    if k is None:
        k = min(indices_true.shape[1], indices_test.shape[1])

    n = indices_true.shape[0]
    recall = 0.0

    for i in range(n):
        true_set = set(indices_true[i, :k])
        test_set = set(indices_test[i, :k])
        recall += len(true_set & test_set) / k

    return recall / n


def test_pynndescent(data, n_neighbors, ground_truth_indices):
    """Test PyNNDescent implementation.

    Note: PyNNDescent includes self as first neighbor, so we request n_neighbors+1
    and remove self-neighbors for fair comparison.
    """
    try:
        from pynndescent import NNDescent
    except ImportError:
        return None, None

    start = time.perf_counter()
    # Request extra neighbor since PyNND includes self
    index = NNDescent(
        data, n_neighbors=n_neighbors + 1, verbose=False, low_memory=False
    )
    indices, distances = index.neighbor_graph
    elapsed = time.perf_counter() - start

    # Remove self-neighbors for fair comparison
    indices = remove_self_neighbors(indices)

    recall = compute_recall(ground_truth_indices, indices)
    return recall, elapsed


def test_pynndescent_rs(data, n_neighbors, ground_truth_indices):
    """Test Rust implementation.

    Note: We need to check if Rust includes self-neighbors and handle accordingly.
    """
    try:
        from pynndescent_rs import NNDescent
    except ImportError:
        return None, None

    start = time.perf_counter()
    index = NNDescent(data, n_neighbors=n_neighbors, verbose=False)
    indices, distances = index.neighbor_graph
    elapsed = time.perf_counter() - start

    recall = compute_recall(ground_truth_indices, indices)
    return recall, elapsed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Correctness benchmark: Compare k-NN accuracy against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_correctness.py              # Standard test
    python benchmark_correctness.py --full       # Include MNIST
    python benchmark_correctness.py --quick      # Quick test (small data only)
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include MNIST dataset (slower)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run on smallest dataset for quick testing",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=N_NEIGHBORS,
        help=f"Number of neighbors (default: {N_NEIGHBORS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_neighbors = args.k

    print("=" * 80)
    print("NNDescent Correctness Benchmark: Recall vs Ground Truth")
    print("=" * 80)
    print(f"n_neighbors = {n_neighbors}")
    print()
    print(
        "Ground truth computed using sklearn.neighbors.NearestNeighbors (brute-force)"
    )
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
    except ImportError:
        print("pynndescent_rs: NOT AVAILABLE")
        rs_available = False

    print()

    # Build datasets list
    datasets = []

    if args.quick:
        # Only smallest dataset
        test_ds = [DATASETS[0]]
    else:
        test_ds = DATASETS

    for ds in test_ds:
        data = generate_data(ds["n"], ds["dim"])
        datasets.append({"name": ds["name"], "data": data})

    # Real-world datasets (only with --full)
    if args.full:
        mnist_data = load_mnist(max_samples=5000)  # Subsample for speed
        if mnist_data is not None:
            datasets.append({"name": "MNIST-5k", "data": mnist_data})

    # Header
    print("-" * 80)
    print(
        f"{'Dataset':<15} {'N':>8} {'Dim':>6} "
        f"{'PyNND Recall':>14} {'Rust Recall':>14} "
        f"{'GT Time':>10} {'Diff':>10}"
    )
    print("-" * 80)

    all_passed = True

    for dataset in datasets:
        name = dataset["name"]
        data = dataset["data"]
        n = data.shape[0]
        dim = data.shape[1]

        # Compute ground truth
        print(f"Computing ground truth for {name}...", end=" ", flush=True)
        gt_start = time.perf_counter()
        gt_indices, gt_distances = compute_ground_truth(data, n_neighbors)
        gt_time = time.perf_counter() - gt_start
        print(f"done ({gt_time:.2f}s)")

        # Test PyNNDescent
        if py_available:
            py_recall, py_time = test_pynndescent(data, n_neighbors, gt_indices)
            py_str = f"{py_recall:.4f}" if py_recall else "N/A"
        else:
            py_recall = None
            py_str = "N/A"

        # Test Rust
        if rs_available:
            rs_recall, rs_time = test_pynndescent_rs(data, n_neighbors, gt_indices)
            rs_str = f"{rs_recall:.4f}" if rs_recall else "N/A"
        else:
            rs_recall = None
            rs_str = "N/A"

        # Compute difference
        if py_recall is not None and rs_recall is not None:
            diff = rs_recall - py_recall
            diff_str = f"{diff:+.4f}"
            # Flag if Rust recall is significantly worse
            if diff < -0.05:
                diff_str += " ⚠️"
                all_passed = False
        else:
            diff_str = "N/A"

        print(
            f"{name:<15} {n:>8} {dim:>6} "
            f"{py_str:>14} {rs_str:>14} "
            f"{gt_time:>9.2f}s {diff_str:>10}"
        )

    print("-" * 80)
    print()

    # Summary
    if all_passed:
        print("✅ All tests passed! Rust recall is comparable to PyNNDescent.")
    else:
        print(
            "⚠️  Some tests show significant recall difference. Investigation recommended."
        )

    print()

    # Detailed comparison on one dataset
    if py_available and rs_available:
        print("Detailed Analysis (medium-1k dataset):")
        print("-" * 40)

        data = generate_data(1000, 50)
        gt_indices, gt_distances = compute_ground_truth(data, n_neighbors)

        from pynndescent import NNDescent as PyNND
        from pynndescent_rs import NNDescent as RsNND

        # Request n_neighbors+1 for PyNND since it includes self
        py_index = PyNND(
            data, n_neighbors=n_neighbors + 1, verbose=False, low_memory=False
        )
        py_indices, _ = py_index.neighbor_graph
        py_indices = remove_self_neighbors(py_indices)

        rs_index = RsNND(data, n_neighbors=n_neighbors, verbose=False)
        rs_indices, _ = rs_index.neighbor_graph

        # Compute recall at different k values (skip k=1, not meaningful for NN-descent)
        print(f"{'k':<6} {'PyNND Recall':>14} {'Rust Recall':>14} {'Difference':>12}")
        print("-" * 50)
        for k in [5, 10, 15, 20, 30]:
            if k <= n_neighbors:
                py_rec = compute_recall(gt_indices, py_indices, k=k)
                rs_rec = compute_recall(gt_indices, rs_indices, k=k)
                diff = rs_rec - py_rec
                print(f"{k:<6} {py_rec:>14.4f} {rs_rec:>14.4f} {diff:>+12.4f}")
        print()


if __name__ == "__main__":
    main()
