#!/usr/bin/env python
"""
Query benchmark comparing pynndescent_rs (Rust) and pynndescent (Python).

Uses ann-benchmarks HDF5 files which include:
  - train: training data
  - test: query points
  - neighbors: ground truth neighbor indices for test queries
  - distances: ground truth distances for test queries

Benchmarks:
  1. Fashion-MNIST-784-euclidean (60k train, 10k test, 784d)
  2. GloVe-100-angular (1.18M train, 10k test, 100d, cosine)

Usage:
    python benchmark_query.py
    python benchmark_query.py --runs 5
    python benchmark_query.py --k 10
"""

import argparse
import os
import time
import urllib.request

import h5py
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")
DATASETS = {
    "fashion-mnist-784-euclidean": {
        "url": "http://vectors.erikbern.com/fashion-mnist-784-euclidean.hdf5",
        "metric": "euclidean",
    },
    "glove-100-angular": {
        "url": "http://vectors.erikbern.com/glove-100-angular.hdf5",
        "metric": "cosine",
    },
}


def download_dataset(name):
    """Download an ann-benchmarks HDF5 dataset if not cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.hdf5")
    if os.path.exists(path):
        return path
    url = DATASETS[name]["url"]
    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")
    return path


def load_dataset(name):
    """Load train, test, ground-truth neighbors and distances from HDF5."""
    path = download_dataset(name)
    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])
        distances = np.array(f["distances"])
    return train, test, neighbors, distances


def recall_at_k(predicted, ground_truth, k):
    """Compute recall@k: fraction of true k-NN found in predicted k-NN."""
    gt = ground_truth[:, :k]
    pred = predicted[:, :k]
    recalls = []
    for i in range(gt.shape[0]):
        gt_set = set(gt[i])
        pred_set = set(pred[i])
        recalls.append(len(gt_set & pred_set) / k)
    return np.mean(recalls)


def benchmark_pynndescent(train, test, k, metric, epsilon, n_runs, n_neighbors=30):
    """Build index and benchmark queries with pynndescent (Python)."""
    from pynndescent import NNDescent

    # Build index (not timed — we only time queries)
    print(f"  Building PyNNDescent index (n_neighbors={n_neighbors})...")
    t0 = time.perf_counter()
    index = NNDescent(
        train,
        n_neighbors=n_neighbors,
        metric=metric,
        low_memory=False,
        verbose=False,
    )
    build_time = time.perf_counter() - t0
    print(f"  Index built in {build_time:.2f}s")

    # Prepare the search function (triggers JIT for query path)
    index.prepare()

    # Warmup query
    _ = index.query(test[:10], k=k, epsilon=epsilon)

    query_times = []
    result_indices = None
    for run in range(n_runs):
        t0 = time.perf_counter()
        indices, dists = index.query(test, k=k, epsilon=epsilon)
        elapsed = time.perf_counter() - t0
        query_times.append(elapsed)
        if result_indices is None:
            result_indices = indices

    return result_indices, query_times, build_time


def benchmark_rust(train, test, k, metric, epsilon, n_runs, n_neighbors=30):
    """Build index and benchmark queries with pynndescent_rs (Rust)."""
    import pynndescent_rs

    # Build index
    print(f"  Building Rust index (n_neighbors={n_neighbors})...")
    t0 = time.perf_counter()
    index = pynndescent_rs.NNDescent(
        train,
        n_neighbors=n_neighbors,
        metric=metric,
        verbose=False,
    )
    build_time = time.perf_counter() - t0
    print(f"  Index built in {build_time:.2f}s")

    # Warmup query
    _ = index.query(test[:10], k=k, epsilon=epsilon)

    query_times = []
    result_indices = None
    for run in range(n_runs):
        t0 = time.perf_counter()
        indices, dists = index.query(test, k=k, epsilon=epsilon)
        elapsed = time.perf_counter() - t0
        query_times.append(elapsed)
        if result_indices is None:
            result_indices = indices

    return result_indices, query_times, build_time


def main():
    parser = argparse.ArgumentParser(description="Query benchmark: Rust vs PyNNDescent")
    parser.add_argument("--runs", type=int, default=3, help="Number of query runs")
    parser.add_argument(
        "--k", type=int, default=10, help="Number of neighbors to query"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Search expansion factor"
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=30, help="Graph n_neighbors for index build"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()), help="Run only this dataset"
    )
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    print(
        f"Query benchmark: k={args.k}, n_neighbors={args.n_neighbors}, epsilon={args.epsilon}, runs={args.runs}"
    )
    print("=" * 80)

    for ds_name in datasets_to_run:
        metric = DATASETS[ds_name]["metric"]
        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}  (metric={metric})")
        print(f"{'=' * 80}")

        train, test, gt_neighbors, gt_distances = load_dataset(ds_name)
        train = np.ascontiguousarray(train, dtype=np.float32)
        test = np.ascontiguousarray(test, dtype=np.float32)
        print(f"  Train: {train.shape}, Test: {test.shape}")
        print(f"  Ground truth neighbors: {gt_neighbors.shape}")

        # --- PyNNDescent ---
        print(f"\n  [PyNNDescent]")
        py_indices, py_times, py_build = benchmark_pynndescent(
            train, test, args.k, metric, args.epsilon, args.runs, args.n_neighbors
        )
        py_recall = recall_at_k(py_indices, gt_neighbors, args.k)

        # --- Rust ---
        print(f"\n  [Rust]")
        rs_indices, rs_times, rs_build = benchmark_rust(
            train, test, args.k, metric, args.epsilon, args.runs, args.n_neighbors
        )
        rs_recall = recall_at_k(rs_indices, gt_neighbors, args.k)

        # --- Results ---
        py_median = np.median(py_times)
        rs_median = np.median(rs_times)
        speedup = py_median / rs_median if rs_median > 0 else float("inf")

        print(f"\n  Results (k={args.k}):")
        print(
            f"  {'':>20s} {'Build (s)':>10s} {'Query (s)':>10s} {'Recall@{}'.format(args.k):>10s}"
        )
        print(
            f"  {'PyNNDescent':>20s} {py_build:>10.3f} {py_median:>10.3f} {py_recall:>10.4f}"
        )
        print(
            f"  {'Rust':>20s} {rs_build:>10.3f} {rs_median:>10.3f} {rs_recall:>10.4f}"
        )
        print(f"  {'Speedup':>20s} {'':>10s} {speedup:>10.2f}x")

        print(f"\n  Query times (all runs):")
        print(f"    PyNNDescent: {['%.3fs' % t for t in py_times]}")
        print(f"    Rust:        {['%.3fs' % t for t in rs_times]}")


if __name__ == "__main__":
    main()
