#!/usr/bin/env python
"""Profile Rust implementation with verbose output."""

import time
import numpy as np

# Generate test data
np.random.seed(42)
n, dim = 10000, 128
data = np.random.randn(n, dim).astype(np.float32)

print(f"Data shape: {data.shape}")
print(f"n_neighbors: 30")
print()

# Profile Rust implementation
print("=" * 60)
print("Rust Implementation (verbose)")
print("=" * 60)

from pynndescent_rs import NNDescent

start = time.perf_counter()
index = NNDescent(
    data,
    n_neighbors=30,
    n_trees=8,
    max_candidates=60,
    n_iters=10,
    verbose=True,  # Enable verbose output
)
neighbors, distances = index.neighbor_graph
elapsed = time.perf_counter() - start

print(f"\nTotal time: {elapsed:.3f}s")
print(f"Valid neighbors: {np.sum(neighbors >= 0)} / {n * 30}")

# Compare with PyNNDescent for recall
print()
print("=" * 60)
print("PyNNDescent for recall comparison")
print("=" * 60)

from pynndescent import NNDescent as PyNND

py_index = PyNND(data, n_neighbors=30, verbose=False)
py_neighbors, _ = py_index.neighbor_graph


# Calculate recall
def recall(rust_neighbors, py_neighbors):
    total = 0
    matches = 0
    for i in range(len(rust_neighbors)):
        rust_set = set(rust_neighbors[i][rust_neighbors[i] >= 0])
        py_set = set(py_neighbors[i])
        matches += len(rust_set & py_set)
        total += len(py_set)
    return matches / total if total > 0 else 0.0


r = recall(neighbors, py_neighbors)
print(f"Recall vs PyNNDescent: {r:.4f}")
