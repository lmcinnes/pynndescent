#!/usr/bin/env python
"""
Compute exact k-NN ground truth for GloVe-100-angular and save to disk.

Uses batched normalized dot products for cosine distance.
"""

import numpy as np
import os
import time
import sys


def compute_exact_cosine_knn(data, k=30, batch_size=1000):
    """Compute exact k-NN graph using cosine distance via batched dot products.

    Cosine distance = 1 - cosine_similarity = 1 - (a·b)/(||a||·||b||)
    For normalized vectors: cosine distance = 1 - dot(a, b)
    So we normalize first, then find max dot products (= min cosine distance).
    """
    n = data.shape[0]
    dim = data.shape[1]

    # Normalize all vectors
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    data_norm = data / norms

    # Output arrays
    indices = np.full((n, k), -1, dtype=np.int32)
    distances = np.full((n, k), np.inf, dtype=np.float32)

    n_batches = (n + batch_size - 1) // batch_size
    t_start = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n)
        batch = data_norm[batch_start:batch_end]  # (batch_size, dim)

        # Compute dot products: batch × all_data^T -> (batch_size, n)
        # similarity[i,j] = dot(batch[i], data_norm[j])
        similarities = batch @ data_norm.T  # (batch_size, n)

        # Convert to cosine distance
        cos_dists = 1.0 - similarities  # (batch_size, n)

        # Exclude self-matches by setting self distance to inf
        for i in range(batch_end - batch_start):
            cos_dists[i, batch_start + i] = np.inf

        # Find k smallest distances using argpartition (faster than full sort)
        # argpartition gives indices of the k smallest, but not sorted
        kth_indices = np.argpartition(cos_dists, k, axis=1)[:, :k]

        # Gather the distances for these k indices
        batch_dists = np.take_along_axis(cos_dists, kth_indices, axis=1)

        # Sort by distance within the k neighbors
        sort_order = np.argsort(batch_dists, axis=1)
        kth_indices = np.take_along_axis(kth_indices, sort_order, axis=1)
        batch_dists = np.take_along_axis(batch_dists, sort_order, axis=1)

        indices[batch_start:batch_end] = kth_indices.astype(np.int32)
        distances[batch_start:batch_end] = batch_dists.astype(np.float32)

        if (batch_idx + 1) % 100 == 0 or batch_idx == n_batches - 1:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            print(
                f"  Batch {batch_idx+1}/{n_batches} "
                f"({batch_end}/{n} points, "
                f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)"
            )

    return indices, distances


def main():
    from benchmark_comparison import load_glove_100

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.join(cache_dir, "glove-100-exact-knn-k30.npz")

    if os.path.exists(output_path):
        print(f"Ground truth already exists at {output_path}")
        gt = np.load(output_path)
        print(f"  indices shape: {gt['indices'].shape}")
        print(f"  distances shape: {gt['distances'].shape}")
        print(
            f"  distance range: [{gt['distances'].min():.6f}, {gt['distances'].max():.6f}]"
        )
        return

    data = load_glove_100()
    if data is None:
        print("Failed to load GloVe data")
        sys.exit(1)

    k = 30
    print(
        f"\nComputing exact {k}-NN for {data.shape[0]} vectors (dim={data.shape[1]})..."
    )
    print(f"This will take a while (~1.18M × 1.18M cosine distances in batches)")

    t0 = time.perf_counter()
    indices, distances = compute_exact_cosine_knn(data, k=k, batch_size=2000)
    elapsed = time.perf_counter() - t0

    print(f"\nExact k-NN computed in {elapsed:.1f}s")
    print(f"  Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    print(f"  Mean distance: {distances.mean():.6f}")

    # Save
    np.savez_compressed(output_path, indices=indices, distances=distances)
    print(f"Saved to {output_path}")

    # Quick sanity check: verify a few points
    from benchmark_comparison import load_glove_100

    norms = np.linalg.norm(data, axis=1)
    for i in [0, 1000, 100000]:
        nn = indices[i, 0]
        d_stored = distances[i, 0]
        d_computed = 1.0 - np.dot(data[i], data[nn]) / (norms[i] * norms[nn])
        print(
            f"  Point {i}: nearest={nn}, dist_stored={d_stored:.8f}, dist_computed={d_computed:.8f}"
        )


if __name__ == "__main__":
    main()
