# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

from warnings import warn

import numba
import numpy as np
from sklearn.utils import check_random_state, check_array
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import (
    csr_matrix,
    coo_matrix,
    isspmatrix_csr,
    vstack as sparse_vstack,
    issparse,
)

import heapq

import pynndescent.sparse as sparse
import pynndescent.sparse_nndescent as sparse_nnd
import pynndescent.distances as pynnd_dist

from pynndescent.utils import (
    tau_rand_int,
    tau_rand,
    make_heap,
    deheap_sort,
    new_build_candidates,
    ts,
    simple_heap_push,
    checked_flagged_heap_push,
    has_been_visited,
    mark_visited,
    check_and_mark_visited,
    generate_graph_update_array,
    apply_graph_update_array,
    initalize_heap_from_graph_indices,
    initalize_heap_from_graph_indices_and_distances,
    sparse_initalize_heap_from_graph_indices,
    EMPTY_GRAPH,
)

from pynndescent.rp_trees import (
    make_forest,
    rptree_leaf_array,
    convert_tree_format,
    FlatTree,
    denumbaify_tree,
    renumbaify_tree,
    select_side,
    select_side_bit,
    sparse_select_side,
    score_linked_tree,
    make_hub_tree,
    make_sparse_hub_tree,
    make_bit_hub_tree,
)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

FLOAT32_EPS = np.finfo(np.float32).eps


def is_c_contiguous(array_like):
    flags = getattr(array_like, "flags", None)
    return flags is not None and flags["C_CONTIGUOUS"]


@numba.njit(parallel=True, cache=False, fastmath=True)
def generate_leaf_updates(
    updates, n_updates_per_thread, leaf_block, dist_thresholds, data, dist, n_threads
):
    """Generate leaf updates into pre-allocated arrays for parallel efficiency."""
    n_leaves = leaf_block.shape[0]
    leaves_per_thread = (n_leaves // n_threads) + 1

    # Reset update counts
    for t in range(n_threads):
        n_updates_per_thread[t] = 0

    for t in numba.prange(n_threads):
        start_leaf = t * leaves_per_thread
        end_leaf = min(start_leaf + leaves_per_thread, n_leaves)
        max_updates = updates.shape[1]
        count = 0

        for leaf_idx in range(start_leaf, end_leaf):
            for i in range(leaf_block.shape[1]):
                p = leaf_block[leaf_idx, i]
                if p < 0:
                    break

                for j in range(i + 1, leaf_block.shape[1]):
                    q = leaf_block[leaf_idx, j]
                    if q < 0:
                        break

                    d = dist(data[p], data[q])
                    max_threshold = max(dist_thresholds[p], dist_thresholds[q])
                    if d < max_threshold:
                        if count < max_updates:
                            updates[t, count, 0] = np.float32(p)
                            updates[t, count, 1] = np.float32(q)
                            updates[t, count, 2] = d
                            count += 1

        n_updates_per_thread[t] = count

    return updates


@numba.njit(
    locals={"d": numba.float32, "p": numba.int32, "q": numba.int32},
    cache=False,
    parallel=True,
    fastmath=True,
)
def init_rp_tree(data, dist, current_graph, leaf_array, n_threads=8):
    n_leaves = leaf_array.shape[0]
    block_size = n_threads * 64
    n_blocks = n_leaves // block_size

    max_leaf_size = leaf_array.shape[1]
    updates_per_thread = (
        int(block_size * max_leaf_size * (max_leaf_size - 1) / (2 * n_threads)) + 1
    )
    updates = np.zeros((n_threads, updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    n_vertices = current_graph[0].shape[0]
    vertex_block_size = n_vertices // n_threads + 1

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1][:, 0]

        generate_leaf_updates(
            updates,
            n_updates_per_thread,
            leaf_block,
            dist_thresholds,
            data,
            dist,
            n_threads,
        )

        for t in numba.prange(n_threads):
            v_block_start = t * vertex_block_size
            v_block_end = min(v_block_start + vertex_block_size, n_vertices)

            for j in range(n_threads):
                for k in range(n_updates_per_thread[j]):
                    p = np.int32(updates[j, k, 0])

                    if p < 0:
                        continue

                    q = np.int32(updates[j, k, 1])
                    d = updates[j, k, 2]

                    if p >= v_block_start and p < v_block_end:
                        checked_flagged_heap_push(
                            current_graph[1][p],
                            current_graph[0][p],
                            current_graph[2][p],
                            d,
                            q,
                            np.uint8(1),
                        )
                    if q >= v_block_start and q < v_block_end:
                        checked_flagged_heap_push(
                            current_graph[1][q],
                            current_graph[0][q],
                            current_graph[2][q],
                            d,
                            p,
                            np.uint8(1),
                        )


@numba.njit(
    fastmath=True,
    locals={"d": numba.float32, "idx": numba.int32, "i": numba.int32},
    cache=False,
)
def init_random(n_neighbors, data, heap, dist, rng_state):
    for i in range(data.shape[0]):
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                d = dist(data[idx], data[i])
                checked_flagged_heap_push(
                    heap[1][i], heap[0][i], heap[2][i], d, idx, np.uint8(1)
                )

    return


@numba.njit(cache=True)
def init_from_neighbor_graph(heap, indices, distances):
    for p in range(indices.shape[0]):
        for k in range(indices.shape[1]):
            q = indices[p, k]
            d = distances[p, k]
            checked_flagged_heap_push(heap[1][p], heap[0][p], heap[2][p], d, q, 0)

    return


@numba.njit(cache=False)
def process_candidates(
    data,
    dist,
    current_graph,
    new_candidate_neighbors,
    old_candidate_neighbors,
    n_blocks,
    block_size,
    n_threads,
    update_array,
    n_updates_per_thread,
):
    """Process candidate neighbors using array-based update generation.

    This is more efficient than the list-based approach because:
    1. No dynamic memory allocation during parallel loops
    2. Better cache locality with contiguous array storage
    3. Each thread writes to its own section of the array
    """
    c = 0
    n_vertices = new_candidate_neighbors.shape[0]
    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_vertices, (i + 1) * block_size)

        new_candidate_block = new_candidate_neighbors[block_start:block_end]
        old_candidate_block = old_candidate_neighbors[block_start:block_end]

        dist_thresholds = current_graph[1][:, 0]

        generate_graph_update_array(
            update_array,
            n_updates_per_thread,
            new_candidate_block,
            old_candidate_block,
            dist_thresholds,
            data,
            dist,
            n_threads,
        )

        c += apply_graph_update_array(
            current_graph, update_array, n_updates_per_thread, n_threads
        )

    return c


@numba.njit()
def nn_descent_internal(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=pynnd_dist.euclidean,
    n_iters=10,
    delta=0.001,
    verbose=False,
):
    n_vertices = data.shape[0]
    block_size = 16384
    n_blocks = n_vertices // block_size
    n_threads = numba.get_num_threads()

    # Pre-allocate update arrays for efficiency
    # Estimate max updates: each candidate pair can generate one update
    max_updates_per_thread = (
        int(
            (max_candidates**2 + max_candidates * (max_candidates - 1) / 2)
            * block_size
            / n_threads
        )
        + 1024
    )
    update_array = np.empty((n_threads, max_updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, max_candidates, rng_state, n_threads
        )

        c = process_candidates(
            data,
            dist,
            current_graph,
            new_candidate_neighbors,
            old_candidate_neighbors,
            n_blocks,
            block_size,
            n_threads,
            update_array,
            n_updates_per_thread,
        )

        if c <= delta * n_neighbors * data.shape[0]:
            if verbose:
                print("\tStopping threshold met -- exiting after", n + 1, "iterations")
            return


@numba.njit()
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=pynnd_dist.euclidean,
    n_iters=10,
    delta=0.001,
    init_graph=EMPTY_GRAPH,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=True,
    verbose=False,
):

    if init_graph[0].shape[0] == 1:  # EMPTY_GRAPH
        current_graph = make_heap(data.shape[0], n_neighbors)

        if rp_tree_init:
            init_rp_tree(data, dist, current_graph, leaf_array)

        init_random(n_neighbors, data, current_graph, dist, rng_state)
    elif (
        init_graph[0].shape[0] == data.shape[0]
        and init_graph[0].shape[1] == n_neighbors
    ):
        current_graph = init_graph
    else:
        raise ValueError("Invalid initial graph specified!")

    nn_descent_internal(
        current_graph,
        data,
        n_neighbors,
        rng_state,
        max_candidates=max_candidates,
        dist=dist,
        n_iters=n_iters,
        delta=delta,
        verbose=verbose,
    )

    return deheap_sort(current_graph[0], current_graph[1])


@numba.njit(parallel=True)
def diversify(indices, distances, data, dist, rng_state, prune_probability=1.0):

    for i in numba.prange(indices.shape[0]):

        new_indices = [indices[i, 0]]
        new_distances = [distances[i, 0]]
        for j in range(1, indices.shape[1]):
            if indices[i, j] < 0:
                break

            flag = True
            for k in range(len(new_indices)):

                c = new_indices[k]

                d = dist(data[indices[i, j]], data[c])
                if new_distances[k] > FLOAT32_EPS and d < distances[i, j]:
                    if tau_rand(rng_state) < prune_probability:
                        flag = False
                        break

            if flag:
                new_indices.append(indices[i, j])
                new_distances.append(distances[i, j])

        for j in range(indices.shape[1]):
            if j < len(new_indices):
                indices[i, j] = new_indices[j]
                distances[i, j] = new_distances[j]
            else:
                indices[i, j] = -1
                distances[i, j] = np.inf

    return indices, distances


@numba.njit()
def compute_degrees(indices):
    n = indices.shape[0]
    k = indices.shape[1]
    degree = np.zeros(n, dtype=np.int32)

    for i in range(n):
        for j in range(k):
            neighbor = indices[i, j]
            if neighbor >= 0:
                degree[i] += 1
                degree[neighbor] += 1

    return degree


@numba.njit()
def find_distance(indices, distances, source, target):
    k = indices.shape[1]
    for j in range(k):
        if indices[source, j] == target:
            return distances[source, j]
        if indices[source, j] < 0:
            break
    return np.float32(np.inf)


@numba.njit(parallel=True)
def diversify_degree_aware(
    indices, distances, data, dist, max_degree, aggressiveness=1.0, alpha=1.0
):
    """Diversify the k-NN graph with degree-aware pruning.

    This function applies relative neighborhood pruning with degree awareness.
    The aggressiveness parameter controls how much extra pruning is applied to
    edges involving high-degree nodes (hubs).

    Parameters
    ----------
    indices : ndarray of shape (n, k)
        The neighbor indices array (modified in place).
    distances : ndarray of shape (n, k)
        The neighbor distances array (modified in place).
    data : ndarray of shape (n, d)
        The original data points.
    dist : callable
        Distance function.
    max_degree : int
        Target maximum degree - nodes above this are candidates for adjusted pruning.
    aggressiveness : float (default=1.0)
        Controls the degree-aware pruning strength. Higher values prune more
        edges, particularly to/from high-degree hub nodes:
        - aggressiveness = 0: Standard diversify behavior (threshold = 1.0 for all)
        - aggressiveness = 1.0: Default, up to ~10% edge reduction vs standard
        - aggressiveness = 2.0: More aggressive, up to ~20% edge reduction
        - aggressiveness = 3.0: Very aggressive, up to ~25% edge reduction

        The threshold_factor for high-degree nodes scales as:
        1.0 + 0.04 * aggressiveness * min(degree_ratio - 1.0, 2.0)

        This means for a node at 2x max_degree with aggressiveness=1.0,
        we accept alternatives up to 4% longer than the direct edge.

    Returns
    -------
    indices : ndarray of shape (n, k)
        The pruned neighbor indices array.
    distances : ndarray of shape (n, k)
        The pruned neighbor distances array.
    """
    n = indices.shape[0]
    k = indices.shape[1]

    # Compute initial degrees (in undirected sense)
    degree = compute_degrees(indices)

    # Base rate of threshold adjustment per unit of excess degree ratio
    # At aggressiveness=1.0, this gives 4% max adjustment
    # Clamp to >= 0 since negative values have unintuitive behavior
    clamped_aggressiveness = max(np.float32(0.0), np.float32(aggressiveness))
    base_rate = np.float32(0.04) * clamped_aggressiveness

    for i in numba.prange(n):
        new_indices = [indices[i, 0]]
        new_distances = [distances[i, 0]]

        for j in range(1, k):
            if indices[i, j] < 0:
                break

            u = indices[i, j]
            d_iu = distances[i, j]

            # Compute threshold factor based on target degree and aggressiveness
            # Higher degree targets get adjusted pruning based on aggressiveness
            tgt_degree_ratio = np.float32(degree[u]) / np.float32(max_degree)

            if tgt_degree_ratio > 1.0:
                # High degree node - adjust threshold based on aggressiveness
                # Positive aggressiveness: threshold > 1.0 (prune more)
                # Negative aggressiveness: threshold < 1.0 (prune less)
                excess_ratio = min(tgt_degree_ratio - 1.0, np.float32(2.0))
                threshold_factor = np.float32(1.0) + base_rate * excess_ratio
                # Clamp to reasonable range [0.8, 1.2] to avoid extreme behavior
                threshold_factor = max(
                    np.float32(0.8), min(np.float32(1.2), threshold_factor)
                )
            else:
                # Normal/low degree - standard threshold
                threshold_factor = np.float32(1.0)

            # Check if there's an alternative path through any retained neighbor
            flag = True
            for m in range(len(new_indices)):
                c = new_indices[m]

                # Compute distance from candidate neighbor c to u
                d_cu = dist(data[u], data[c])

                # Prune if alternative path is acceptable
                if (
                    new_distances[m] > FLOAT32_EPS
                    and d_cu < d_iu * threshold_factor * alpha
                ):
                    flag = False
                    break

            if flag:
                new_indices.append(u)
                new_distances.append(d_iu)

        # Write back the retained edges
        for j in range(k):
            if j < len(new_indices):
                indices[i, j] = new_indices[j]
                distances[i, j] = new_distances[j]
            else:
                indices[i, j] = -1
                distances[i, j] = np.inf

    return indices, distances


@numba.njit(parallel=True)
def diversify_csr(
    graph_indptr,
    graph_indices,
    graph_data,
    source_data,
    dist,
    rng_state,
    prune_probability=1.0,
):
    n_nodes = graph_indptr.shape[0] - 1

    for i in numba.prange(n_nodes):

        current_indices = graph_indices[graph_indptr[i] : graph_indptr[i + 1]]
        current_data = graph_data[graph_indptr[i] : graph_indptr[i + 1]]

        order = np.argsort(current_data)
        retained = np.ones(order.shape[0], dtype=np.int8)

        for idx in range(1, order.shape[0]):
            j = order[idx]
            for k in range(idx):
                l = order[k]
                if retained[l] == 1:

                    d = dist(
                        source_data[current_indices[j]], source_data[current_indices[k]]
                    )
                    if current_data[l] > FLOAT32_EPS and d < current_data[j]:
                        if tau_rand(rng_state) < prune_probability:
                            retained[j] = 0
                            break

        for idx in range(order.shape[0]):
            j = order[idx]
            if retained[j] == 0:
                graph_data[graph_indptr[i] + j] = 0

    return


@numba.njit()
def compute_degrees_csr(graph_indptr, graph_indices):
    """Compute the undirected degree of each node in a CSR graph.

    For the reverse graph (transpose of forward graph), this counts how many
    nodes point TO each node (in-degree in original graph).

    Parameters
    ----------
    graph_indptr: array of int
        CSR row pointer array
    graph_indices: array of int
        CSR column indices array

    Returns
    -------
    degrees: array of int
        The degree of each node
    """
    n_nodes = graph_indptr.shape[0] - 1
    degrees = np.zeros(n_nodes, dtype=np.int32)

    # Count outgoing edges for each node
    for i in range(n_nodes):
        degrees[i] += graph_indptr[i + 1] - graph_indptr[i]

    # Count incoming edges (edges where this node is a target)
    for i in range(graph_indptr[-1]):
        if graph_indices[i] < n_nodes:
            degrees[graph_indices[i]] += 1

    return degrees


@numba.njit(parallel=True)
def diversify_csr_degree_aware(
    graph_indptr,
    graph_indices,
    graph_data,
    source_data,
    dist,
    rng_state,
    max_degree,
    aggressiveness=1.0,
    prune_probability=1.0,
):
    """Perform degree-aware diversification on a CSR format graph.

    This is the reverse diversification step, operating on the transposed graph.
    Higher degree nodes (hubs) get more aggressive pruning.

    Parameters
    ----------
    graph_indptr: array of int
        CSR row pointer array
    graph_indices: array of int
        CSR column indices array
    graph_data: array of float
        CSR data array (distances)
    source_data: array
        The original data points
    dist: callable
        Distance function
    rng_state: array
        Random state for probabilistic pruning
    max_degree: int
        The maximum degree considered for scaling
    aggressiveness: float (default 1.0)
        Controls how aggressively high-degree nodes are pruned.
        0.0 = standard diversification (no degree awareness)
        Higher values = more aggressive pruning of hub nodes
    prune_probability: float (default 1.0)
        Probability of pruning an eligible edge

    Returns
    -------
    None (modifies graph_data in place)
    """
    # Pre-compute degrees for all nodes (cannot be done in parallel section)
    degrees = compute_degrees_csr(graph_indptr, graph_indices)

    n_nodes = graph_indptr.shape[0] - 1

    for i in numba.prange(n_nodes):
        current_indices = graph_indices[graph_indptr[i] : graph_indptr[i + 1]]
        current_data = graph_data[graph_indptr[i] : graph_indptr[i + 1]]

        order = np.argsort(current_data)
        retained = np.ones(order.shape[0], dtype=np.int8)

        for idx in range(order.shape[0]):

            j = order[idx]
            if current_data[j] == 0:  # Already pruned or zero distance
                continue

            for k in range(idx):
                compare_idx = order[k]

                if retained[compare_idx] == 0:
                    continue

                d = dist(
                    source_data[current_indices[compare_idx]],
                    source_data[current_indices[j]],
                )

                # Compute degree-based threshold factor for node j
                # High-degree nodes get a relaxed threshold (accept longer paths)
                target_degree = 0
                if current_indices[j] < n_nodes:
                    target_degree = degrees[current_indices[j]]

                degree_ratio = target_degree / max(max_degree, 1)
                # Threshold increases with degree ratio, capped at 2x the base
                threshold_factor = 1.0 + 0.04 * aggressiveness * min(
                    degree_ratio - 1.0, 2.0
                )
                threshold_factor = max(threshold_factor, 1.0)  # Never go below 1.0

                # Prune if there's a shorter path through a retained neighbor
                if d * threshold_factor < current_data[j]:
                    if (
                        prune_probability >= 1.0
                        or tau_rand(rng_state) < prune_probability
                    ):
                        retained[j] = 0
                        break

        for idx in range(order.shape[0]):
            j = order[idx]
            if retained[j] == 0:
                graph_data[graph_indptr[i] + j] = 0

    return


@numba.njit(parallel=True)
def degree_prune_internal(indptr, data, max_degree=20):
    for i in numba.prange(indptr.shape[0] - 1):
        row_data = data[indptr[i] : indptr[i + 1]]
        if row_data.shape[0] > max_degree:
            cut_value = np.sort(row_data)[max_degree]
            for j in range(indptr[i], indptr[i + 1]):
                if data[j] > cut_value:
                    data[j] = 0.0

    return


def degree_prune(graph, max_degree=20):
    """Prune the k-neighbors graph back so that nodes have a maximum
    degree of ``max_degree``.

    Parameters
    ----------
    graph: sparse matrix
        The adjacency matrix of the graph

    max_degree: int (optional, default 20)
        The maximum degree of any node in the pruned graph

    Returns
    -------
    result: sparse matrix
        The pruned graph.
    """
    degree_prune_internal(graph.indptr, graph.data, max_degree)
    graph.eliminate_zeros()
    return graph


def resort_tree_indices(tree, tree_order):
    """Given a new data indexing, resort the tree indices to match"""
    new_tree = FlatTree(
        tree.hyperplanes,
        tree.offsets,
        tree.children,
        tree.indices[tree_order].astype(np.int32, order="C"),
        tree.leaf_size,
    )
    return new_tree


@numba.njit(cache=False, fastmath=True)
def rerank(nn_inds, queries, data, dist, n_neighbors, deheap_sort_function):
    heap = make_heap(queries.shape[0], n_neighbors)
    for i in range(queries.shape[0]):
        indices = heap[0][i]
        distances = heap[1][i]
        for j in range(nn_inds.shape[1]):
            idx = nn_inds[i, j]
            if idx < 0:
                continue
            d = dist(queries[i], data[idx])
            simple_heap_push(distances, indices, d, idx)

    result_nn_inds, result_nn_dists = deheap_sort_function(heap[0], heap[1])
    return result_nn_inds, result_nn_dists


class NNDescent:
    """NNDescent for fast approximate nearest neighbor queries. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    graph_data in many cases. This implementation provides a straightfoward
    interface, with access to some tuning parameters.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The training graph_data set to find nearest neighbors in.

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
            * hellinger
            * wasserstein-1d
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    n_neighbors: int (optional, default=30)
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    n_trees: int (optional, default=None)
        This implementation uses random projection forests for initializing the index
        build process. This parameter controls the number of trees in that forest. A
        larger number will result in more accurate neighbor computation at the cost
        of performance. The default of None means a value will be chosen based on the
        size of the data (typically 3-12 trees). Benchmarks show that 2-4 trees are
        usually sufficient for best recall.

    leaf_size: int (optional, default=None)
        The maximum number of points in a leaf for the random projection trees.
        The default of None means a value will be chosen based on n_neighbors
        (typically 60-200, computed as 5 * n_neighbors capped at 256). Benchmarks
        show that larger leaf sizes (100-200) often yield the best recall.

    pruning_degree_multiplier: float (optional, default=1.5)
        How aggressively to prune the graph. Since the search graph is undirected
        (and thus includes nearest neighbors and reverse nearest neighbors) vertices
        can have very high degree -- the graph will be pruned such that no
        vertex has degree greater than
        ``pruning_degree_multiplier * n_neighbors``.

    diversify_prob: float (optional, default=1.0)
        The search graph get "diversified" by removing potentially unnecessary
        edges. This controls the volume of edges removed. A value of 0.0 ensures
        that no edges get removed, and larger values result in significantly more
        aggressive edge removal. A value of 1.0 will prune all edges that it can.

    n_search_trees: int (optional, default=1)
        The number of random projection trees to use in initializing searching or
        querying.

        .. deprecated:: 0.5.5

    search_tree_leaf_size: int (optional, default=None)
        The maximum number of points in a leaf for the search tree (hub tree).
        This is independent of leaf_size which controls the init RP trees.
        The default of None means a value will be chosen automatically (currently 30).
        Smaller values may improve search accuracy at the cost of more tree traversals.

    max_search_tree_depth: int (optional, default=None)
        Maximum depth of the search tree. If None, uses max_rptree_depth.
        This allows independent tuning of search tree depth vs init tree depth.

    tree_init: bool (optional, default=True)
        Whether to use random projection trees for initialization.

    init_graph: np.ndarray (optional, default=None)
        2D array of indices of candidate neighbours of the shape
        (data.shape[0], n_neighbours). If the j-th neighbour of the i-th
        instances is unknown, use init_graph[i, j] = -1

    init_dist: np.ndarray (optional, default=None)
        2D array with the same shape as init_graph,
        such that metric(data[i], data[init_graph[i, j]]) equals
        init_dist[i, j]

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    algorithm: string (optional, default='standard')
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        The``'alternative'`` algorithm has been deprecated and is no longer
        available.

    low_memory: boolean (optional, default=True)
        Whether to use a lower memory, but more computationally expensive
        approach to index construction.

    max_candidates: int (optional, default=None)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    max_rptree_depth: int (optional, default=200)
        Maximum depth of random projection trees used for initializing NN-descent.
        Increasing this may result in a richer, deeper random projection forest,
        but it may be composed of many degenerate branches. Increase leaf_size
        in order to keep shallower, wider nondegenerate trees. Such wide trees,
        however, may yield poor performance of the preparation of the NN descent.

    n_iters: int (optional, default=None)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing. The default of None means
        a value will be chosen based on the size of the graph_data.

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    compressed: bool (optional, default=False)
        Whether to prune out data not needed for searching the index. This will
        result in a significantly smaller index, particularly useful for saving,
        but will remove information that might otherwise be useful.

    parallel_batch_queries: bool (optional, default=False)
        Whether to use parallelism of batched queries. This can be useful for large
        batches of queries on multicore machines, but results in performance degradation
        for single queries, so is poor for streaming use.

    verbose: bool (optional, default=False)
        Whether to print status graph_data during the computation.
    """

    def __init__(
        self,
        data,
        metric="euclidean",
        metric_kwds=None,
        n_neighbors=30,
        n_trees=None,
        leaf_size=None,
        pruning_degree_multiplier=1.5,
        diversify_prob=1.0,
        diversify_method="standard",
        degree_prune_aggressiveness=1.0,
        n_search_trees=1,
        search_tree_leaf_size=None,
        max_search_tree_depth=None,
        quantization=None,
        tree_init=True,
        init_graph=None,
        init_dist=None,
        random_state=None,
        low_memory=True,
        max_candidates=None,
        max_rptree_depth=200,
        n_iters=None,
        delta=0.001,
        n_jobs=None,
        compressed=False,
        parallel_batch_queries=False,
        verbose=False,
    ):

        if n_trees is None:
            n_trees = max(3, min(12, int(round(2.0 * np.log10(data.shape[0])))))
        if n_iters is None:
            n_iters = max(5, int(round(np.log2(data.shape[0]))))

        self.n_trees = n_trees
        self.n_trees_after_update = max(2, int(np.round(self.n_trees / 3)))
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.leaf_size = leaf_size
        self.prune_degree_multiplier = pruning_degree_multiplier
        self.diversify_prob = diversify_prob
        self.diversify_method = diversify_method
        self.degree_prune_aggressiveness = degree_prune_aggressiveness
        self.n_search_trees = n_search_trees
        self.search_tree_leaf_size = search_tree_leaf_size
        self.max_search_tree_depth = max_search_tree_depth
        self.max_rptree_depth = max_rptree_depth
        self.max_candidates = max_candidates
        self.quantization = quantization
        self.low_memory = low_memory
        self.n_iters = n_iters
        self.delta = delta
        self.dim = data.shape[1]
        self.n_jobs = n_jobs
        self.compressed = compressed
        self.parallel_batch_queries = parallel_batch_queries
        self.verbose = verbose

        if getattr(data, "dtype", None) == np.float32 and (
            issparse(data) or is_c_contiguous(data)
        ):
            copy_on_normalize = True
        else:
            copy_on_normalize = False

        if metric in ("bit_hamming", "bit_jaccard"):
            data = check_array(data, dtype=np.uint8, order="C")
            self._input_dtype = np.uint8
        else:
            data = check_array(data, dtype=np.float32, accept_sparse="csr", order="C")
            self._input_dtype = np.float32

        self._raw_data = data

        if not tree_init or n_trees == 0 or init_graph is not None:
            self.tree_init = False
        else:
            self.tree_init = True

        metric_kwds = metric_kwds or {}
        self._dist_args = tuple(metric_kwds.values())

        self.random_state = random_state

        current_random_state = check_random_state(self.random_state)

        self._distance_correction = None

        self._set_distance_func()

        if metric in (
            "cosine",
            "dot",
            "correlation",
            "dice",
            "jaccard",
            "hellinger",
            "hamming",
            "bit_hamming",
            "bit_jaccard",
        ):
            self._angular_trees = True
            if metric in ("bit_hamming", "bit_jaccard"):
                self._bit_trees = True
            else:
                self._bit_trees = False
        else:
            self._angular_trees = False
            self._bit_trees = False

        if metric == "dot":
            data = normalize(data, norm="l2", copy=copy_on_normalize)
            self._raw_data = data

        self.rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
        self.search_rng_state = current_random_state.randint(
            INT32_MIN, INT32_MAX, 3
        ).astype(np.int64)
        # Warm up the rng state
        for i in range(10):
            _ = tau_rand_int(self.search_rng_state)

        if self.tree_init:
            if verbose:
                print(ts(), "Building RP forest with", str(n_trees), "trees")
            self._rp_forest = make_forest(
                data,
                n_neighbors,
                n_trees,
                leaf_size,
                self.rng_state,
                current_random_state,
                self.n_jobs,
                self._angular_trees,
                self._bit_trees,
                max_depth=self.max_rptree_depth,
            )
            leaf_array = rptree_leaf_array(self._rp_forest)
        else:
            self._rp_forest = None
            leaf_array = np.array([[-1]])

        if self.max_candidates is None:
            effective_max_candidates = min(60, self.n_neighbors)
        else:
            effective_max_candidates = self.max_candidates

        # Set threading constraints
        self._original_num_threads = numba.get_num_threads()
        if self.n_jobs != -1 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)

        if isspmatrix_csr(self._raw_data):

            self._is_sparse = True

            if not self._raw_data.has_sorted_indices:
                self._raw_data.sort_indices()

            if metric in sparse.sparse_named_distances:
                if metric in sparse.sparse_fast_distance_alternatives:
                    _distance_func = sparse.sparse_fast_distance_alternatives[metric][
                        "dist"
                    ]
                    self._distance_correction = (
                        sparse.sparse_fast_distance_alternatives[metric]["correction"]
                    )
                else:
                    _distance_func = sparse.sparse_named_distances[metric]
            elif callable(metric):
                _distance_func = metric
            else:
                raise ValueError(
                    "Metric {} not supported for sparse data".format(metric)
                )

            if metric in sparse.sparse_need_n_features:
                metric_kwds["n_features"] = self._raw_data.shape[1]
            self._dist_args = tuple(metric_kwds.values())

            # Create a partial function for distances with arguments
            if len(self._dist_args) > 0:

                dist_args = self._dist_args

                @numba.njit()
                def _partial_dist_func(ind1, data1, ind2, data2):
                    return _distance_func(ind1, data1, ind2, data2, *dist_args)

                self._distance_func = _partial_dist_func
            else:
                self._distance_func = _distance_func

            if init_graph is None:
                _init_graph = EMPTY_GRAPH
            else:
                if init_graph.shape[0] != self._raw_data.shape[0]:
                    raise ValueError("Init graph size does not match dataset size!")
                _init_graph = make_heap(init_graph.shape[0], self.n_neighbors)
                _init_graph = sparse_initalize_heap_from_graph_indices(
                    _init_graph,
                    init_graph,
                    self._raw_data.indptr,
                    self._raw_data.indices,
                    self._raw_data.data,
                    self._distance_func,
                )

            if verbose:
                print(ts(), "metric NN descent for", str(n_iters), "iterations")

            self._neighbor_graph = sparse_nnd.nn_descent(
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                self.n_neighbors,
                self.rng_state,
                max_candidates=effective_max_candidates,
                dist=self._distance_func,
                n_iters=self.n_iters,
                delta=self.delta,
                rp_tree_init=True,
                leaf_array=leaf_array,
                init_graph=_init_graph,
                low_memory=self.low_memory,
                verbose=verbose,
            )

        else:

            self._is_sparse = False

            if init_graph is None:
                _init_graph = EMPTY_GRAPH
            else:
                if init_graph.shape[0] != self._raw_data.shape[0]:
                    raise ValueError("Init graph size does not match dataset size!")
                _init_graph = make_heap(init_graph.shape[0], self.n_neighbors)
                if init_dist is None:
                    _init_graph = initalize_heap_from_graph_indices(
                        _init_graph, init_graph, data, self._distance_func
                    )
                elif init_graph.shape != init_dist.shape:
                    raise ValueError(
                        "The shapes of init graph and init distances do not match!"
                    )
                else:
                    _init_graph = initalize_heap_from_graph_indices_and_distances(
                        _init_graph, init_graph, init_dist
                    )

            if verbose:
                print(ts(), "NN descent for", str(n_iters), "iterations")

            self._neighbor_graph = nn_descent(
                self._raw_data,
                self.n_neighbors,
                self.rng_state,
                effective_max_candidates,
                self._distance_func,
                self.n_iters,
                self.delta,
                low_memory=self.low_memory,
                rp_tree_init=True,
                init_graph=_init_graph,
                leaf_array=leaf_array,
                verbose=verbose,
            )

        if np.any(self._neighbor_graph[0] < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                " Results may be less than ideal. Try re-running with"
                " different parameters."
            )

        numba.set_num_threads(self._original_num_threads)

    def _set_distance_func(self):
        self._is_proxy_distance = False
        if callable(self.metric):
            _distance_func = self.metric
        elif self.metric in pynnd_dist.proxy_distances:
            self._is_proxy_distance = True
            _distance_func = pynnd_dist.proxy_distances[self.metric]["proxy_dist"]
            self._true_distance_func = pynnd_dist.proxy_distances[self.metric][
                "true_dist"
            ]
        elif self.metric in pynnd_dist.named_distances:
            if self.metric in pynnd_dist.fast_distance_alternatives:
                _distance_func = pynnd_dist.fast_distance_alternatives[self.metric][
                    "dist"
                ]
                self._distance_correction = pynnd_dist.fast_distance_alternatives[
                    self.metric
                ]["correction"]
            else:
                _distance_func = pynnd_dist.named_distances[self.metric]
        else:
            raise ValueError("Metric is neither callable, " + "nor a recognised string")

        # Create a partial function for distances with arguments
        if len(self._dist_args) > 0:
            dist_args = self._dist_args

            @numba.njit()
            def _partial_dist_func(x, y):
                return _distance_func(x, y, *dist_args)

            self._distance_func = _partial_dist_func
        else:
            self._distance_func = _distance_func

    def __getstate__(self):
        if not hasattr(self, "_search_graph"):
            self._init_search_graph()
        if not hasattr(self, "_search_function"):
            if self._is_sparse:
                self._init_sparse_search_function()
            else:
                self._init_search_function()
        result = self.__dict__.copy()
        if hasattr(self, "_rp_forest"):
            del result["_rp_forest"]
        result["_search_forest"] = tuple(
            [denumbaify_tree(tree) for tree in self._search_forest]
        )
        return result

    def __setstate__(self, d):
        self.__dict__ = d
        self._set_distance_func()
        self._search_forest = tuple(
            [renumbaify_tree(tree) for tree in d["_search_forest"]]
        )
        if self._is_sparse:
            self._init_sparse_search_function()
        else:
            self._init_search_function()

    def _init_search_graph(self):

        # Set threading constraints
        self._original_num_threads = numba.get_num_threads()
        if self.n_jobs != -1 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)

        # Determine search tree parameters (use dedicated params or fall back to init params)
        search_leaf_size = (
            self.search_tree_leaf_size
            if self.search_tree_leaf_size is not None
            else (self.leaf_size if self.leaf_size is not None else 30)
        )
        search_tree_depth = (
            self.max_search_tree_depth
            if self.max_search_tree_depth is not None
            else self.max_rptree_depth
        )

        if not hasattr(self, "_search_forest"):
            if self._rp_forest is None:
                if self.tree_init:
                    # We don't have a forest, so make a small search forest
                    current_random_state = check_random_state(self.random_state)
                    rp_forest = make_forest(
                        self._raw_data,
                        self.n_neighbors,
                        self.n_search_trees,
                        search_leaf_size,
                        self.rng_state,
                        current_random_state,
                        self.n_jobs,
                        self._angular_trees,
                        max_depth=search_tree_depth,
                    )
                    self._search_forest = [
                        convert_tree_format(
                            tree, self._raw_data.shape[0], self._raw_data.shape[1]
                        )
                        for tree in rp_forest
                    ]
                else:
                    self._search_forest = []
            else:
                # Build a graph-informed hub tree that minimizes edge cuts
                if self.verbose:
                    print(ts(), "Building hub-based search tree")

                if self._is_sparse:
                    # Sparse data - use simplified hub tree (faster, better quality)
                    gi_tree = make_sparse_hub_tree(
                        self._raw_data.indices,
                        self._raw_data.indptr,
                        self._raw_data.data,
                        self._neighbor_graph[0],
                        self.rng_state,
                        leaf_size=search_leaf_size,
                        angular=self._angular_trees,
                        max_depth=search_tree_depth,
                    )
                    del self._rp_forest
                    self._search_forest = [
                        convert_tree_format(
                            gi_tree,
                            self._raw_data.shape[0],
                            self._raw_data.indptr.shape[0] - 1,
                        )
                    ]
                elif getattr(self, "_bit_trees", False):
                    # Bit-packed data - use simplified hub tree (faster, better quality)
                    gi_tree = make_bit_hub_tree(
                        self._raw_data,
                        self._neighbor_graph[0],
                        self.rng_state,
                        leaf_size=search_leaf_size,
                        max_depth=search_tree_depth,
                    )
                    del self._rp_forest
                    self._search_forest = [
                        convert_tree_format(
                            gi_tree, self._raw_data.shape[0], self._raw_data.shape[1]
                        )
                    ]
                elif self.quantization == "binary" and hasattr(self, "_quantized_data"):
                    # Quantized binary data - use simplified hub tree (faster, better quality)
                    gi_tree = make_bit_hub_tree(
                        self._quantized_data,
                        self._neighbor_graph[0],
                        self.rng_state,
                        leaf_size=search_leaf_size,
                        max_depth=search_tree_depth,
                    )
                    del self._rp_forest
                    self._search_forest = [
                        convert_tree_format(
                            gi_tree,
                            self._quantized_data.shape[0],
                            self._quantized_data.shape[1],
                        )
                    ]
                    self._bit_trees = True
                else:
                    # Dense data - use simplified hub tree (faster, better quality)
                    gi_tree = make_hub_tree(
                        self._raw_data,
                        self._neighbor_graph[0],
                        self.rng_state,
                        leaf_size=search_leaf_size,
                        angular=self._angular_trees,
                        max_depth=search_tree_depth,
                    )
                    del self._rp_forest
                    self._search_forest = [
                        convert_tree_format(
                            gi_tree, self._raw_data.shape[0], self._raw_data.shape[1]
                        )
                    ]

        nnz_pre_diversify = np.sum(self._neighbor_graph[0] >= 0)
        if self._is_sparse:
            if self.compressed:
                diversified_rows, diversified_data = sparse.diversify(
                    self._neighbor_graph[0],
                    self._neighbor_graph[1],
                    self._raw_data.indices,
                    self._raw_data.indptr,
                    self._raw_data.data,
                    self._distance_func,
                    self.rng_state,
                    self.diversify_prob,
                )
            else:
                diversified_rows, diversified_data = sparse.diversify(
                    self._neighbor_graph[0].copy(),
                    self._neighbor_graph[1].copy(),
                    self._raw_data.indices,
                    self._raw_data.indptr,
                    self._raw_data.data,
                    self._distance_func,
                    self.rng_state,
                    self.diversify_prob,
                )
        else:
            if self.diversify_method == "degree_aware":
                # Use degree-aware diversification
                max_degree = int(self.prune_degree_multiplier * self.n_neighbors)
                if self.compressed:
                    diversified_rows, diversified_data = diversify_degree_aware(
                        self._neighbor_graph[0],
                        self._neighbor_graph[1],
                        self._raw_data,
                        self._distance_func,
                        max_degree,
                        self.degree_prune_aggressiveness,
                        self.diversify_prob,
                    )
                else:
                    diversified_rows, diversified_data = diversify_degree_aware(
                        self._neighbor_graph[0].copy(),
                        self._neighbor_graph[1].copy(),
                        self._raw_data,
                        self._distance_func,
                        max_degree,
                        self.degree_prune_aggressiveness,
                        self.diversify_prob,
                    )
            else:
                # Standard diversification
                if self.compressed:
                    diversified_rows, diversified_data = diversify(
                        self._neighbor_graph[0],
                        self._neighbor_graph[1],
                        self._raw_data,
                        self._distance_func,
                        self.rng_state,
                        self.diversify_prob,
                    )
                else:
                    diversified_rows, diversified_data = diversify(
                        self._neighbor_graph[0].copy(),
                        self._neighbor_graph[1].copy(),
                        self._raw_data,
                        self._distance_func,
                        self.rng_state,
                        self.diversify_prob,
                    )

        self._search_graph = coo_matrix(
            (self._raw_data.shape[0], self._raw_data.shape[0]), dtype=np.float32
        )

        # Preserve any distance 0 points
        diversified_data[diversified_data == 0.0] = FLOAT32_EPS

        self._search_graph.row = np.repeat(
            np.arange(diversified_rows.shape[0], dtype=np.int32),
            diversified_rows.shape[1],
        )
        self._search_graph.col = diversified_rows.ravel()
        self._search_graph.data = diversified_data.ravel()

        # Get rid of any -1 index entries
        self._search_graph = self._search_graph.tocsr()
        self._search_graph.data[self._search_graph.indices == -1] = 0.0
        self._search_graph.eliminate_zeros()

        self._min_distance = np.min(self._search_graph.data)

        if self.verbose:
            print(
                ts(),
                "Forward diversification reduced edges from {} to {}".format(
                    nnz_pre_diversify, self._search_graph.nnz
                ),
            )

        # Reverse graph
        pre_reverse_diversify_nnz = self._search_graph.nnz
        reverse_graph = self._search_graph.transpose()
        if self._is_sparse:
            sparse.diversify_csr(
                reverse_graph.indptr,
                reverse_graph.indices,
                reverse_graph.data,
                self._raw_data.indptr,
                self._raw_data.indices,
                self._raw_data.data,
                self._distance_func,
                self.rng_state,
                self.diversify_prob,
            )
        elif self.diversify_method == "degree_aware":
            diversify_csr_degree_aware(
                reverse_graph.indptr,
                reverse_graph.indices,
                reverse_graph.data,
                self._raw_data,
                self._distance_func,
                self.rng_state,
                self.n_neighbors,  # max_degree
                self.degree_prune_aggressiveness,
                self.diversify_prob,
            )
        else:
            diversify_csr(
                reverse_graph.indptr,
                reverse_graph.indices,
                reverse_graph.data,
                self._raw_data,
                self._distance_func,
                self.rng_state,
                self.diversify_prob,
            )
        reverse_graph.eliminate_zeros()

        if self.verbose:
            print(
                ts(),
                "Reverse diversification reduced edges from {} to {}".format(
                    pre_reverse_diversify_nnz, reverse_graph.nnz
                ),
            )
        reverse_graph = reverse_graph.tocsr()
        reverse_graph.sort_indices()
        self._search_graph = self._search_graph.tocsr()
        self._search_graph.sort_indices()
        self._search_graph = self._search_graph.maximum(reverse_graph).tocsr()

        # Eliminate the diagonal
        self._search_graph.setdiag(0.0)
        self._search_graph.eliminate_zeros()

        pre_prune_nnz = self._search_graph.nnz
        self._search_graph = degree_prune(
            self._search_graph,
            int(np.round(self.prune_degree_multiplier * self.n_neighbors)),
        )
        self._search_graph.eliminate_zeros()
        self._search_graph = (self._search_graph != 0).astype(np.uint8)

        if self.verbose:
            print(
                ts(),
                "Degree pruning reduced edges from {} to {}".format(
                    pre_prune_nnz, self._search_graph.nnz
                ),
            )

        self._visited = np.zeros(
            (self._raw_data.shape[0] // 8) + 1, dtype=np.uint8, order="C"
        )

        # reorder according to the search tree leaf order
        if self.verbose:
            print(ts(), "Resorting data and graph based on tree order")

        if self.tree_init:
            self._vertex_order = self._search_forest[0].indices
            row_ordered_graph = self._search_graph[self._vertex_order, :].tocsc()
            self._search_graph = row_ordered_graph[:, self._vertex_order]
            self._search_graph = self._search_graph.tocsr()
            self._search_graph.sort_indices()

            if self._is_sparse:
                self._raw_data = self._raw_data[self._vertex_order, :]
            else:
                self._raw_data = np.ascontiguousarray(
                    self._raw_data[self._vertex_order, :]
                )
                if hasattr(self, "_quantized_data"):
                    self._quantized_data = np.ascontiguousarray(
                        self._quantized_data[self._vertex_order, :]
                    )

            tree_order = np.argsort(self._vertex_order)
            self._search_forest = tuple(
                resort_tree_indices(tree, tree_order)
                for tree in self._search_forest[: self.n_search_trees]
            )
        else:
            self._vertex_order = np.arange(self._raw_data.shape[0])

        if self.compressed:
            if self.verbose:
                print(ts(), "Compressing index by removing unneeded attributes")
            if hasattr(self, "_rp_forest"):
                del self._rp_forest
            del self._neighbor_graph

        numba.set_num_threads(self._original_num_threads)

    def _init_search_function(self):

        if self.verbose:
            print(ts(), "Building and compiling search function")

        if self.tree_init:
            tree_hyperplanes = self._search_forest[0].hyperplanes
            tree_offsets = self._search_forest[0].offsets
            tree_indices = self._search_forest[0].indices
            tree_children = self._search_forest[0].children

            if self._bit_trees:

                @numba.njit(
                    [
                        numba.types.Array(numba.types.int32, 1, "C", readonly=True)(
                            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
                            numba.types.Array(
                                numba.types.int64, 1, "C", readonly=False
                            ),
                        )
                    ],
                    locals={"node": numba.types.uint32, "side": numba.types.boolean},
                )
                def tree_search_closure(point, rng_state):
                    node = 0
                    while tree_children[node, 0] > 0:
                        side = select_side_bit(
                            tree_hyperplanes[node], tree_offsets[node], point, rng_state
                        )
                        if side == 0:
                            node = tree_children[node, 0]
                        else:
                            node = tree_children[node, 1]

                    return -tree_children[node]

            else:

                @numba.njit(
                    [
                        numba.types.Array(numba.types.int32, 1, "C", readonly=True)(
                            numba.types.Array(
                                numba.types.float32, 1, "C", readonly=True
                            ),
                            numba.types.Array(
                                numba.types.int64, 1, "C", readonly=False
                            ),
                        )
                    ],
                    locals={"node": numba.types.uint32, "side": numba.types.boolean},
                )
                def tree_search_closure(point, rng_state):
                    node = 0
                    while tree_children[node, 0] > 0:
                        side = select_side(
                            tree_hyperplanes[node], tree_offsets[node], point, rng_state
                        )
                        if side == 0:
                            node = tree_children[node, 0]
                        else:
                            node = tree_children[node, 1]

                    return -tree_children[node]

            self._tree_search = tree_search_closure
        else:

            @numba.njit()
            def tree_search_closure(point, rng_state):
                return (0, 0)

            self._tree_search = tree_search_closure
            tree_indices = np.zeros(1, dtype=np.int64)

        if self.quantization is not None:
            data = self._quantized_data
            dist = self._quantized_distance_func
        else:
            data = self._raw_data
            dist = self._distance_func

        indptr = self._search_graph.indptr
        indices = self._search_graph.indices
        n_neighbors = self.n_neighbors
        parallel_search = self.parallel_batch_queries
        min_distance = self._min_distance

        if dist == pynnd_dist.bit_hamming or dist == pynnd_dist.bit_jaccard:
            data_type = numba.types.uint8[::1]
            query_data_type = numba.types.uint8[::1]
        elif self.quantization == "uint8" or self.quantization == "uint4":
            data_type = numba.types.uint8[::1]
            query_data_type = numba.types.float32[::1]
        else:
            data_type = numba.types.float32[::1]
            query_data_type = numba.types.float32[::1]

        if self.metric in (
            "cosine",
            "dot",
        ):
            normalize_query = True
        else:
            normalize_query = False

        @numba.njit(
            fastmath=True,
            locals={
                "current_query": query_data_type,
                "i": numba.types.uint32,
                "j": numba.types.uint32,
                "heap_priorities": numba.types.float32[::1],
                "heap_indices": numba.types.int32[::1],
                "candidate": numba.types.int32,
                "vertex": numba.types.int32,
                "d": numba.types.float32,
                "d_vertex": numba.types.float32,
                "visited": numba.types.uint8[::1],
                "indices": numba.types.int32[::1],
                "indptr": numba.types.int32[::1],
                "data": data_type,
                "heap_size": numba.types.int16,
                "distance_scale": numba.types.float32,
                "distance_bound": numba.types.float32,
                "seed_scale": numba.types.float32,
            },
            parallel=self.parallel_batch_queries,
        )
        def search_closure(query_points, k, epsilon, visited, rng_state):

            result = make_heap(query_points.shape[0], k)
            internal_rng_state = np.copy(rng_state)

            for i in numba.prange(query_points.shape[0]):
                # Avoid races on visited if parallel
                if parallel_search:
                    visited_nodes = np.zeros_like(visited)
                else:
                    visited_nodes = visited
                    visited_nodes[:] = 0

                if normalize_query:
                    norm = np.sqrt((query_points[i] ** 2).sum())
                    if norm > 0.0:
                        current_query = query_points[i] / norm
                    else:
                        continue
                else:
                    current_query = query_points[i]

                heap_priorities = result[1][i]
                heap_indices = result[0][i]
                seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]
                # heapq.heapify(seed_set)

                ############ Init from Tree ################
                index_bounds = tree_search_closure(current_query, internal_rng_state)
                candidate_indices = tree_indices[index_bounds[0] : index_bounds[1]]

                n_initial_points = candidate_indices.shape[0]

                for j in range(n_initial_points):
                    candidate = candidate_indices[j]
                    d = np.float32(dist(current_query, data[candidate]))
                    # indices are guaranteed different
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    mark_visited(visited_nodes, candidate)

                ############ Random samples if needed ################
                n_random_samples = min(k, n_neighbors) - n_initial_points

                if n_random_samples > 0:
                    for j in range(n_random_samples):
                        candidate = np.int32(
                            np.abs(tau_rand_int(internal_rng_state)) % data.shape[0]
                        )
                        if check_and_mark_visited(visited_nodes, candidate) == 0:
                            d = np.float32(dist(current_query, data[candidate]))
                            simple_heap_push(
                                heap_priorities, heap_indices, d, candidate
                            )
                            heapq.heappush(seed_set, (d, candidate))

                ############ Search ##############
                distance_bound = heap_priorities[0] + (
                    epsilon * (heap_priorities[0] - min_distance)
                )

                # Find smallest seed point
                d_vertex, vertex = heapq.heappop(seed_set)

                while d_vertex < distance_bound:

                    for j in range(indptr[vertex], indptr[vertex + 1]):

                        candidate = indices[j]

                        if check_and_mark_visited(visited_nodes, candidate) == 0:

                            d = np.float32(dist(current_query, data[candidate]))

                            if d < distance_bound:
                                simple_heap_push(
                                    heap_priorities, heap_indices, d, candidate
                                )
                                heapq.heappush(seed_set, (d, candidate))
                                # Update bound
                                distance_bound = heap_priorities[0] + (
                                    epsilon * (heap_priorities[0] - min_distance)
                                )

                    # find new smallest seed point
                    if len(seed_set) == 0:
                        break
                    else:
                        d_vertex, vertex = heapq.heappop(seed_set)

            return result

        self._search_function = search_closure
        if hasattr(deheap_sort, "py_func"):
            self._deheap_function = numba.njit(parallel=self.parallel_batch_queries)(
                deheap_sort.py_func
            )
        else:
            self._deheap_function = deheap_sort

        if hasattr(rerank, "py_func"):
            self._rerank_function = numba.njit(
                parallel=self.parallel_batch_queries, fastmath=True
            )(rerank.py_func)
        else:
            self._rerank_function = rerank

        # Force compilation of the search function (hardcoded k, epsilon)
        query_data = (
            self._raw_data[:1]
            if self.quantization != "binary"
            else self._quantized_data[:1]
        )
        inds, dists, _ = self._search_function(
            query_data, 5, 0.0, self._visited, self.search_rng_state
        )
        _ = self._deheap_function(inds, dists)

    def _init_sparse_search_function(self):

        if self.verbose:
            print(ts(), "Building and compiling sparse search function")

        if self.tree_init:
            tree_hyperplanes = self._search_forest[0].hyperplanes
            tree_offsets = self._search_forest[0].offsets
            tree_indices = self._search_forest[0].indices
            tree_children = self._search_forest[0].children

            @numba.njit(
                [
                    numba.types.Array(numba.types.int32, 1, "C", readonly=True)(
                        numba.types.Array(numba.types.int32, 1, "C", readonly=True),
                        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
                        numba.types.Array(numba.types.int64, 1, "C", readonly=False),
                    )
                ],
                locals={"node": numba.types.uint32, "side": numba.types.boolean},
            )
            def sparse_tree_search_closure(point_inds, point_data, rng_state):
                node = 0
                while tree_children[node, 0] > 0:
                    side = sparse_select_side(
                        tree_hyperplanes[node],
                        tree_offsets[node],
                        point_inds,
                        point_data,
                        rng_state,
                    )
                    if side == 0:
                        node = tree_children[node, 0]
                    else:
                        node = tree_children[node, 1]

                return -tree_children[node]

            self._tree_search = sparse_tree_search_closure
        else:

            @numba.njit()
            def sparse_tree_search_closure(point_inds, point_data, rng_state):
                return (0, 0)

            self._tree_search = sparse_tree_search_closure
            tree_indices = np.zeros(1, dtype=np.int64)

        from pynndescent.distances import alternative_dot, alternative_cosine

        data_inds = self._raw_data.indices
        data_indptr = self._raw_data.indptr
        data_data = self._raw_data.data
        indptr = self._search_graph.indptr
        indices = self._search_graph.indices
        dist = self._distance_func
        n_neighbors = self.n_neighbors
        parallel_search = self.parallel_batch_queries

        @numba.njit(
            fastmath=True,
            locals={
                "current_query": numba.types.float32[::1],
                "i": numba.types.uint32,
                "heap_priorities": numba.types.float32[::1],
                "heap_indices": numba.types.int32[::1],
                "candidate": numba.types.int32,
                "d": numba.types.float32,
                "visited": numba.types.uint8[::1],
                "indices": numba.types.int32[::1],
                "indptr": numba.types.int32[::1],
                "data": numba.types.float32[:, ::1],
                "heap_size": numba.types.int16,
                "distance_scale": numba.types.float32,
                "seed_scale": numba.types.float32,
            },
            parallel=self.parallel_batch_queries,
        )
        def search_closure(
            query_inds, query_indptr, query_data, k, epsilon, visited, rng_state
        ):

            n_query_points = query_indptr.shape[0] - 1
            n_index_points = data_indptr.shape[0] - 1
            result = make_heap(n_query_points, k)
            distance_scale = 1.0 + epsilon
            internal_rng_state = np.copy(rng_state)

            for i in numba.prange(n_query_points):
                # Avoid races on visited if parallel
                if parallel_search:
                    visited_nodes = np.zeros_like(visited)
                else:
                    visited_nodes = visited
                    visited_nodes[:] = 0

                current_query_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
                current_query_data = query_data[query_indptr[i] : query_indptr[i + 1]]

                if dist == alternative_dot or dist == alternative_cosine:
                    norm = np.sqrt((current_query_data**2).sum())
                    if norm > 0.0:
                        current_query_data = current_query_data / norm
                    else:
                        continue

                heap_priorities = result[1][i]
                heap_indices = result[0][i]
                seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]
                heapq.heapify(seed_set)

                ############ Init ################
                index_bounds = sparse_tree_search_closure(
                    current_query_inds, current_query_data, internal_rng_state
                )
                candidate_indices = tree_indices[index_bounds[0] : index_bounds[1]]

                n_initial_points = candidate_indices.shape[0]
                n_random_samples = min(k, n_neighbors) - n_initial_points

                for j in range(n_initial_points):
                    candidate = candidate_indices[j]

                    from_inds = data_inds[
                        data_indptr[candidate] : data_indptr[candidate + 1]
                    ]
                    from_data = data_data[
                        data_indptr[candidate] : data_indptr[candidate + 1]
                    ]

                    d = np.float32(
                        dist(
                            from_inds, from_data, current_query_inds, current_query_data
                        )
                    )
                    # indices are guaranteed different
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    mark_visited(visited_nodes, candidate)

                if n_random_samples > 0:
                    for j in range(n_random_samples):
                        candidate = np.int32(
                            np.abs(tau_rand_int(internal_rng_state)) % n_index_points
                        )
                        if check_and_mark_visited(visited_nodes, candidate) == 0:
                            from_inds = data_inds[
                                data_indptr[candidate] : data_indptr[candidate + 1]
                            ]
                            from_data = data_data[
                                data_indptr[candidate] : data_indptr[candidate + 1]
                            ]

                            d = np.float32(
                                dist(
                                    from_inds,
                                    from_data,
                                    current_query_inds,
                                    current_query_data,
                                )
                            )

                            simple_heap_push(
                                heap_priorities, heap_indices, d, candidate
                            )
                            heapq.heappush(seed_set, (d, candidate))

                ############ Search ##############
                distance_bound = distance_scale * heap_priorities[0]

                # Find smallest seed point
                d_vertex, vertex = heapq.heappop(seed_set)

                while d_vertex < distance_bound:

                    for j in range(indptr[vertex], indptr[vertex + 1]):

                        candidate = indices[j]

                        if check_and_mark_visited(visited_nodes, candidate) == 0:

                            from_inds = data_inds[
                                data_indptr[candidate] : data_indptr[candidate + 1]
                            ]
                            from_data = data_data[
                                data_indptr[candidate] : data_indptr[candidate + 1]
                            ]

                            d = np.float32(
                                dist(
                                    from_inds,
                                    from_data,
                                    current_query_inds,
                                    current_query_data,
                                )
                            )

                            if d < distance_bound:
                                simple_heap_push(
                                    heap_priorities, heap_indices, d, candidate
                                )
                                heapq.heappush(seed_set, (d, candidate))
                                # Update bound
                                distance_bound = distance_scale * heap_priorities[0]

                    # find new smallest seed point
                    if len(seed_set) == 0:
                        break
                    else:
                        d_vertex, vertex = heapq.heappop(seed_set)

            return result

        self._search_function = search_closure
        if hasattr(deheap_sort, "py_func"):
            self._deheap_function = numba.njit(parallel=self.parallel_batch_queries)(
                deheap_sort.py_func
            )
        else:
            self._deheap_function = deheap_sort

        # Force compilation of the search function (hardcoded k, epsilon)
        query_data = self._raw_data[:1]
        inds, dists, _ = self._search_function(
            query_data.indices,
            query_data.indptr,
            query_data.data,
            5,
            0.0,
            self._visited,
            self.search_rng_state,
        )
        _ = self._deheap_function(inds, dists)

    @property
    def neighbor_graph(self):
        if self.compressed and not hasattr(self, "_neighbor_graph"):
            warn("Compressed indexes do not have neighbor graph information.")
            return None
        if self._distance_correction is not None:
            result = (
                self._neighbor_graph[0].copy(),
                self._distance_correction(self._neighbor_graph[1]),
            )
        else:
            result = (self._neighbor_graph[0].copy(), self._neighbor_graph[1].copy())

        return result

    def compress_index(self):
        import gc

        self.prepare()
        self.compressed = True

        if hasattr(self, "_rp_forest"):
            del self._rp_forest
        if hasattr(self, "_neighbor_graph"):
            del self._neighbor_graph

        gc.collect()
        return

    def prepare(self):
        if self.quantization is not None:
            if self.quantization == "binary":
                # Quantize data to binary and set bit-based distance functions
                self._quantized_data = np.packbits(
                    (self._raw_data > 0).astype(np.uint8), axis=1
                )
                if self.metric in pynnd_dist.quantized_distances["binary"]:
                    self._quantized_distance_func = pynnd_dist.quantized_distances[
                        "binary"
                    ][self.metric]
                    self._is_proxy_distance = True
                    self._true_distance_func = self._distance_func
                else:
                    raise ValueError(
                        f"Not binary quantization version of {self.metric}"
                    )
            elif self.quantization == "uint8":
                # Quantize data to uint8 and set uint8-based distance functions
                current_random_state = check_random_state(self.random_state)
                sample_data = self._raw_data[
                    current_random_state.choice(
                        self._raw_data.shape[0],
                        min(10000, self._raw_data.shape[0]),
                        replace=False,
                    )
                ].ravel()
                if len(np.unique(sample_data)) <= 256:
                    self._quantized_values = np.unique(sample_data).astype(np.float32)
                else:
                    self._quantized_values = np.quantile(
                        sample_data, np.linspace(0, 1, 256)
                    ).astype(np.float32)
                self._quantized_data = np.searchsorted(
                    self._quantized_values, self._raw_data
                ).astype(np.uint8, order="C")
                if self.metric in pynnd_dist.quantized_distances["uint8"]:
                    self._quantized_distance_func_base = pynnd_dist.quantized_distances[
                        "uint8"
                    ][self.metric]
                    quantized_vals = self._quantized_values
                    quantized_dist = self._quantized_distance_func_base

                    @numba.njit(fastmath=True)
                    def quantized_distance_func_closure(a, b):
                        return quantized_dist(a, b, quantized_vals)

                    self._quantized_distance_func = quantized_distance_func_closure
                    self._is_proxy_distance = True
                    self._true_distance_func = self._distance_func
                else:
                    raise ValueError(f"Not uint8 quantization version of {self.metric}")
            elif self.quantization == "uint4":
                # Quantize data to uint4 and set uint4-based distance functions
                current_random_state = check_random_state(self.random_state)
                sample_data = self._raw_data[
                    current_random_state.choice(
                        self._raw_data.shape[0],
                        min(10000, self._raw_data.shape[0]),
                        replace=False,
                    )
                ].ravel()
                self._quantized_values = np.quantile(
                    sample_data, np.linspace(0, 1, 16)
                ).astype(np.float32)
                quantized_data_8bit = np.searchsorted(
                    self._quantized_values, self._raw_data
                ).astype(np.uint8, order="C")
                # Pack two uint4 into one uint8
                self._quantized_data = (
                    (quantized_data_8bit[:, ::2] << 4) | (quantized_data_8bit[:, 1::2])
                ).astype(np.uint8, order="C")
                if self.metric in pynnd_dist.quantized_distances["uint4"]:
                    self._quantized_distance_func_base = pynnd_dist.quantized_distances[
                        "uint4"
                    ][self.metric]
                    quantized_vals = self._quantized_values
                    quantized_dist = self._quantized_distance_func_base

                    @numba.njit(fastmath=True)
                    def quantized_distance_func_closure(a, b):
                        return quantized_dist(a, b, quantized_vals)

                    self._quantized_distance_func = quantized_distance_func_closure
                    self._is_proxy_distance = True
                    self._true_distance_func = self._distance_func
                else:
                    raise ValueError(f"Not uint4 quantization version of {self.metric}")
            else:
                raise ValueError(f"Unrecognized quantization type {self.quantization}")

        if not hasattr(self, "_search_graph"):
            self._init_search_graph()

        if not hasattr(self, "_search_function"):
            if self._is_sparse:
                self._init_sparse_search_function()
            else:
                self._init_search_function()
        return

    def query(self, query_data, k=10, epsilon=0.1, proxy_beam_size=4):
        """Query the training graph_data for the k nearest neighbors

        Parameters
        ----------
        query_data: array-like, last dimension self.dim
            An array of points to query

        k: integer (default = 10)
            The number of nearest neighbors to return

        epsilon: float (optional, default=0.1)
            When searching for nearest neighbors of a query point this values
            controls the trade-off between accuracy and search cost. Larger values
            produce more accurate nearest neighbor results at larger computational
            cost for the search. Values should be in the range 0.0 to 0.5, but
            should probably not exceed 0.3 without good reason.

        Returns
        -------
        indices, distances: array (n_query_points, k), array (n_query_points, k)
            The first array, ``indices``, provides the indices of the graph_data
            points in the training set that are the nearest neighbors of
            each query point. Thus ``indices[i, j]`` is the index into the
            training graph_data of the jth nearest neighbor of the ith query points.

            Similarly ``distances`` provides the distances to the neighbors
            of the query points such that ``distances[i, j]`` is the distance
            from the ith query point to its jth nearest neighbor in the
            training graph_data.
        """
        if not hasattr(self, "_search_graph") or not hasattr(self, "_search_function"):
            self.prepare()

        if self._is_proxy_distance:
            search_k = proxy_beam_size * k
        else:
            search_k = k

        if not self._is_sparse:
            # Standard case
            if self.metric in ("bit_hamming", "bit_jaccard"):
                query_data = np.asarray(query_data).astype(np.uint8, order="C")
            else:
                query_data = np.asarray(query_data).astype(np.float32, order="C")

            if self.quantization is not None:
                epsilon += 1e-32  # ensure epsilon > 0 for quantized searches
                if self.quantization == "binary":
                    proxy_query_data = np.packbits(
                        (query_data > 0).astype(np.uint8), axis=1
                    )
                elif self.quantization == "uint8" or self.quantization == "uint4":
                    proxy_query_data = query_data
                else:
                    raise ValueError(
                        f"Unrecognized quantization type {self.quantization}"
                    )
            else:
                proxy_query_data = query_data

            indices, dists, _ = self._search_function(
                proxy_query_data,
                search_k,
                epsilon,
                self._visited,
                self.search_rng_state,
            )
        else:
            # Sparse case
            query_data = check_array(query_data, accept_sparse="csr", dtype=np.float32)
            if not isspmatrix_csr(query_data):
                query_data = csr_matrix(query_data, dtype=np.float32)
            if not query_data.has_sorted_indices:
                query_data.sort_indices()

            indices, dists, _ = self._search_function(
                query_data.indices,
                query_data.indptr,
                query_data.data,
                search_k,
                epsilon,
                self._visited,
                self.search_rng_state,
            )

        indices, dists = self._deheap_function(indices, dists)

        if self._is_proxy_distance:
            indices, dists = rerank(
                indices,
                query_data,
                self._raw_data,
                self._true_distance_func,
                k,
                self._deheap_function,
            )

        # Sort to input graph_data order
        indices = self._vertex_order[indices]

        if self._distance_correction is not None:
            dists = self._distance_correction(dists)

        return indices, dists

    def update(self, xs_fresh=None, xs_updated=None, updated_indices=None):
        """
        Updates the index with a) fresh data (that is appended to
        the existing data), and b) data that was only updated (but should not be appended
        to the existing data).

        Not applicable to sparse data yet.

        Parameters
        ----------
        xs_fresh: np.ndarray (optional, default=None)
            2D array of the shape (n_fresh, dim) where dim is the dimension
            of the data from which we built self.

        xs_updated: np.ndarray (optional, default=None)
            2D array of the shape (n_updates, dim) where dim is the dimension
            of the data from which we built self.

        updated_indices: array-like of size n_updates (optional, default=None)
            Something that is convertable to list of ints.
            If self is currently built from xs, then xs[update_indices[i]]
            will be replaced by xs_updated[i].

        Returns
        -------
            None
        """
        current_random_state = check_random_state(self.random_state)
        rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
        error_sparse_to_do = NotImplementedError("Sparse update not complete yet")
        # input checks
        if xs_updated is not None:
            xs_updated = check_array(
                xs_updated, dtype=self._input_dtype, accept_sparse="csr", order="C"
            )
            if updated_indices is None:
                raise ValueError(
                    "If xs_updated are provided, updated_indices must also be provided!"
                )
            if self._is_sparse:
                raise error_sparse_to_do
            else:
                try:
                    updated_indices = list(map(int, updated_indices))
                except (TypeError, ValueError):
                    raise ValueError(
                        "Could not convert updated indices to list of int(s)."
                    )
                n1 = len(updated_indices)
                n2 = xs_updated.shape[0]
                if n1 != n2:
                    raise ValueError(
                        f"Number of updated indices ({n1}) must match "
                        f"number of rows of xs_updated ({n2})."
                    )
        else:
            if updated_indices is not None:
                warn(
                    "xs_updated not provided, while update_indices provided. "
                    "They will be ignored."
                )
                updated_indices = None
        if updated_indices is None:
            # make an empty iterable instead
            xs_updated = []
            updated_indices = []
        if xs_fresh is None:
            if self._is_sparse:
                xs_fresh = csr_matrix(
                    ([], [], []),
                    shape=(0, self._raw_data.shape[1]),
                    dtype=self._input_dtype,
                )
            else:
                xs_fresh = np.zeros(
                    (0, self._raw_data.shape[1]), dtype=self._input_dtype
                )
        else:
            xs_fresh = check_array(
                xs_fresh, dtype=self._input_dtype, accept_sparse="csr", order="C"
            )
        # data preparation
        if hasattr(self, "_vertex_order"):
            original_order = np.argsort(self._vertex_order)
        else:
            original_order = np.ones(self._raw_data.shape[0], dtype=np.bool_)
        if self._is_sparse:
            self._raw_data = sparse_vstack([self._raw_data, xs_fresh])
            if updated_indices:
                # cannot be reached due to the check above,
                # but will leave this here as a marker
                raise error_sparse_to_do
        else:
            self._raw_data = self._raw_data[original_order, :]
            for x_updated, i_fresh in zip(xs_updated, updated_indices):
                self._raw_data[i_fresh] = x_updated
            self._raw_data = np.ascontiguousarray(np.vstack([self._raw_data, xs_fresh]))
            ns, ds = self._neighbor_graph
            n_examples, n_neighbors = ns.shape
            indices_set = set(updated_indices)  # for fast "is element" checks
            for i in range(n_examples):
                # maybe update whole row
                if i in indices_set:
                    ns[i] = -1
                    ds[i] = np.inf
                    continue
                # maybe update some columns
                for j in range(n_neighbors):
                    if ns[i, j] in indices_set:
                        ns[i, j] = -1
                        ds[i, j] = np.inf
        # update neighbors
        if self._is_sparse:
            raise error_sparse_to_do
        else:
            self.n_trees = self.n_trees_after_update
            self._rp_forest = make_forest(
                self._raw_data,
                self.n_neighbors,
                self.n_trees,
                self.leaf_size,
                rng_state,
                current_random_state,
                self.n_jobs,
                self._angular_trees,
                max_depth=self.max_rptree_depth,
            )
            leaf_array = rptree_leaf_array(self._rp_forest)
            current_graph = make_heap(self._raw_data.shape[0], self.n_neighbors)
            init_from_neighbor_graph(
                current_graph, self._neighbor_graph[0], self._neighbor_graph[1]
            )
            init_rp_tree(self._raw_data, self._distance_func, current_graph, leaf_array)

            if self.max_candidates is None:
                effective_max_candidates = min(60, self.n_neighbors)
            else:
                effective_max_candidates = self.max_candidates

            self._neighbor_graph = nn_descent(
                self._raw_data,
                self.n_neighbors,
                self.rng_state,
                effective_max_candidates,
                self._distance_func,
                self.n_iters,
                self.delta,
                init_graph=current_graph,
                low_memory=self.low_memory,
                rp_tree_init=False,
                leaf_array=np.array([[-1], [-1]]),
                verbose=self.verbose,
            )

            # Remove search graph and search function
            # and rerun prepare if it was graph_data previously
            if (
                hasattr(self, "_search_graph")
                or hasattr(self, "_search_function")
                or hasattr(self, "_search_forest")
            ):
                if hasattr(self, "_search_graph"):
                    del self._search_graph

                if hasattr(self, "_search_forest"):
                    del self._search_forest

                if hasattr(self, "_search_function"):
                    del self._search_function

                self.prepare()


class PyNNDescentTransformer(BaseEstimator, TransformerMixin):
    """PyNNDescentTransformer for fast approximate nearest neighbor transformer.
    It uses the NNDescent algorithm, and is thus
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    graph_data in many cases.

    Transform X into a (weighted) graph of k nearest neighbors

    The transformed graph_data is a sparse graph as returned by kneighbors_graph.

    Parameters
    ----------
    n_neighbors: int (optional, default=5)
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
            * hellinger
            * wasserstein-1d
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    n_trees: int (optional, default=None)
        This implementation uses random projection forests for initialization
        of searches. This parameter controls the number of trees in that
        forest. A larger number will result in more accurate neighbor
        computation at the cost of performance. The default of None means
        a value will be chosen based on the size of the graph_data.

    leaf_size: int (optional, default=None)
        The maximum number of points in a leaf for the random projection trees.
        The default of None means a value will be chosen based on n_neighbors.

    pruning_degree_multiplier: float (optional, default=1.5)
        How aggressively to prune the graph. Since the search graph is undirected
        (and thus includes nearest neighbors and reverse nearest neighbors) vertices
        can have very high degree -- the graph will be pruned such that no
        vertex has degree greater than
        ``pruning_degree_multiplier * n_neighbors``.

    diversify_prob: float (optional, default=1.0)
        The search graph get "diversified" by removing potentially unnecessary
        edges. This controls the volume of edges removed. A value of 0.0 ensures
        that no edges get removed, and larger values result in significantly more
        aggressive edge removal. A value of 1.0 will prune all edges that it can.

    n_search_trees: int (optional, default=1)
        The number of random projection trees to use in initializing searching or
        querying.

        .. deprecated::  0.5.5

    search_epsilon: float (optional, default=0.1)
        When searching for nearest neighbors of a query point this values
        controls the trade-off between accuracy and search cost. Larger values
        produce more accurate nearest neighbor results at larger computational
        cost for the search. Values should be in the range 0.0 to 0.5, but
        should probably not exceed 0.3 without good reason.

    tree_init: bool (optional, default=True)
        Whether to use random projection trees for initialization.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs: int or None (optional, default=None)
        The maximum number of parallel threads to be run at a time. If none
        this will default to using all the cores available. Note that there is
        not perfect parallelism, so at several pints the algorithm will be
        single threaded.

    low_memory: boolean (optional, default=False)
        Whether to use a lower memory, but more computationally expensive
        approach to index construction. This defaults to false as for most
        cases it speeds index construction, but if you are having issues
        with excessive memory use for your dataset consider setting this
        to True.

    max_candidates: int (optional, default=20)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    n_iters: int (optional, default=None)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing. The default of None means
        a value will be chosen based on the size of the graph_data.

    early_termination_value: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    parallel_batch_queries: bool (optional, default=False)
        Whether to use parallelism of batched queries. This can be useful for large
        batches of queries on multicore machines, but results in performance degradation
        for single queries, so is poor for streaming use.

    verbose: bool (optional, default=False)
        Whether to print status graph_data during the computation.

    Examples
    --------
    >>> from sklearn.manifold import Isomap
    >>> from pynndescent import PyNNDescentTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> estimator = make_pipeline(
    ...     PyNNDescentTransformer(n_neighbors=5),
    ...     Isomap(neighbors_algorithm='precomputed'))
    """

    def __init__(
        self,
        n_neighbors=30,
        metric="euclidean",
        metric_kwds=None,
        n_trees=None,
        leaf_size=None,
        search_epsilon=0.1,
        pruning_degree_multiplier=1.5,
        diversify_prob=1.0,
        n_search_trees=1,
        tree_init=True,
        random_state=None,
        n_jobs=None,
        low_memory=True,
        max_candidates=None,
        n_iters=None,
        early_termination_value=0.001,
        parallel_batch_queries=False,
        verbose=False,
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.search_epsilon = search_epsilon
        self.pruning_degree_multiplier = pruning_degree_multiplier
        self.diversify_prob = diversify_prob
        self.n_search_trees = n_search_trees
        self.tree_init = tree_init
        self.random_state = random_state
        self.low_memory = low_memory
        self.max_candidates = max_candidates
        self.n_iters = n_iters
        self.early_termination_value = early_termination_value
        self.n_jobs = n_jobs
        self.parallel_batch_queries = parallel_batch_queries
        self.verbose = verbose

    def fit(self, X, compress_index=True):
        """Fit the PyNNDescent transformer to build KNN graphs with
        neighbors given by the dataset X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample graph_data

        Returns
        -------
        transformer : PyNNDescentTransformer
            The trained transformer
        """
        self.n_samples_fit = X.shape[0]

        if self.metric_kwds is None:
            metric_kwds = {}
        else:
            metric_kwds = self.metric_kwds

        if self.verbose:
            print(ts(), "Creating index")

        # Compatibility with sklearn, which doesn't consider
        # a point its own neighbor for these purposes.
        effective_n_neighbors = self.n_neighbors + 1

        self.index_ = NNDescent(
            X,
            metric=self.metric,
            metric_kwds=metric_kwds,
            n_neighbors=effective_n_neighbors,
            n_trees=self.n_trees,
            leaf_size=self.leaf_size,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            diversify_prob=self.diversify_prob,
            n_search_trees=self.n_search_trees,
            tree_init=self.tree_init,
            random_state=self.random_state,
            low_memory=self.low_memory,
            max_candidates=self.max_candidates,
            n_iters=self.n_iters,
            delta=self.early_termination_value,
            n_jobs=self.n_jobs,
            compressed=compress_index,
            parallel_batch_queries=self.parallel_batch_queries,
            verbose=self.verbose,
        )

        return self

    def transform(self, X, y=None):
        """Computes the (weighted) graph of Neighbors for points in X

        Parameters
        ----------
        X : array-like, shape (n_samples_transform, n_features)
            Sample graph_data

        Returns
        -------
        Xt : CSR sparse matrix, shape (n_samples_transform, n_samples_fit)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
        """

        if X is None:
            n_samples_transform = self.n_samples_fit
        else:
            n_samples_transform = X.shape[0]

        if X is None:
            indices, distances = self.index_.neighbor_graph
        else:
            indices, distances = self.index_.query(
                X, k=self.n_neighbors, epsilon=self.search_epsilon
            )

        if self.verbose:
            print(ts(), "Constructing neighbor matrix")
        result = coo_matrix((n_samples_transform, self.n_samples_fit), dtype=np.float32)
        result.row = np.repeat(
            np.arange(indices.shape[0], dtype=np.int32), indices.shape[1]
        )
        result.col = indices.ravel()
        result.data = distances.ravel()

        return result.tocsr()

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to graph_data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training set.

        y : ignored

        Returns
        -------
        Xt : CSR sparse matrix, shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
        """
        self.fit(X, compress_index=False)
        result = self.transform(X=None)

        if self.verbose:
            print(ts(), "Compressing index")
        self.index_.compress_index()

        return result
