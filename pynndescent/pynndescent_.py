# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

from warnings import warn

import numba
import numpy as np
from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_csr

import heapq

import pynndescent.sparse as sparse
import pynndescent.sparse_nndescent as sparse_nnd
import pynndescent.distances as dist
import pynndescent.threaded as threaded
import pynndescent.sparse_threaded as sparse_threaded

from pynndescent.utils import (
    tau_rand_int,
    make_heap,
    heap_push,
    seed,
    deheap_sort,
    new_build_candidates,
    ts,
    simple_heap_push,
    has_been_visited,
    mark_visited,
    apply_graph_updates_high_memory,
    apply_graph_updates_low_memory,
)

from pynndescent.rp_trees import (
    make_forest,
    rptree_leaf_array,
    search_flat_tree,
    convert_tree_format,
    FlatTree,
)

update_type = numba.types.List(
    numba.types.List((numba.types.int64, numba.types.int64, numba.types.float64))
)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

FLOAT32_EPS = np.finfo(np.float32).eps


@numba.njit(
    fastmath=True,
    locals={
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
)
def search_from_init(
    current_query,
    data,
    indptr,
    indices,
    heap_priorities,
    heap_indices,
    epsilon,
    visited,
    dist,
    dist_args,
):
    distance_scale = 1.0 + epsilon
    distance_bound = distance_scale * heap_priorities[0]
    heap_size = heap_priorities.shape[0]

    seed_set = [(heap_priorities[j], heap_indices[j]) for j in range(heap_size)]
    heapq.heapify(seed_set)

    # Find smallest seed point
    d_vertex, vertex = heapq.heappop(seed_set)

    while d_vertex < distance_bound:

        for j in range(indptr[vertex], indptr[vertex + 1]):

            candidate = indices[j]

            if has_been_visited(visited, candidate) == 0:
                mark_visited(visited, candidate)

                d = dist(data[candidate], current_query, *dist_args)

                if d < distance_bound:
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    # Update bound
                    distance_bound = distance_scale * heap_priorities[0]

        # find new smallest seed point
        if len(seed_set) == 0:
            break
        else:
            d_vertex, vertex = heapq.heappop(seed_set)

    return heap_priorities, heap_indices


@numba.njit(
    fastmath=True,
    locals={
        "heap_priorities": numba.types.float32[::1],
        "heap_indices": numba.types.int32[::1],
        "indices": numba.types.int32[::1],
        "candidate": numba.types.int32,
        "current_query": numba.types.float32[::1],
        "d": numba.types.float32,
        "n_random_samples": numba.types.int32,
        "visited": numba.types.uint8[::1],
    },
)
def search_init(
    current_query, k, data, forest, n_neighbors, visited, dist, dist_args, rng_state
):

    heap_priorities = np.float32(np.inf) + np.zeros(k, dtype=np.float32)
    heap_indices = np.int32(-1) + np.zeros(k, dtype=np.int32)

    n_random_samples = min(k, n_neighbors)

    for tree in forest:
        indices = search_flat_tree(
            current_query,
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        n_initial_points = indices.shape[0]
        n_random_samples = min(k, n_neighbors) - n_initial_points

        for j in range(n_initial_points):
            candidate = indices[j]
            d = dist(data[candidate], current_query, *dist_args)
            # indices are guaranteed different
            simple_heap_push(heap_priorities, heap_indices, d, candidate)
            mark_visited(visited, candidate)

    if n_random_samples > 0:
        for i in range(n_random_samples):
            candidate = np.abs(tau_rand_int(rng_state)) % data.shape[0]
            if has_been_visited(visited, candidate) == 0:
                d = dist(data[candidate], current_query, *dist_args)
                simple_heap_push(heap_priorities, heap_indices, d, candidate)
                mark_visited(visited, candidate)

    return heap_priorities, heap_indices


@numba.njit(
    locals={
        "current_query": numba.types.float32[::1],
        "i": numba.types.uint32,
        "heap_priorities": numba.types.float32[::1],
        "heap_indices": numba.types.int32[::1],
        "result": numba.types.float32[:, :, ::1],
    }
)
def search(
    query_points,
    k,
    data,
    forest,
    indptr,
    indices,
    epsilon,
    n_neighbors,
    visited,
    dist,
    dist_args,
    rng_state,
):

    result = make_heap(query_points.shape[0], k)
    for i in range(query_points.shape[0]):
        visited[:] = 0
        current_query = query_points[i]
        heap_priorities, heap_indices = search_init(
            current_query,
            k,
            data,
            forest,
            n_neighbors,
            visited,
            dist,
            dist_args,
            rng_state,
        )
        heap_priorities, heap_indices = search_from_init(
            current_query,
            data,
            indptr,
            indices,
            heap_priorities,
            heap_indices,
            epsilon,
            visited,
            dist,
            dist_args,
        )

        result[0, i] = heap_indices
        result[1, i] = heap_priorities

    return result


@numba.njit(parallel=True)
def generate_leaf_updates(leaf_block, dist_thresholds, data, dist, dist_args):

    updates = [[(-1, -1, np.inf)] for i in range(leaf_block.shape[0])]

    for n in numba.prange(leaf_block.shape[0]):
        for i in range(leaf_block.shape[1]):
            p = leaf_block[n, i]
            if p < 0:
                break

            for j in range(i + 1, leaf_block.shape[1]):
                q = leaf_block[n, j]
                if q < 0:
                    break

                d = dist(data[p], data[q], *dist_args)
                if d < dist_thresholds[p] or d < dist_thresholds[q]:
                    updates[n].append((p, q, d))

    return updates


@numba.njit()
def init_rp_tree(data, dist, dist_args, current_graph, leaf_array):

    n_leaves = leaf_array.shape[0]
    block_size = 65536
    n_blocks = n_leaves // block_size

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1, :, 0]

        updates = generate_leaf_updates(
            leaf_block, dist_thresholds, data, dist, dist_args
        )

        for j in range(len(updates)):
            for k in range(len(updates[j])):
                p, q, d = updates[j][k]

                if p == -1 or q == -1:
                    continue

                heap_push(current_graph, p, d, q, 1)
                heap_push(current_graph, q, d, p, 1)


@numba.njit(fastmath=True)
def init_random(
    n_neighbors, data, heap, dist, dist_args, rng_state, seed_per_row=False
):
    for i in range(data.shape[0]):
        if seed_per_row:
            seed(rng_state, i)
        if heap[0, i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0, i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                d = dist(data[idx], data[i], *dist_args)
                heap_push(heap, i, d, idx, 1)

    return


@numba.njit(parallel=True)
def generate_graph_updates(
    new_candidate_block, old_candidate_block, dist_thresholds, data, dist, dist_args
):

    block_size = new_candidate_block.shape[0]
    updates = [[(-1, -1, np.inf)] for i in range(block_size)]
    max_candidates = new_candidate_block.shape[1]

    for i in numba.prange(block_size):
        for j in range(max_candidates):
            p = int(new_candidate_block[i, j])
            if p < 0:
                continue
            for k in range(j, max_candidates):
                q = int(new_candidate_block[i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q], *dist_args)
                if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                    updates[i].append((p, q, d))

            for k in range(max_candidates):
                q = int(old_candidate_block[i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q], *dist_args)
                if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                    updates[i].append((p, q, d))

    return updates


@numba.njit()
def nn_descent_internal_low_memory_parallel(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    verbose=False,
    seed_per_row=False,
):
    n_vertices = data.shape[0]
    block_size = 16384
    n_blocks = n_vertices // block_size

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            n_vertices,
            n_neighbors,
            max_candidates,
            rng_state,
            seed_per_row,
        )

        c = 0
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[0, block_start:block_end]
            old_candidate_block = old_candidate_neighbors[0, block_start:block_end]
            dist_thresholds = current_graph[1, :, 0]

            updates = generate_graph_updates(
                new_candidate_block,
                old_candidate_block,
                dist_thresholds,
                data,
                dist,
                dist_args,
            )

            c += apply_graph_updates_low_memory(current_graph, updates)

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit()
def nn_descent_internal_high_memory_parallel(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    verbose=False,
    seed_per_row=False,
):
    n_vertices = data.shape[0]
    block_size = 16384
    n_blocks = n_vertices // block_size

    in_graph = [
        set(current_graph[0, i].astype(np.int64)) for i in range(current_graph.shape[1])
    ]

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            n_vertices,
            n_neighbors,
            max_candidates,
            rng_state,
            seed_per_row,
        )

        c = 0
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[0, block_start:block_end]
            old_candidate_block = old_candidate_neighbors[0, block_start:block_end]
            dist_thresholds = current_graph[1, :, 0]

            updates = generate_graph_updates(
                new_candidate_block,
                old_candidate_block,
                dist_thresholds,
                data,
                dist,
                dist_args,
            )

            c += apply_graph_updates_high_memory(current_graph, updates, in_graph)

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit()
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=False,
    verbose=False,
    seed_per_row=False,
):

    current_graph = make_heap(data.shape[0], n_neighbors)

    if rp_tree_init:
        init_rp_tree(data, dist, dist_args, current_graph, leaf_array)

    init_random(
        n_neighbors, data, current_graph, dist, dist_args, rng_state, seed_per_row
    )

    if low_memory:
        nn_descent_internal_low_memory_parallel(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            dist_args=dist_args,
            n_iters=n_iters,
            delta=delta,
            verbose=verbose,
            seed_per_row=seed_per_row,
        )
    else:
        nn_descent_internal_high_memory_parallel(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            dist_args=dist_args,
            n_iters=n_iters,
            delta=delta,
            verbose=verbose,
            seed_per_row=seed_per_row,
        )

    return deheap_sort(current_graph)


@numba.njit(parallel=True)
def diversify(indices, distances, data, dist, dist_args, epsilon=0.01):

    for i in numba.prange(indices.shape[0]):

        new_indices = [indices[i, 0]]
        new_distances = [distances[i, 0]]
        for j in range(1, indices.shape[1]):
            if indices[i, j] < 0:
                break

            flag = True
            for k in range(len(new_indices)):
                c = new_indices[k]
                d = dist(data[indices[i, j]], data[c], *dist_args)
                if new_distances[k] > FLOAT32_EPS and d < epsilon * distances[i, j]:
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


@numba.njit(parallel=True)
def diversify_csr(
    graph_indptr, graph_indices, graph_data, source_data, dist, dist_args, epsilon=0.01
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
                if retained[k] == 1:
                    d = dist(
                        source_data[current_indices[j]],
                        source_data[current_indices[k]],
                        *dist_args
                    )
                    if current_data[k] > FLOAT32_EPS and d < epsilon * current_data[j]:
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


class NNDescent(object):
    """NNDescent for fast approximate nearest neighbor queries. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    graph_data in many cases. This implementation provides a straightfoward
    interface, with access to some tuning parameters.

    Parameters
    ----------
    data: array os shape (n_samples, n_features)
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
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    n_neighbors: int (optional, default=15)
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    n_trees: int (optional, default=None)
        This implementation uses random projection forests for initializing the index
        build process. This parameter controls the number of trees in that forest. A
        larger number will result in more accurate neighbor computation at the cost
        of performance. The default of None means a value will be chosen based on the
        size of the graph_data.

    leaf_size: int (optional, default=None)
        The maximum number of points in a leaf for the random projection trees.
        The default of None means a value will be chosen based on n_neighbors.

    pruning_degree_multiplier: float (optional, default=2.0)
        How aggressively to prune the graph. Since the search graph is undirected
        (and thus includes nearest neighbors and reverse nearest neighbors) vertices
        can have very high degree -- the graph will be pruned such that no
        vertex has degree greater than
        ``pruning_degree_multiplier * n_neighbors``.

    diversify_epsilon: float (optional, default=1.0)
        The search graph get "diversified" by removing potentially unnecessary
        edges. This controls the volume of edges removed. A value of 0.0 ensures
        that no edges get removed, and larger values result in significantly more
        aggressive edge removal. Values above 1.0 are not recommended.

    tree_init: bool (optional, default=True)
        Whether to use random projection trees for initialization.

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

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose: bool (optional, default=False)
        Whether to print status graph_data during the computation.
    """

    def __init__(
        self,
        data,
        metric="euclidean",
        metric_kwds=None,
        n_neighbors=15,
        n_trees=None,
        leaf_size=None,
        pruning_degree_multiplier=2.0,
        diversify_epsilon=1.0,
        n_search_trees=1,
        tree_init=True,
        random_state=None,
        algorithm="standard",
        low_memory=False,
        max_candidates=None,
        n_iters=None,
        delta=0.001,
        n_jobs=None,
        seed_per_row=False,
        verbose=False,
    ):

        if n_trees is None:
            n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
        if n_iters is None:
            n_iters = max(5, int(round(np.log2(data.shape[0]))))

        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.leaf_size = leaf_size
        self.prune_degree_multiplier = pruning_degree_multiplier
        self.diversify_epsilon = diversify_epsilon
        self.n_search_trees = n_search_trees
        self.max_candidates = max_candidates
        self.low_memory = low_memory
        self.n_iters = n_iters
        self.delta = delta
        self.dim = data.shape[1]
        self.n_jobs = n_jobs
        self.verbose = verbose

        data = check_array(data, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = data

        if not tree_init or n_trees == 0:
            self.tree_init = False
        else:
            self.tree_init = True

        metric_kwds = metric_kwds or {}
        self._dist_args = tuple(metric_kwds.values())

        self.random_state = random_state

        current_random_state = check_random_state(self.random_state)

        self._distance_correction = None

        if callable(metric):
            self._distance_func = metric
        elif metric in dist.named_distances:
            if metric in dist.fast_distance_alternatives:
                self._distance_func = dist.fast_distance_alternatives[metric]["dist"]
                self._distance_correction = dist.fast_distance_alternatives[metric][
                    "correction"
                ]
            else:
                self._distance_func = dist.named_distances[metric]
        else:
            raise ValueError("Metric is neither callable, " + "nor a recognised string")

        if metric in ("cosine", "correlation", "dice", "jaccard"):
            self._angular_trees = True
        else:
            self._angular_trees = False

        self.rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )

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
            )
            leaf_array = rptree_leaf_array(self._rp_forest)
        else:
            self._rp_forest = None
            leaf_array = np.array([[-1]])

        if self.max_candidates is None:
            effective_max_candidates = min(60, self.n_neighbors)
        else:
            effective_max_candidates = self.max_candidates

        if threaded.effective_n_jobs_with_context(n_jobs) != 1:
            if algorithm != "standard":
                raise ValueError(
                    "Algorithm {} not supported in parallel mode".format(algorithm)
                )
            if verbose:
                print(ts(), "parallel NN descent for", str(n_iters), "iterations")

            if isspmatrix_csr(self._raw_data):
                # Sparse case
                self._is_sparse = True
                if metric in sparse.sparse_named_distances:
                    self._distance_func = sparse.sparse_named_distances[metric]
                    if metric in sparse.sparse_need_n_features:
                        metric_kwds["n_features"] = self._raw_data.shape[1]
                    self._dist_args = tuple(metric_kwds.values())
                else:
                    raise ValueError(
                        "Metric {} not supported for sparse graph_data".format(metric)
                    )
                self._neighbor_graph = sparse_threaded.sparse_nn_descent(
                    self._raw_data.indices,
                    self._raw_data.indptr,
                    self._raw_data.data,
                    self._raw_data.shape[0],
                    self.n_neighbors,
                    self.rng_state,
                    effective_max_candidates,
                    self._distance_func,
                    self._dist_args,
                    self.n_iters,
                    self.delta,
                    rp_tree_init=self.tree_init,
                    leaf_array=leaf_array,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    seed_per_row=seed_per_row,
                )
            else:
                # Regular case
                self._is_sparse = False
                self._neighbor_graph = threaded.nn_descent(
                    self._raw_data,
                    self.n_neighbors,
                    self.rng_state,
                    effective_max_candidates,
                    self._distance_func,
                    self._dist_args,
                    self.n_iters,
                    self.delta,
                    rp_tree_init=self.tree_init,
                    leaf_array=leaf_array,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    seed_per_row=seed_per_row,
                )
        elif algorithm == "standard" or leaf_array.shape[0] == 1:
            if isspmatrix_csr(self._raw_data):

                self._is_sparse = True

                if not self._raw_data.has_sorted_indices:
                    self._raw_data.sort_indices()

                if metric in sparse.sparse_named_distances:
                    self._distance_func = sparse.sparse_named_distances[metric]
                    if metric in sparse.sparse_need_n_features:
                        metric_kwds["n_features"] = self._raw_data.shape[1]
                    self._dist_args = tuple(metric_kwds.values())
                    if self._distance_correction is not None:
                        self._distance_correction = None
                else:
                    raise ValueError(
                        "Metric {} not supported for sparse graph_data".format(metric)
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
                    dist_args=self._dist_args,
                    n_iters=self.n_iters,
                    delta=self.delta,
                    rp_tree_init=True,
                    leaf_array=leaf_array,
                    low_memory=self.low_memory,
                    verbose=verbose,
                )

            else:

                self._is_sparse = False

                if verbose:
                    print(ts(), "NN descent for", str(n_iters), "iterations")

                self._neighbor_graph = nn_descent(
                    self._raw_data,
                    self.n_neighbors,
                    self.rng_state,
                    effective_max_candidates,
                    self._distance_func,
                    self._dist_args,
                    self.n_iters,
                    self.delta,
                    low_memory=self.low_memory,
                    rp_tree_init=True,
                    leaf_array=leaf_array,
                    verbose=verbose,
                    seed_per_row=seed_per_row,
                )
        else:
            raise ValueError("Unknown algorithm selected")

        if np.any(self._neighbor_graph[0] < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                "Results may be less than ideal. Try re-running with"
                "different parameters."
            )

    def _init_search_graph(self):
        if hasattr(self, "_search_graph"):
            return

        self._rp_forest = [
            convert_tree_format(tree, self._raw_data.shape[0])
            for tree in self._rp_forest
        ]

        if self._is_sparse:
            diversified_rows, diversified_data = sparse.diversify(
                self._neighbor_graph[0],
                self._neighbor_graph[1],
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                self._distance_func,
                self._dist_args,
                self.diversify_epsilon,
            )
        else:
            diversified_rows, diversified_data = diversify(
                self._neighbor_graph[0],
                self._neighbor_graph[1],
                self._raw_data,
                self._distance_func,
                self._dist_args,
                self.diversify_epsilon,
            )

        self._search_graph = lil_matrix(
            (self._raw_data.shape[0], self._raw_data.shape[0]), dtype=np.float32
        )

        # Preserve any distance 0 points
        diversified_data[diversified_data == 0.0] = FLOAT32_EPS

        self._search_graph.rows = diversified_rows
        self._search_graph.data = diversified_data

        # Get rid of any -1 index entries
        self._search_graph = self._search_graph.tocsr()
        self._search_graph.data[self._search_graph.indices == -1] = 0.0
        self._search_graph.eliminate_zeros()

        # Reverse graph
        reverse_graph = lil_matrix(
            (self._raw_data.shape[0], self._raw_data.shape[0]), dtype=np.float32
        )
        reverse_data = self._neighbor_graph[1].copy()
        reverse_data[reverse_data == 0.0] = FLOAT32_EPS
        reverse_graph.rows = self._neighbor_graph[0]
        reverse_graph.data = reverse_data
        reverse_graph = reverse_graph.tocsr()
        reverse_graph.data[reverse_graph.indices == -1] = 0.0
        reverse_graph.eliminate_zeros()
        reverse_graph = reverse_graph.transpose()
        if self._is_sparse:
            sparse.diversify_csr(
                reverse_graph.indptr,
                reverse_graph.indices,
                reverse_graph.data,
                self._raw_data.indptr,
                self._raw_data.indices,
                self._raw_data.data,
                self._distance_func,
                self._dist_args,
                self.diversify_epsilon,
            )
            pass
        else:
            diversify_csr(
                reverse_graph.indptr,
                reverse_graph.indices,
                reverse_graph.data,
                self._raw_data,
                self._distance_func,
                self._dist_args,
                self.diversify_epsilon,
            )
        reverse_graph.eliminate_zeros()

        self._search_graph = self._search_graph.maximum(reverse_graph).tocsr()

        # Eliminate the diagonal
        n_vertices = self._search_graph.shape[0]
        self._search_graph[np.arange(n_vertices), np.arange(n_vertices)] = 0.0

        self._search_graph.eliminate_zeros()

        self._search_graph = degree_prune(
            self._search_graph,
            int(np.round(self.prune_degree_multiplier * self.n_neighbors)),
        )
        self._search_graph.eliminate_zeros()
        self._search_graph = (self._search_graph != 0).astype(np.int8)

        self._visited = np.zeros(
            (self._raw_data.shape[0] // 8) + 1, dtype=np.uint8, order="C"
        )

        # reorder according to the search tree leaf order
        self._vertex_order = self._rp_forest[0].indices
        row_ordered_graph = self._search_graph[self._vertex_order, :]
        self._search_graph = row_ordered_graph[:, self._vertex_order]
        self._search_graph = self._search_graph.tocsr()
        self._search_graph.sort_indices()

        if self._is_sparse:
            self._raw_data = self._raw_data[self._vertex_order, :]
        else:
            self._raw_data = np.ascontiguousarray(self._raw_data[self._vertex_order, :])

        tree_order = np.argsort(self._vertex_order)
        self._search_forest = tuple(
            resort_tree_indices(tree, tree_order)
            for tree in self._rp_forest[: self.n_search_trees]
        )

    @property
    def neighbor_graph(self):
        if self._distance_correction is not None:
            result = (
                self._neighbor_graph[0].copy(),
                self._distance_correction(self._neighbor_graph[1]),
            )
        else:
            result = (self._neighbor_graph[0].copy(), self._neighbor_graph[1].copy())

        return result

    def query(self, query_data, k=10, epsilon=0.1):
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

        n_search_trees: int (default 1)
            The number of random projection trees to use in initializing the
            search. More trees will tend to produce more accurate results,
            but cost runtime performance.

        queue_size: float (default 1.0)
            The multiplier of the internal search queue. This controls the
            speed/accuracy tradeoff. Low values will search faster but with
            more approximate results. High values will search more
            accurately, but will require more computation to do so. Values
            should generally be in the range 1.0 to 10.0.

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
        if not self._is_sparse:
            # Standard case
            # query_data = check_array(query_data, dtype=np.float64, order='C')
            query_data = np.asarray(query_data).astype(np.float32, order="C")
            self._init_search_graph()
            result = search(
                query_data,
                k,
                self._raw_data,
                self._search_forest,
                self._search_graph.indptr,
                self._search_graph.indices,
                epsilon,
                self.n_neighbors,
                self._visited,
                self._distance_func,
                self._dist_args,
                self.rng_state,
            )
        else:
            # Sparse case
            query_data = check_array(query_data, accept_sparse="csr", dtype=np.float32)
            if not isspmatrix_csr(query_data):
                query_data = csr_matrix(query_data, dtype=np.float32)
            if not query_data.has_sorted_indices:
                query_data.sort_indices()
            self._init_search_graph()

            result = sparse_nnd.search(
                query_data.indices,
                query_data.indptr,
                query_data.data,
                k,
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                self._search_forest,
                self._search_graph.indptr,
                self._search_graph.indices,
                epsilon,
                self.n_neighbors,
                self._visited,
                self._distance_func,
                self._dist_args,
                self.rng_state,
            )

        indices, dists = deheap_sort(result)
        indices, dists = indices[:, :k], dists[:, :k]
        # Sort to input graph_data order
        indices = self._vertex_order[indices]

        if self._distance_correction is not None:
            dists = self._distance_correction(dists)

        return indices, dists


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

    pruning_degree_multiplier: float (optional, default=2.0)
        How aggressively to prune the graph. Since the search graph is undirected
        (and thus includes nearest neighbors and reverse nearest neighbors) vertices
        can have very high degree -- the graph will be pruned such that no
        vertex has degree greater than
        ``pruning_degree_multiplier * n_neighbors``.

    diversify_epsilon: float (optional, default=0.5)
        The search graph get "diversified" by removing potentially unnecessary
        edges. This controls the volume of edges removed. A value of 0.0 ensures
        that no edges get removed, and larger values result in significantly more
        aggressive edge removal. Values above 1.0 are not recommended.

    n_search_trees: float (optional, default=1)
        The number of random projection trees to use in initializing searching or
        querying.

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

    algorithm: string (optional, default='standard')
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        To use the alternative algorithm specify ``'alternative'``.


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
        n_neighbors=15,
        metric="euclidean",
        metric_kwds=None,
        n_trees=None,
        leaf_size=None,
        search_epsilon=0.1,
        pruning_degree_multiplier=2.0,
        diversify_epsilon=1.0,
        n_search_trees=1,
        tree_init=True,
        random_state=None,
        algorithm="standard",
        low_memory=False,
        max_candidates=None,
        n_iters=None,
        early_termination_value=0.001,
        verbose=False,
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.search_epsilon = search_epsilon
        self.pruning_degree_multiplier = pruning_degree_multiplier
        self.diversify_epsilon = diversify_epsilon
        self.n_search_trees = n_search_trees
        self.tree_init = tree_init
        self.random_state = random_state
        self.algorithm = algorithm
        self.low_memory = low_memory
        self.max_candidates = max_candidates
        self.n_iters = n_iters
        self.early_termination_value = early_termination_value
        self.verbose = verbose

    def fit(self, X):
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

        self.index_ = NNDescent(
            X,
            self.metric,
            metric_kwds,
            self.n_neighbors,
            self.n_trees,
            self.leaf_size,
            self.pruning_degree_multiplier,
            self.diversify_epsilon,
            self.n_search_trees,
            self.tree_init,
            self.random_state,
            self.algorithm,
            self.low_memory,
            self.max_candidates,
            self.n_iters,
            self.early_termination_value,
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

        result = lil_matrix((n_samples_transform, self.n_samples_fit), dtype=np.float32)
        result.rows = indices
        result.data = distances

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
        return self.fit(X).transform(X=None)
