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
    rejection_sample,
    tau_rand_int,
    seed,
    make_heap,
    heap_push,
    unchecked_heap_push,
    deheap_sort,
    new_build_candidates,
    ts,
)

from pynndescent.rp_trees import make_forest, rptree_leaf_array, search_flat_tree


INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

FLOAT32_EPS = np.finfo(np.float32).eps

@numba.njit(fastmath=True)
def init_from_random(n_neighbors, data, query_points, heap, dist, dist_args, rng_state):
    for i in range(query_points.shape[0]):
        if heap[0, i, 0] == -1:
            for j in range(n_neighbors):
                idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                d = dist(data[idx], query_points[i], *dist_args)
                heap_push(heap, i, d, idx, 1)
    return


@numba.njit(fastmath=True)
def init_from_tree(tree, data, query_points, heap, dist, dist_args, rng_state):
    for i in range(query_points.shape[0]):
        indices = search_flat_tree(
            query_points[i],
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        for j in range(indices.shape[0]):
            if indices[j] >= 0:
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)

    return


def initialise_search(
    forest, n_search_trees, data, query_points, n_neighbors, dist, dist_args, rng_state
):
    results = make_heap(query_points.shape[0], n_neighbors)
    if forest is not None:
        for i in range(n_search_trees):
            tree = forest[i]
            init_from_tree(
                tree, data, query_points, results, dist, dist_args, rng_state
            )
    init_from_random(
        n_neighbors, data, query_points, results, dist, dist_args, rng_state
    )

    return results


@numba.njit(fastmath=True, locals={"candidate": numba.types.int64})
def initialized_nnd_search(
    data, indptr, indices, initialization, query_points, epsilon, dist,
        dist_args
):
    tried = np.zeros(data.shape[0], dtype=np.uint8)

    for i in range(query_points.shape[0]):

        tried[:] = 0
        seed_set = [(np.inf, -1)]
        distance_bound = (1.0 + epsilon) * initialization[1, i, 0]
        current_query = query_points[i]

        for j in range(initialization.shape[2]):
            heapq.heappush(
                seed_set, (initialization[1, i, j], int(initialization[0, i, j]))
            )
            tried[int(initialization[0, i, j])] = 1

        # Find smallest seed point
        d_vertex, vertex = heapq.heappop(seed_set)

        while d_vertex < distance_bound:

            for j in range(indptr[vertex], indptr[vertex + 1]):

                candidate = indices[j]

                if tried[candidate] == 0:

                    tried[candidate] = 1

                    d = dist(data[candidate], current_query, *dist_args)

                    if d < distance_bound:
                        unchecked_heap_push(initialization, i, d, candidate, 1)
                        heapq.heappush(seed_set, (d, candidate))


            distance_bound = (1.0 + epsilon) * initialization[1, i, 0]
            d_vertex, vertex = heapq.heappop(seed_set)

    return initialization


@numba.njit(fastmath=True)
def init_current_graph(
    data, dist, dist_args, n_neighbors, rng_state, seed_per_row=False
):
    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        if seed_per_row:
            seed(rng_state, i)
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]], *dist_args)
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
    return current_graph


@numba.njit(fastmath=True)
def init_rp_tree(data, dist, dist_args, current_graph, leaf_array, tried=None):
    if tried is None:
        tried = set([(-1, -1)])

    for n in range(leaf_array.shape[0]):
        for i in range(leaf_array.shape[1]):
            p = leaf_array[n, i]
            if p < 0:
                break
            for j in range(i + 1, leaf_array.shape[1]):
                q = leaf_array[n, j]
                if q < 0:
                    break
                if (p, q) in tried:
                    continue
                d = dist(data[p], data[q], *dist_args)
                heap_push(current_graph, p, d, q, 1)
                tried.add((p, q))
                if p != q:
                    heap_push(current_graph, q, d, p, 1)
                    tried.add((q, p))


@numba.njit(fastmath=True)
def nn_descent_internal_low_memory(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
    seed_per_row=False,
):
    n_vertices = data.shape[0]

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            n_vertices,
            n_neighbors,
            max_candidates,
            rng_state,
            rho,
            seed_per_row,
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    d = dist(data[p], data[q], *dist_args)
                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    d = dist(data[p], data[q], *dist_args)
                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit(fastmath=True)
def nn_descent_internal_high_memory(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    tried,
    max_candidates=50,
    dist=dist.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
    seed_per_row=False,
):
    n_vertices = data.shape[0]

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            n_vertices,
            n_neighbors,
            max_candidates,
            rng_state,
            rho,
            seed_per_row,
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    d = dist(data[p], data[q], *dist_args)
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    d = dist(data[p], data[q], *dist_args)
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit(fastmath=True)
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rho=0.5,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=False,
    verbose=False,
    seed_per_row=False,
):
    tried = set([(-1, -1)])

    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        if seed_per_row:
            seed(rng_state, i)
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]], *dist_args)
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
            tried.add((i, indices[j]))
            tried.add((indices[j], i))

    if rp_tree_init:
        init_rp_tree(data, dist, dist_args, current_graph, leaf_array, tried=tried)

    if low_memory:
        nn_descent_internal_low_memory(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            dist_args=dist_args,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
            seed_per_row=seed_per_row,
        )
    else:
        nn_descent_internal_high_memory(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            tried,
            max_candidates=max_candidates,
            dist=dist,
            dist_args=dist_args,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
            seed_per_row=seed_per_row,
        )

    return deheap_sort(current_graph)


@numba.njit()
def diversify(indices, distances, data, dist, dist_args, epsilon=0.01):

    for i in range(indices.shape[0]):

        new_indices = [indices[i, 0]]
        new_distances = [distances[i, 0]]
        for j in range(1, indices.shape[1]):
            if indices[i, j] < 0:
                break

            flag = True
            for k in range(len(new_indices)):
                c = new_indices[k]
                d = dist(data[indices[i, j]], data[c], *dist_args)
                if new_distances[k] > FLOAT32_EPS \
                        and d < epsilon * distances[i, j]:
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

    result = graph.tolil()
    for i, row_data in enumerate(result.data):
        if len(row_data) > max_degree:
            cut_value = np.sort(row_data)[max_degree]
            row_data = [x if x <= cut_value else 0.0 for x in row_data]
            result.data[i] = row_data
    result = result.tocsr()
    result.eliminate_zeros()
    return result


class NNDescent(object):
    """NNDescent for fast approximate nearest neighbor queries. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    data in many cases. This implementation provides a straightfoward
    interface, with access to some tuning parameters.

    Parameters
    ----------
    data: array os shape (n_samples, n_features)
        The training data set to find nearest neighbors in.

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

    n_neighbors: int (optional, default=15)
        The number of neighbors to use in k-neighbor graph data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    n_trees: int (optional, default=None)
        This implementation uses random projection forests for initialization
        of searches. This parameter controls the number of trees in that
        forest. A larger number will result in more accurate neighbor
        computation at the cost of performance. The default of None means
        a value will be chosen based on the size of the data.

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
        a value will be chosen based on the size of the data.

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    rho: float (optional, default=0.5)
        Controls the random sampling of potential candidates in any given
        iteration of NN-descent. Larger values will result in less accurate
        indexes and less accurate searching. Don't tweak this value unless
        you know what you're doing.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose: bool (optional, default=False)
        Whether to print status data during the computation.
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
        diversify_epsilon=0.5,
        tree_init=True,
        random_state=np.random,
        algorithm="standard",
        low_memory=False,
        max_candidates=20,
        n_iters=None,
        delta=0.001,
        rho=0.5,
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
        self.max_candidates = max_candidates
        self.low_memory = low_memory
        self.n_iters = n_iters
        self.delta = delta
        self.rho = rho
        self.dim = data.shape[1]
        self.verbose = verbose

        data = check_array(data, dtype=np.float32, accept_sparse="csr", order='C')
        self._raw_data = data

        if not tree_init or n_trees == 0:
            self.tree_init = False
        else:
            self.tree_init = True

        metric_kwds = metric_kwds or {}
        self._dist_args = tuple(metric_kwds.values())

        self.random_state = check_random_state(random_state)

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

        self.rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
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
                self._angular_trees,
            )
            leaf_array = rptree_leaf_array(self._rp_forest)
        else:
            self._rp_forest = None
            leaf_array = np.array([[-1]])

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
                        "Metric {} not supported for sparse data".format(metric)
                    )
                self._neighbor_graph = sparse_threaded.sparse_nn_descent(
                    self._raw_data.indices,
                    self._raw_data.indptr,
                    self._raw_data.data,
                    self._raw_data.shape[0],
                    self.n_neighbors,
                    self.rng_state,
                    self.max_candidates,
                    self._distance_func,
                    self._dist_args,
                    self.n_iters,
                    self.delta,
                    self.rho,
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
                    self.max_candidates,
                    self._distance_func,
                    self._dist_args,
                    self.n_iters,
                    self.delta,
                    self.rho,
                    rp_tree_init=self.tree_init,
                    leaf_array=leaf_array,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    seed_per_row=seed_per_row,
                )
        elif algorithm == "standard" or leaf_array.shape[0] == 1:
            if isspmatrix_csr(self._raw_data):

                self._is_sparse = True

                if metric in sparse.sparse_named_distances:
                    self._distance_func = sparse.sparse_named_distances[metric]
                    if metric in sparse.sparse_need_n_features:
                        metric_kwds["n_features"] = self._raw_data.shape[1]
                    self._dist_args = tuple(metric_kwds.values())
                    if self._distance_correction is not None:
                        self._distance_correction = None
                else:
                    raise ValueError(
                        "Metric {} not supported for sparse data".format(metric)
                    )

                if verbose:
                    print(ts(), "metric NN descent for", str(n_iters), "iterations")

                self._neighbor_graph = sparse_nnd.sparse_nn_descent(
                    self._raw_data.indices,
                    self._raw_data.indptr,
                    self._raw_data.data,
                    self._raw_data.shape[0],
                    self.n_neighbors,
                    self.rng_state,
                    self.max_candidates,
                    rho=self.rho,
                    low_memory=self.low_memory,
                    sparse_dist=self._distance_func,
                    dist_args=self._dist_args,
                    n_iters=self.n_iters,
                    rp_tree_init=False,
                    leaf_array=leaf_array,
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
                    self.max_candidates,
                    self._distance_func,
                    self._dist_args,
                    self.n_iters,
                    self.delta,
                    self.rho,
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

        if self._is_sparse:
            diversified_rows, diversified_data = sparse.sparse_diversify(
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

        self._search_graph = self._search_graph.maximum(
            self._search_graph.transpose()
        ).tocsr()
        self._search_graph.eliminate_zeros()

        self._search_graph = degree_prune(self._search_graph,
                                          int(np.round(self.prune_degree_multiplier
                                                        *
                                                        self.n_neighbors)))
        self._search_graph.eliminate_zeros()
        self._search_graph = (self._search_graph != 0).astype(np.int8)


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


    def query(self, query_data, k=10, epsilon=0.1, n_search_trees=1, queue_size=1.0):
        """Query the training data for the k nearest neighbors

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
            The first array, ``indices``, provides the indices of the data
            points in the training set that are the nearest neighbors of
            each query point. Thus ``indices[i, j]`` is the index into the
            training data of the jth nearest neighbor of the ith query points.

            Similarly ``distances`` provides the distances to the neighbors
            of the query points such that ``distances[i, j]`` is the distance
            from the ith query point to its jth nearest neighbor in the
            training data.
        """
        if not self._is_sparse:
            # Standard case
            # query_data = check_array(query_data, dtype=np.float64, order='C')
            query_data = np.asarray(query_data).astype(np.float32, order='C')
            self._init_search_graph()
            init = initialise_search(
                self._rp_forest,
                n_search_trees,
                self._raw_data,
                query_data,
                int(k * queue_size),
                self._distance_func,
                self._dist_args,
                self.rng_state,
            )
            result = initialized_nnd_search(
                self._raw_data,
                self._search_graph.indptr,
                self._search_graph.indices,
                init,
                query_data,
                epsilon,
                self._distance_func,
                self._dist_args,
            )
        else:
            # Sparse case
            query_data = check_array(query_data, accept_sparse="csr")
            if not isspmatrix_csr(query_data):
                query_data = csr_matrix(query_data)
            self._init_search_graph()
            init = sparse_nnd.sparse_initialise_search(
                self._rp_forest,
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                query_data.indices,
                query_data.indptr,
                query_data.data,
                int(k * queue_size),
                self.rng_state,
                self._distance_func,
                self._dist_args,
            )
            result = sparse_nnd.sparse_initialized_nnd_search(
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                self._search_graph.indptr,
                self._search_graph.indices,
                init,
                query_data.indices,
                query_data.indptr,
                query_data.data,
                epsilon,
                self._distance_func,
                self._dist_args,
            )

        indices, dists = deheap_sort(result)
        indices, dists = indices[:, :k], dists[:, :k]
        if self._distance_correction is not None:
            dists = self._distance_correction(dists)
        return indices, dists


class PyNNDescentTransformer(BaseEstimator, TransformerMixin):
    """PyNNDescentTransformer for fast approximate nearest neighbor transformer.
    It uses the NNDescent algorithm, and is thus
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    data in many cases.

    Transform X into a (weighted) graph of k nearest neighbors

    The transformed data is a sparse graph as returned by kneighbors_graph.

    Parameters
    ----------
    n_neighbors: int (optional, default=5)
        The number of neighbors to use in k-neighbor graph data structure
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
        a value will be chosen based on the size of the data.

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
        a value will be chosen based on the size of the data.

    early_termination_value: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    sampling_rate: float (optional, default=0.5)
        Controls the random sampling of potential candidates in any given
        iteration of NN-descent. Larger values will result in less accurate
        indexes and less accurate searching. Don't tweak this value unless
        you know what you're doing.

    verbose: bool (optional, default=False)
        Whether to print status data during the computation.

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
        n_neighbors=5,
        metric="euclidean",
        metric_kwds=None,
        n_trees=None,
        leaf_size=None,
        search_queue_size=1.0,
        search_epsilon=0.1,
        pruning_degree_multiplier=2.0,
        diversify_epsilon=0.5,
        tree_init=True,
        random_state=np.random,
        algorithm="standard",
        low_memory=False,
        max_candidates=20,
        n_iters=None,
        early_termination_value=0.001,
        sampling_rate=0.5,
        verbose=False,
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.search_queue_size = search_queue_size
        self.search_epsilon = search_epsilon
        self.pruning_degree_multiplier = pruning_degree_multiplier
        self.diversify_epsilon = diversify_epsilon
        self.tree_init = tree_init
        self.random_state = random_state
        self.algorithm = algorithm
        self.low_memory = low_memory
        self.max_candidates = max_candidates
        self.n_iters = n_iters
        self.early_termination_value = early_termination_value
        self.sampling_rate = sampling_rate
        self.verbose = verbose

    def fit(self, X):
        """Fit the PyNNDescent transformer to build KNN graphs with
        neighbors given by the dataset X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample data

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

        self.pynndescent_ = NNDescent(
            X,
            self.metric,
            metric_kwds,
            self.n_neighbors,
            self.n_trees,
            self.leaf_size,
            self.pruning_degree_multiplier,
            self.diversify_epsilon,
            self.tree_init,
            self.random_state,
            self.algorithm,
            self.low_memory,
            self.max_candidates,
            self.n_iters,
            self.early_termination_value,
            self.sampling_rate,
            verbose=self.verbose,
        )

        return self

    def transform(self, X, y=None):
        """Computes the (weighted) graph of Neighbors for points in X

        Parameters
        ----------
        X : array-like, shape (n_samples_transform, n_features)
            Sample data

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
            indices, distances = self.pynndescent_.neighbor_graph
        else:
            indices, distances = self.pynndescent_.query(
                X,
                k=self.n_neighbors,
                queue_size=self.search_queue_size,
                epsilon=self.search_epsilon,
            )

        result = lil_matrix((n_samples_transform, self.n_samples_fit), dtype=np.float32)
        result.rows = indices
        result.data = distances

        return result.tocsr()

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

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
