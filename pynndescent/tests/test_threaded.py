import joblib
import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal
from nose import SkipTest

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from pynndescent import distances
from pynndescent import pynndescent_
from pynndescent import NNDescent
from pynndescent import threaded
from pynndescent import utils

from pynndescent.rp_trees import make_forest, rptree_leaf_array

data = np.array(
    [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
)
chunk_size = 4
n_neighbors = 2
max_candidates = 8

dist = distances.named_distances["euclidean"]
dist_args = ()

# In all tests, set seed_per_row=True so that we can comapre the regular
# algorithm against the threded algorithm.


def new_rng_state():
    return np.empty((3,), dtype=np.int64)


def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return (
        np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum()
        / expected.size
    )


def test_effective_n_jobs_with_context():
    assert_equal(threaded.effective_n_jobs_with_context(), 1, "Default to 1 job")
    assert_equal(
        threaded.effective_n_jobs_with_context(-1),
        joblib.cpu_count(),
        "Use all cores with num_jobs=-1",
    )
    assert_equal(
        threaded.effective_n_jobs_with_context(2), 2, "Use n_jobs if specified"
    )
    with joblib.parallel_backend("threading"):
        assert_equal(
            threaded.effective_n_jobs_with_context(),
            joblib.cpu_count(),
            "Use all cores with context manager",
        )
    with joblib.parallel_backend("threading", n_jobs=3):
        assert_equal(
            threaded.effective_n_jobs_with_context(),
            3,
            "Use n_jobs from context manager",
        )
    with joblib.parallel_backend("threading", n_jobs=3):
        assert_equal(
            threaded.effective_n_jobs_with_context(2),
            2,
            "Use n_jobs specified rather than from context manager",
        )


def test_init_random():
    current_graph = utils.make_heap(data.shape[0], n_neighbors)
    pynndescent_.init_random(
        n_neighbors, data, current_graph, dist, new_rng_state(), seed_per_row=True,
    )
    parallel = joblib.Parallel(n_jobs=2, prefer="threads")
    current_graph_threaded = utils.make_heap(data.shape[0], n_neighbors)
    threaded.init_random(
        current_graph_threaded,
        data,
        dist,
        n_neighbors,
        chunk_size=chunk_size,
        rng_state=new_rng_state(),
        parallel=parallel,
        seed_per_row=True,
    )

    assert_allclose(current_graph_threaded, current_graph)


def test_init_rp_tree():

    # Use more graph_data than the other tests since otherwise init_rp_tree has nothing to do
    np.random.seed(42)
    N = 100
    D = 128
    chunk_size = N // 8
    n_neighbors = 25
    data = np.random.rand(N, D).astype(np.float32)

    rng_state = new_rng_state()
    random_state = check_random_state(42)
    current_graph = utils.make_heap(data.shape[0], n_neighbors)
    _rp_forest = make_forest(
        data,
        n_neighbors,
        n_trees=8,
        leaf_size=None,
        rng_state=rng_state,
        random_state=random_state,
    )
    leaf_array = rptree_leaf_array(_rp_forest)
    pynndescent_.init_rp_tree(data, dist, current_graph, leaf_array)

    rng_state = new_rng_state()
    random_state = check_random_state(42)
    current_graph_threaded = utils.make_heap(data.shape[0], n_neighbors)
    _rp_forest = make_forest(
        data,
        n_neighbors,
        n_trees=8,
        leaf_size=None,
        rng_state=rng_state,
        random_state=random_state,
    )
    leaf_array = rptree_leaf_array(_rp_forest)
    parallel = joblib.Parallel(n_jobs=2, prefer="threads")
    threaded.init_rp_tree(
        data, dist, current_graph_threaded, leaf_array, chunk_size, parallel
    )

    assert_allclose(current_graph_threaded, current_graph)


def test_new_build_candidates():
    np.random.seed(42)
    N = 100
    D = 128
    chunk_size = N // 8
    n_neighbors = 25
    data = np.random.rand(N, D).astype(np.float32)
    n_vertices = data.shape[0]

    current_graph = utils.make_heap(data.shape[0], n_neighbors)
    pynndescent_.init_random(
        n_neighbors, data, current_graph, dist, new_rng_state(), seed_per_row=True,
    )
    new_candidate_neighbors, old_candidate_neighbors = utils.new_build_candidates(
        current_graph,
        n_vertices,
        n_neighbors,
        max_candidates,
        rng_state=new_rng_state(),
        seed_per_row=True,
    )

    current_graph = utils.make_heap(data.shape[0], n_neighbors)
    pynndescent_.init_random(
        n_neighbors, data, current_graph, dist, new_rng_state(), seed_per_row=True,
    )
    parallel = joblib.Parallel(n_jobs=2, prefer="threads")
    (
        new_candidate_neighbors_threaded,
        old_candidate_neighbors_threaded,
    ) = threaded.new_build_candidates(
        current_graph,
        n_vertices,
        n_neighbors,
        max_candidates,
        chunk_size=chunk_size,
        rng_state=new_rng_state(),
        parallel=parallel,
        seed_per_row=True,
    )

    assert_allclose(new_candidate_neighbors_threaded, new_candidate_neighbors)
    assert_allclose(old_candidate_neighbors_threaded, old_candidate_neighbors)


def test_mark_candidate_results():

    np.random.seed(42)
    N = 100
    D = 128
    chunk_size = N // 8
    n_neighbors = 25
    data = np.random.rand(N, D).astype(np.float32)
    n_vertices = data.shape[0]

    current_graph = utils.make_heap(data.shape[0], n_neighbors)
    pynndescent_.init_random(
        n_neighbors, data, current_graph, dist, new_rng_state(), seed_per_row=True,
    )
    pynndescent_.nn_descent_internal_low_memory_parallel(
        current_graph, data, n_neighbors, new_rng_state(), n_iters=2, seed_per_row=True
    )
    current_graph_threaded = (
        current_graph[0].copy(), current_graph[1].copy(), current_graph[2].copy(),
    )
    new_candidate_neighbors, old_candidate_neighbors = utils.new_build_candidates(
        current_graph,
        n_vertices,
        n_neighbors,
        max_candidates,
        rng_state=new_rng_state(),
        seed_per_row=True,
    )

    parallel = joblib.Parallel(n_jobs=2, prefer="threads")
    (
        new_candidate_neighbors_threaded,
        old_candidate_neighbors_threaded,
    ) = threaded.new_build_candidates(
        current_graph_threaded,
        n_vertices,
        n_neighbors,
        max_candidates,
        chunk_size=chunk_size,
        rng_state=new_rng_state(),
        parallel=parallel,
        seed_per_row=True,
    )

    assert_allclose(current_graph_threaded, current_graph)


def test_nn_descent():

    np.random.seed(42)
    N = 100
    # D = 128
    D = 4
    chunk_size = N // 8
    n_neighbors = 25
    data = np.random.rand(N, D).astype(np.float32)

    nn_indices, nn_distances = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=1,
        random_state=42,
        delta=0,
        tree_init=False,
        seed_per_row=True,
    )._neighbor_graph

    for i in range(data.shape[0]):
        assert_equal(
            len(nn_indices[i]),
            len(np.unique(nn_indices[i])),
            "Duplicate graph_indices in unthreaded knn graph",
        )

    nn_indices_threaded, nn_distances_threaded = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=1,
        random_state=42,
        delta=0,
        tree_init=False,
        seed_per_row=True,
        n_jobs=2,
    )._neighbor_graph

    for i in range(data.shape[0]):
        assert_equal(
            len(nn_indices_threaded[i]),
            len(np.unique(nn_indices_threaded[i])),
            "Duplicate graph_indices in threaded knn graph",
        )

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute").fit(data)
    _, nn_gold_indices = nbrs.kneighbors(data)

    assert_allclose(nn_indices_threaded, nn_indices)
    assert_allclose(nn_distances_threaded, nn_distances)


def test_nn_decent_with_parallel_backend():

    np.random.seed(42)
    N = 100
    D = 128
    chunk_size = N // 8
    n_neighbors = 25
    data = np.random.rand(N, D).astype(np.float32)

    nn_indices, nn_distances = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=1,
        random_state=42,
        delta=0,
        tree_init=False,
        seed_per_row=True,
    )._neighbor_graph

    with joblib.parallel_backend("threading"):
        nn_indices_threaded, nn_distances_threaded = NNDescent(
            data,
            n_neighbors=n_neighbors,
            max_candidates=max_candidates,
            n_iters=1,
            random_state=42,
            delta=0,
            tree_init=False,
            seed_per_row=True,
        )._neighbor_graph

    assert_allclose(nn_indices_threaded, nn_indices)
    assert_allclose(nn_distances_threaded, nn_distances)


@SkipTest
def test_nn_decent_with_n_jobs_minus_one():
    nn_indices, nn_distances = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=2,
        delta=0,
        tree_init=False,
        seed_per_row=True,
    )._neighbor_graph

    for i in range(data.shape[0]):
        assert_equal(
            len(nn_indices[i]),
            len(np.unique(nn_indices[i])),
            "Duplicate graph_indices in unthreaded knn graph",
        )

    nn_indices_threaded, nn_distances_threaded = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=2,
        delta=0,
        tree_init=False,
        seed_per_row=True,
        n_jobs=-1,
    )._neighbor_graph

    for i in range(data.shape[0]):
        assert_equal(
            len(nn_indices_threaded[i]),
            len(np.unique(nn_indices_threaded[i])),
            "Duplicate graph_indices in threaded knn graph",
        )

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute").fit(data)
    _, nn_gold_indices = nbrs.kneighbors(data)

    assert_allclose(nn_indices_threaded, nn_indices)
    assert_allclose(nn_distances_threaded, nn_distances)


def test_heap_updates():
    heap_updates = np.array(
        [
            [4, 1, 15, 0],
            [3, 3, 12, 0],
            [2, 2, 14, 0],
            [1, 5, 29, 0],
            [4, 7, 40, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    num_heap_updates = 5
    chunk_size = 2
    threaded.sort_heap_updates(heap_updates, num_heap_updates)

    sorted_heap_updates = np.array(
        [
            [1, 5, 29, 0],
            [2, 2, 14, 0],
            [3, 3, 12, 0],
            [4, 1, 15, 0],
            [4, 7, 40, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert_allclose(heap_updates, sorted_heap_updates)

    offsets = threaded.chunk_heap_updates(
        sorted_heap_updates, num_heap_updates, 6, chunk_size
    )

    assert_allclose(offsets, np.array([0, 1, 3, 5]))

    chunk0 = sorted_heap_updates[offsets[0] : offsets[1]]
    assert_allclose(chunk0, np.array([[1, 5, 29, 0]]))

    chunk1 = sorted_heap_updates[offsets[1] : offsets[2]]
    assert_allclose(chunk1, np.array([[2, 2, 14, 0], [3, 3, 12, 0]]))

    chunk2 = sorted_heap_updates[offsets[2] : offsets[3]]
    assert_allclose(chunk2, np.array([[4, 1, 15, 0], [4, 7, 40, 0]]))
