import numpy as np

from numpy.testing import assert_allclose

from sklearn.neighbors import NearestNeighbors

from pynndescent import distances
from pynndescent import pynndescent_
from pynndescent import NNDescent
from pynndescent import threaded
from pynndescent import utils

data = np.array(
    [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
)
chunk_size = 4
n_neighbors = 2
max_candidates = 8

dist = distances.named_distances["euclidean"]
dist_args = ()

# np.random.seed(42)
#
# N = 100000
# D = 128
# dataset = np.random.rand(N, D).astype(np.float32)
#
# chunk_size = 4000//8
# n_neighbors = 25
# max_candidates = 50
#
# data = dataset[:4000].astype(np.float32)

# In all tests, set seed_per_row=True so that we can comapre the regular
# algorithm against the threded algorithm.

def new_rng_state():
    return np.empty((3,), dtype=np.int64)


def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum() / expected.size


def test_init_current_graph():
    current_graph = pynndescent_.init_current_graph(data, dist, dist_args, n_neighbors, rng_state=new_rng_state(), seed_per_row=True)
    current_graph_threaded = threaded.init_current_graph(
        data, dist, dist_args, n_neighbors, chunk_size=chunk_size, rng_state=new_rng_state(),
        seed_per_row=True
    )

    assert_allclose(current_graph_threaded, current_graph)


def test_new_build_candidates():
    n_vertices = data.shape[0]

    current_graph = pynndescent_.init_current_graph(data, dist, dist_args, n_neighbors, rng_state=new_rng_state(), seed_per_row=True)
    new_candidate_neighbors, old_candidate_neighbors = utils.new_build_candidates(
        current_graph,
        n_vertices,
        n_neighbors,
        max_candidates,
        rng_state=new_rng_state(),
        seed_per_row=True
    )

    current_graph = pynndescent_.init_current_graph(data, dist, dist_args, n_neighbors, rng_state=new_rng_state(), seed_per_row=True)
    new_candidate_neighbors_threaded, old_candidate_neighbors_threaded = threaded.new_build_candidates(
        current_graph,
        n_vertices,
        n_neighbors,
        max_candidates,
        chunk_size=chunk_size,
        rng_state=new_rng_state(),
        seed_per_row=True
    )

    assert_allclose(new_candidate_neighbors_threaded, new_candidate_neighbors)
    assert_allclose(old_candidate_neighbors_threaded, old_candidate_neighbors)


def test_nn_descent():
    nn_indices, nn_distances = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=2,
        delta=0,
        tree_init=False,
        seed_per_row=True
    )._neighbor_graph

    nn_indices_threaded, nn_distances_threaded = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=2,
        delta=0,
        tree_init=False,
        seed_per_row=True,
        algorithm='threaded',
        chunk_size=chunk_size
    )._neighbor_graph

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(data)
    _, nn_gold_indices = nbrs.kneighbors(data)

    print("regular accuracy", accuracy(nn_gold_indices, nn_indices))
    print("threaded accuracy", accuracy(nn_gold_indices, nn_indices_threaded))

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
        dtype=np.float64,
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
        dtype=np.float64,
    )
    assert_allclose(heap_updates, sorted_heap_updates)

    offsets = threaded.chunk_heap_updates(sorted_heap_updates, num_heap_updates, 6, chunk_size)

    assert_allclose(offsets, np.array([0, 1, 3, 5]))

    chunk0 = sorted_heap_updates[offsets[0] : offsets[1]]
    assert_allclose(chunk0, np.array([[1, 5, 29, 0]]))

    chunk1 = sorted_heap_updates[offsets[1] : offsets[2]]
    assert_allclose(chunk1, np.array([[2, 2, 14, 0], [3, 3, 12, 0]]))

    chunk2 = sorted_heap_updates[offsets[2] : offsets[3]]
    assert_allclose(chunk2, np.array([[4, 1, 15, 0], [4, 7, 40, 0]]))
