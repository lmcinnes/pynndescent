"""Tests of the public NNDescent and PyNNDescentTransformer API: constructor
options, neighbor_graph, query, prepare, update, pickling, error paths and
recall against brute force."""
import pickle
import warnings

import numba
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

import pynndescent.pynndescent_ as pynnd_module
from pynndescent import NNDescent, PyNNDescentTransformer


def _recall(found, truth):
	return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])


def _brute(queries, data, k, metric="euclidean"):
	return np.argsort(cdist(queries, data, metric), axis=1, kind="stable")[:, :k]


def _graph_recall(index, data, metric="euclidean"):
	indices, _ = index.neighbor_graph
	return _recall(indices, _brute(data, data, indices.shape[1], metric))


@pytest.fixture(scope="module")
def data():
	return np.random.RandomState(0).normal(size=(300, 8)).astype(np.float32)


@pytest.fixture(scope="module")
def queries():
	return np.random.RandomState(1).normal(size=(60, 8)).astype(np.float32)


@pytest.fixture(scope="module")
def index(data):
	return NNDescent(data, n_neighbors=15, random_state=0, n_jobs=1)


def _build(data, **kwargs):
	kwargs.setdefault("n_neighbors", 15)
	kwargs.setdefault("random_state", 0)
	kwargs.setdefault("n_jobs", 1)
	return NNDescent(data, **kwargs)


# ------------------------------------------------------------ neighbor_graph


def test_neighbor_graph_structure(index, data):
	indices, distances = index.neighbor_graph
	n = data.shape[0]
	assert indices.shape == distances.shape == (n, 15)
	assert indices.dtype == np.int32 and distances.dtype == np.float32
	np.testing.assert_array_equal(indices[:, 0], np.arange(n))
	assert (distances[:, 0] == 0).all()
	assert indices.min() >= 0 and indices.max() < n
	assert (np.diff(distances, axis=1) >= 0).all()
	assert all(len(set(row)) == 15 for row in indices)
	true = cdist(data, data)
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, indices, axis=1), rtol=1e-4, atol=1e-5
	)


def test_neighbor_graph_recall_against_sklearn(index, data):
	truth = NearestNeighbors(n_neighbors=15).fit(data).kneighbors(data)[1]
	found, _ = index.neighbor_graph
	assert _recall(found, truth) >= 0.95


def test_neighbor_graph_returns_independent_copies(index):
	first = index.neighbor_graph
	first[0][:] = -5
	first[1][:] = -5
	second = index.neighbor_graph
	assert (second[0] >= 0).all()
	assert (second[1] >= 0).all()


def test_neighbor_graph_exact_when_k_covers_all_points():
	small = np.random.RandomState(3).normal(size=(12, 3)).astype(np.float32)
	index = _build(small, n_neighbors=12)
	indices, distances = index.neighbor_graph
	truth = cdist(small, small)
	for i in range(12):
		assert sorted(indices[i]) == list(range(12))
	np.testing.assert_allclose(distances, np.sort(truth, axis=1), rtol=1e-4, atol=1e-5)


def test_too_few_points_warns_and_pads(data):
	with pytest.warns(UserWarning):
		index = _build(data[:5], n_neighbors=10)
	indices, distances = index.neighbor_graph
	assert indices.shape == (5, 10)
	assert (indices[:, :5] >= 0).all()
	assert (indices[:, 5:] == -1).all()
	assert np.isinf(distances[:, 5:]).all()


# ----------------------------------------------------------- constructor: metric


@pytest.mark.parametrize(
	"metric, scipy_metric",
	[("euclidean", "euclidean"), ("cosine", "cosine"), ("manhattan", "cityblock")],
)
def test_metric_graph_and_query_match_scipy(data, queries, metric, scipy_metric):
	index = _build(data, metric=metric)
	indices, distances = index.neighbor_graph
	true = cdist(data, data, scipy_metric)
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, indices, axis=1), rtol=1e-3, atol=2e-5
	)
	assert _graph_recall(index, data, scipy_metric) >= 0.9
	found, found_d = index.query(queries, k=5, epsilon=0.3)
	true_q = cdist(queries, data, scipy_metric)
	assert _recall(found, _brute(queries, data, 5, scipy_metric)) >= 0.9
	np.testing.assert_allclose(
		found_d, np.take_along_axis(true_q, found, axis=1), rtol=1e-3, atol=2e-5
	)


def test_metric_changes_neighbours(data, queries):
	"""On 8-d Gaussian data many queries have different exact 3-neighbour sets
	under L1 and L2 (checked by brute force); each index recovers the sets of
	its own metric (recall >= 0.9 on those queries), so the metric choice changes the neighbours."""
	true_l2 = _brute(queries, data, 3, "euclidean")
	true_l1 = _brute(queries, data, 3, "cityblock")
	differ = np.array([set(a) != set(b) for a, b in zip(true_l2, true_l1)])
	assert differ.sum() >= 10
	euclid = _build(data).query(queries, k=3, epsilon=0.3)[0]
	manhattan = _build(data, metric="manhattan").query(queries, k=3, epsilon=0.3)[0]
	assert _recall(euclid[differ], true_l2[differ]) >= 0.9
	assert _recall(manhattan[differ], true_l1[differ]) >= 0.9
	assert np.mean([set(a) != set(b) for a, b in zip(euclid[differ], manhattan[differ])]) >= 0.8


def test_metric_kwds_minkowski_p(data):
	index = _build(data, metric="minkowski", metric_kwds={"p": 3})
	indices, distances = index.neighbor_graph
	true = cdist(data, data, "minkowski", p=3)
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, indices, axis=1), rtol=1e-3, atol=2e-5
	)
	# p=1 gives a different graph from p=3
	other = _build(data, metric="minkowski", metric_kwds={"p": 1})
	assert not np.allclose(other.neighbor_graph[1], distances)


def test_callable_metric(data, queries):
	@numba.njit(fastmath=True)
	def l1(x, y):
		result = 0.0
		for i in range(x.shape[0]):
			result += abs(x[i] - y[i])
		return result

	index = _build(data, metric=l1)
	indices, distances = index.neighbor_graph
	true = cdist(data, data, "cityblock")
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, indices, axis=1), rtol=1e-3, atol=2e-5
	)
	found, _ = index.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5, "cityblock")) >= 0.9


def test_unknown_metric_raises(data):
	with pytest.raises(ValueError):
		NNDescent(data, metric="not-a-metric")


# ----------------------------------------------------- constructor: graph options


def test_n_neighbors_sets_graph_width(data):
	for k in (5, 20):
		index = _build(data, n_neighbors=k)
		indices, distances = index.neighbor_graph
		assert indices.shape == (300, k) and distances.shape == (300, k)
		assert index.n_neighbors == k
	assert _graph_recall(_build(data, n_neighbors=5), data) >= 0.95


def test_default_n_trees_and_n_iters_follow_data_size(data):
	"""For 300 points the defaults are 5 trees, 8 iterations and 2 trees after an
	update (formulas copied from the constructor: max(3, min(12, round(2 log10 n))),
	max(5, round(log2 n)), max(2, round(n_trees / 3))); pins the current output."""
	index = _build(data)
	assert index.n_trees == 5
	assert index.n_iters == 8
	assert index.n_trees_after_update == 2
	explicit = _build(data, n_trees=7, n_iters=3)
	assert explicit.n_trees == 7 and explicit.n_iters == 3


def test_n_trees_builds_that_many_trees(data):
	index = _build(data, n_trees=4, leaf_size=50)
	leaves = index._rp_forest
	members = leaves[leaves >= 0]
	np.testing.assert_array_equal(np.bincount(members, minlength=300), 4)
	assert _graph_recall(index, data) >= 0.9


def test_n_trees_zero_disables_tree_init(data):
	index = _build(data, n_trees=0)
	assert index.tree_init is False
	assert index._rp_forest is None
	assert _graph_recall(index, data) >= 0.9


def test_leaf_size_bounds_leaves_and_changes_initial_graph(data):
	small = _build(data, n_trees=4, leaf_size=20, n_iters=0)
	large = _build(data, n_trees=4, leaf_size=100, n_iters=0)
	assert small._rp_forest.shape[1] <= 20
	assert large._rp_forest.shape[1] <= 100
	assert large._rp_forest.shape[1] > small._rp_forest.shape[1]
	assert _graph_recall(large, data) > _graph_recall(small, data)


def test_tree_init_false_starts_from_random_graph(data):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		with_trees = _build(data, n_iters=0)
		without = _build(data, tree_init=False, n_iters=0)
	assert without.tree_init is False and without._rp_forest is None
	assert with_trees.tree_init is True
	assert _graph_recall(with_trees, data) > _graph_recall(without, data) + 0.3


def test_n_iters_zero_leaves_initial_graph(data):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		none = _build(data, n_trees=0, n_iters=0)
	assert _graph_recall(none, data) < 0.5
	assert _graph_recall(_build(data, n_trees=0), data) >= 0.9


def test_delta_stops_descent_early(data):
	early = _build(data, n_trees=0, delta=1e9)
	full = _build(data, n_trees=0, delta=0.0)
	assert _graph_recall(early, data) + 0.3 < _graph_recall(full, data)
	assert full.delta == 0.0 and early.delta == 1e9


def test_max_candidates_limits_per_iteration_work(data):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		few = _build(data, n_trees=0, max_candidates=1, n_iters=1, delta=0.0)
	many = _build(data, n_trees=0, max_candidates=30, n_iters=1, delta=0.0)
	assert _graph_recall(many, data) > _graph_recall(few, data) + 0.3
	assert few.max_candidates == 1


@pytest.mark.parametrize("low_memory", [True, False])
def test_low_memory_both_settings_build_good_graphs(data, low_memory):
	index = _build(data, low_memory=low_memory)
	assert index.low_memory is low_memory
	assert _graph_recall(index, data) >= 0.95


def test_init_graph_with_exact_neighbours_is_preserved(data):
	truth = NearestNeighbors(n_neighbors=16).fit(data).kneighbors(data)[1][:, 1:]
	index = _build(data, init_graph=truth.astype(np.int32), n_iters=0)
	assert index.tree_init is False
	indices, _ = index.neighbor_graph
	np.testing.assert_array_equal(indices[:, 1:], truth[:, :14])
	assert _graph_recall(index, data) >= 0.999


def test_init_graph_with_init_dist(data):
	true = cdist(data, data, "cityblock")
	truth = np.argsort(true, axis=1)[:, 1:16].astype(np.int32)
	init_dist = np.take_along_axis(true, truth, axis=1).astype(np.float32)
	index = _build(data, metric="manhattan", init_graph=truth, init_dist=init_dist, n_iters=0)
	assert _graph_recall(index, data, "cityblock") >= 0.999
	np.testing.assert_allclose(
		index.neighbor_graph[1][:, 1:], init_dist[:, :14], rtol=1e-5
	)
	# distances the caller supplied are used as given
	scaled = _build(
		data, metric="manhattan", init_graph=truth, init_dist=init_dist * 2, n_iters=0
	)
	np.testing.assert_allclose(
		scaled.neighbor_graph[1][:, 1:], 2 * init_dist[:, :14], rtol=1e-5
	)


def test_init_graph_with_unknown_entries_is_completed(data):
	truth = NearestNeighbors(n_neighbors=16).fit(data).kneighbors(data)[1][:, 1:]
	partial = truth.astype(np.int32)
	partial[:, 7:] = -1
	index = _build(data, init_graph=partial)
	assert _graph_recall(index, data) >= 0.95


def test_init_graph_errors(data):
	with pytest.raises(ValueError):
		NNDescent(data, n_neighbors=5, init_graph=np.zeros((10, 5), dtype=np.int32))
	with pytest.raises(ValueError):
		NNDescent(
			data,
			n_neighbors=5,
			init_graph=np.zeros((300, 5), dtype=np.int32),
			init_dist=np.zeros((300, 4), dtype=np.float32),
		)
	sparse_data = sp.random(40, 10, density=0.4, format="csr", random_state=0)
	with pytest.raises(ValueError):
		NNDescent(sparse_data, n_neighbors=5, init_graph=np.zeros((9, 5), dtype=np.int32))


def test_random_state_makes_builds_reproducible(data):
	a = _build(data, random_state=7)
	b = _build(data, random_state=7)
	c = _build(data, random_state=8)
	np.testing.assert_array_equal(a.neighbor_graph[0], b.neighbor_graph[0])
	np.testing.assert_array_equal(a.rng_state, b.rng_state)
	assert not np.array_equal(a.rng_state, c.rng_state)
	state = np.random.RandomState(7)
	d = _build(data, random_state=state)
	assert d.rng_state.shape == (3,)


def test_n_jobs_is_restored_and_builds_agree(data):
	before = numba.get_num_threads()
	one = _build(data, n_jobs=1)
	assert numba.get_num_threads() == before
	if numba.config.NUMBA_NUM_THREADS >= 2:
		two = _build(data, n_jobs=2)
		assert numba.get_num_threads() == before
		assert _graph_recall(two, data) >= 0.95
	for value in (None, -1):
		other = _build(data, n_jobs=value)
		assert numba.get_num_threads() == before
		assert _graph_recall(other, data) >= 0.95
	assert _graph_recall(one, data) >= 0.95


def test_verbose_prints_progress_only_when_asked(data, capsys):
	_build(data, verbose=False)
	assert capsys.readouterr().out == ""
	_build(data, verbose=True)
	assert capsys.readouterr().out != ""


def test_search_tree_leaf_budget_validation_and_use(data, queries):
	with pytest.raises(ValueError):
		NNDescent(data, search_tree_leaf_budget=0)
	index = _build(data, search_tree_leaf_budget=3)
	assert index.search_tree_leaf_budget == 3
	found, _ = index.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5)) >= 0.9


def test_search_tree_leaf_size_bounds_search_tree_leaves(data):
	index = _build(data, search_tree_leaf_size=10)
	index.prepare()
	assert index._search_forest[0].leaf_size <= 10


# ------------------------------------------------------- search graph options


def _search_graph_edges(data, **kwargs):
	index = _build(data, **kwargs)
	index.prepare()
	return index._search_graph


def test_search_graph_structure(data):
	graph = _search_graph_edges(data)
	assert graph.shape == (300, 300)
	assert graph.dtype == np.uint8
	assert (graph.data == 1).all()
	assert graph.diagonal().sum() == 0
	assert graph.has_sorted_indices
	for i in range(300):
		row = graph.indices[graph.indptr[i] : graph.indptr[i + 1]]
		assert (np.diff(row) > 0).all()


def test_prepare_builds_permutation_and_is_idempotent(data, queries):
	index = _build(data)
	assert not hasattr(index, "_search_graph")
	index.prepare()
	assert hasattr(index, "_search_graph") and hasattr(index, "_search_function")
	np.testing.assert_array_equal(np.sort(index._vertex_order), np.arange(300))
	graph = index._search_graph
	first = index.query(queries, k=5, epsilon=0.3)
	index.prepare()
	assert index._search_graph is graph
	second = index.query(queries, k=5, epsilon=0.3)
	assert _recall(first[0], second[0]) >= 0.9
	# neighbor_graph still refers to the original point order
	np.testing.assert_array_equal(index.neighbor_graph[0][:, 0], np.arange(300))


def test_pruning_degree_multiplier_bounds_search_graph_degree(data):
	tight = _search_graph_edges(data, pruning_degree_multiplier=0.5)
	loose = _search_graph_edges(data, pruning_degree_multiplier=3.0)
	max_degree = int(np.round(0.5 * 15))
	assert np.diff(tight.indptr).max() <= max_degree + 1
	assert tight.nnz < loose.nnz
	assert np.diff(loose.indptr).max() > max_degree + 1


def test_diversify_prob_controls_edge_removal(data):
	keep_all = _search_graph_edges(data, diversify_prob=0.0)
	prune_all = _search_graph_edges(data, diversify_prob=1.0)
	half = _search_graph_edges(data, diversify_prob=0.5)
	assert prune_all.nnz < half.nnz < keep_all.nnz


def test_degree_aware_diversify_matches_standard_at_zero_aggressiveness(data):
	standard = _search_graph_edges(data)
	zero = _search_graph_edges(
		data, diversify_method="degree_aware", degree_prune_aggressiveness=0.0
	)
	np.testing.assert_array_equal(standard.indptr, zero.indptr)
	np.testing.assert_array_equal(standard.indices, zero.indices)
	aggressive = _search_graph_edges(
		data, diversify_method="degree_aware", degree_prune_aggressiveness=3.0,
		pruning_degree_multiplier=0.5,
	)
	assert aggressive.nnz <= standard.nnz


def test_degree_aware_index_still_finds_neighbours(data, queries):
	index = _build(data, diversify_method="degree_aware")
	found, _ = index.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5)) >= 0.9


def test_compressed_option_drops_construction_data(data, queries):
	index = _build(data, compressed=True)
	assert index.neighbor_graph[0].shape == (300, 15)
	index.prepare()
	assert not hasattr(index, "_neighbor_graph")
	with pytest.warns(UserWarning):
		assert index.neighbor_graph is None
	found, _ = index.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5)) >= 0.9


def test_compress_index_matches_uncompressed_results(data, queries):
	plain = _build(data)
	squeezed = _build(data)
	before = plain.query(queries, k=5, epsilon=0.3)
	squeezed.compress_index()
	assert squeezed.compressed is True
	assert not hasattr(squeezed, "_neighbor_graph")
	assert not hasattr(squeezed, "_rp_forest")
	after = squeezed.query(queries, k=5, epsilon=0.3)
	np.testing.assert_array_equal(before[0], after[0])
	np.testing.assert_allclose(before[1], after[1])
	with pytest.warns(UserWarning):
		assert squeezed.neighbor_graph is None


# ---------------------------------------------------------------------- query


def test_query_output_structure_and_distances(index, data, queries):
	for k in (1, 5, 15):
		indices, distances = index.query(queries, k=k)
		assert indices.shape == distances.shape == (60, k)
		assert indices.dtype == np.int32 and distances.dtype == np.float32
		assert indices.min() >= 0 and indices.max() < 300
		assert (np.diff(distances, axis=1) >= 0).all()
		true = cdist(queries, data)
		np.testing.assert_allclose(
			distances, np.take_along_axis(true, indices, axis=1), rtol=1e-4, atol=1e-5
		)
		assert all(len(set(row)) == k for row in indices)


def test_query_default_k_is_ten(index, queries):
	assert index.query(queries)[0].shape == (60, 10)


def test_query_recall_against_brute_force(index, data, queries):
	found, _ = index.query(queries, k=10, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 10)) >= 0.95


def test_query_of_training_points_returns_themselves(index, data):
	found, distances = index.query(data[:50], k=3)
	np.testing.assert_array_equal(found[:, 0], np.arange(50))
	np.testing.assert_allclose(distances[:, 0], 0, atol=1e-3)


def test_query_epsilon_trades_accuracy_for_search_effort(data):
	rng = np.random.RandomState(5)
	many = rng.normal(size=(200, 8)).astype(np.float32)
	index = _build(data, n_trees=4)
	truth = _brute(many, data, 10)
	tight = _recall(index.query(many, k=10, epsilon=0.0)[0], truth)
	wide = _recall(index.query(many, k=10, epsilon=0.3)[0], truth)
	assert wide > tight
	assert wide >= 0.95


def test_query_accepts_lists_float64_and_fortran_order(index, queries):
	reference = index.query(queries[:5], k=4)[0]
	np.testing.assert_array_equal(index.query(queries[:5].tolist(), k=4)[0], reference)
	np.testing.assert_array_equal(
		index.query(queries[:5].astype(np.float64), k=4)[0], reference
	)
	np.testing.assert_array_equal(
		index.query(np.asfortranarray(queries[:5]), k=4)[0], reference
	)


def test_query_results_survive_later_queries(index, queries):
	first = index.query(queries[:5], k=4)
	saved = (first[0].copy(), first[1].copy())
	index.query(queries[10:15], k=4)
	np.testing.assert_array_equal(first[0], saved[0])
	np.testing.assert_array_equal(first[1], saved[1])


def test_query_indices_and_return_distances_false(index, queries):
	with_distances = index.query(queries[:8], k=4)[0]
	only = index.query(queries[:8], k=4, return_distances=False)
	assert isinstance(only, np.ndarray) and only.shape == (8, 4)
	np.testing.assert_array_equal(only, with_distances)
	np.testing.assert_array_equal(index.query_indices(queries[:8], k=4), with_distances)


def test_query_error_paths(index):
	with pytest.raises(ValueError):
		index.query(np.zeros((3, 5), dtype=np.float32))
	with pytest.raises(ValueError):
		index.query(np.zeros(8, dtype=np.float32))
	with pytest.raises(ValueError):
		index.query(np.zeros((2, 3, 8), dtype=np.float32))
	with pytest.raises(ValueError):
		index.query(np.zeros((2, 8), dtype=np.float32), k=-1)
	with pytest.raises(ValueError):
		index.query(np.zeros((2, 8), dtype=np.float32), pre_normalized=True)


def test_parallel_batch_queries_agree_with_serial(data, queries):
	parallel = _build(data, parallel_batch_queries=True)
	assert parallel.parallel_batch_queries is True
	found, distances = parallel.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5)) >= 0.95
	true = cdist(queries, data)
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, found, axis=1), rtol=1e-4, atol=1e-5
	)


def test_cosine_pre_normalized_matches_plain_query(data, queries):
	index = _build(data, metric="cosine")
	normalized = queries / np.linalg.norm(queries, axis=1, keepdims=True)
	plain = index.query(queries, k=5, epsilon=0.3)
	fast = index.query(normalized.astype(np.float32), k=5, epsilon=0.3, pre_normalized=True)
	assert _recall(plain[0], fast[0]) >= 0.95
	np.testing.assert_allclose(plain[1][:, 0], fast[1][:, 0], atol=1e-5)
	true = cdist(queries, data, "cosine")
	np.testing.assert_allclose(
		plain[1], np.take_along_axis(true, plain[0], axis=1), rtol=1e-3, atol=2e-5
	)


# ------------------------------------------------------ sparse data and queries


@pytest.fixture(scope="module")
def sparse_pair():
	train = sp.random(300, 30, density=0.3, format="csr", random_state=0, dtype=np.float32)
	test = sp.random(30, 30, density=0.3, format="csr", random_state=1, dtype=np.float32)
	return train, test


def test_sparse_index_query_matches_brute_force(sparse_pair):
	train, test = sparse_pair
	index = _build(train)
	assert index._is_sparse
	found, distances = index.query(test, k=5, epsilon=0.3)
	true = cdist(test.toarray(), train.toarray())
	assert found.shape == (30, 5)
	assert _recall(found, np.argsort(true, axis=1)[:, :5]) >= 0.9
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, found, axis=1), rtol=1e-3, atol=1e-4
	)
	indices, graph_dists = index.neighbor_graph
	np.testing.assert_array_equal(indices[:, 0], np.arange(300))
	assert _graph_recall(index, train.toarray()) >= 0.9


def test_sparse_query_input_types_agree(sparse_pair):
	train, test = sparse_pair
	index = _build(train)
	reference = index.query(test, k=5, epsilon=0.3)[0]
	np.testing.assert_array_equal(
		index.query(sp.csr_array(test), k=5, epsilon=0.3)[0], reference
	)
	np.testing.assert_array_equal(
		index.query(test.toarray(), k=5, epsilon=0.3)[0], reference
	)
	np.testing.assert_array_equal(
		index.query(test.tocsc(), k=5, epsilon=0.3)[0], reference
	)
	unsorted = test.copy()
	row = slice(unsorted.indptr[0], unsorted.indptr[1])
	unsorted.indices[row] = unsorted.indices[row][::-1].copy()
	unsorted.data[row] = unsorted.data[row][::-1].copy()
	unsorted.has_sorted_indices = False
	np.testing.assert_array_equal(index.query(unsorted, k=5, epsilon=0.3)[0], reference)


def test_sparse_pre_normalized_is_rejected(sparse_pair):
	train, test = sparse_pair
	index = _build(train)
	with pytest.raises(ValueError):
		index.query(test, pre_normalized=True)


def test_sparse_update_is_not_implemented(sparse_pair):
	train, _ = sparse_pair
	index = _build(train)
	with pytest.raises(NotImplementedError):
		index.update(xs_fresh=train[:3].toarray())


# ------------------------------------------------------------- quantization


@pytest.fixture(scope="module")
def uint8_index(data):
	return _build(data, quantization="uint8")


def test_uint8_quantization_reranks_with_exact_distances(uint8_index, data, queries):
	found, distances = uint8_index.query(queries, k=5, epsilon=0.3)
	true = cdist(queries, data)
	assert _recall(found, np.argsort(true, axis=1)[:, :5]) >= 0.95
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, found, axis=1), rtol=1e-4, atol=1e-5
	)


def test_quantized_query_width_options(uint8_index, data, queries):
	truth = _brute(queries, data, 5)
	proxy_only = uint8_index.query(queries, k=5, epsilon=0.3, rerank_width=0)
	assert proxy_only[0].shape == (60, 5)
	assert _recall(proxy_only[0], truth) >= 0.8
	wide = uint8_index.query(queries, k=5, epsilon=0.3, candidate_width=30)
	assert _recall(wide[0], truth) >= 0.95
	beam = uint8_index.query(queries, k=5, epsilon=0.3, proxy_beam_size=3)
	assert beam[0].shape == (60, 5)
	assert _recall(beam[0], truth) >= 0.95
	with pytest.raises(ValueError):
		uint8_index.query(queries, k=5, candidate_width=3)
	with pytest.raises(ValueError):
		uint8_index.query(queries, k=5, candidate_width=10, rerank_width=11)
	with pytest.raises(ValueError):
		uint8_index.query(queries, k=5, candidate_width=10, rerank_width=2)


def test_quantization_errors(data):
	with pytest.raises(ValueError):
		_build(data, quantization="not-a-scheme").prepare()
	with pytest.raises(ValueError):
		_build(data, metric="manhattan", quantization="binary").prepare()


# ----------------------------------------------------------------------- update


def test_update_with_fresh_points(data, queries):
	"""Appending 20 points keeps the graph and queries accurate against brute force; pins the current output for the tree count after update."""
	index = _build(data)
	index.prepare()
	fresh = np.random.RandomState(31).normal(size=(20, 8)).astype(np.float32)
	index.update(xs_fresh=fresh)
	everything = np.vstack([data, fresh])
	assert index._raw_data.shape == (320, 8)
	assert index.n_trees == index.n_trees_after_update
	indices, distances = index.neighbor_graph
	assert indices.shape == (320, 15)
	np.testing.assert_array_equal(indices[:, 0], np.arange(320))
	assert _recall(indices, _brute(everything, everything, 15)) >= 0.95
	found, found_d = index.query(queries, k=5, epsilon=0.3)
	assert found.max() < 320
	assert _recall(found, _brute(queries, everything, 5)) >= 0.95
	own, own_d = index.query(fresh, k=1)
	np.testing.assert_array_equal(own[:, 0], np.arange(300, 320))
	np.testing.assert_allclose(own_d, 0, atol=1e-3)


def test_update_with_replaced_points(data):
	index = _build(data)
	index.prepare()
	replacement = np.random.RandomState(32).normal(size=(3, 8)).astype(np.float32) + 5
	index.update(xs_updated=replacement, updated_indices=[0, 5, 10])
	changed = data.copy()
	changed[[0, 5, 10]] = replacement
	assert index._raw_data.shape == (300, 8)
	found, found_d = index.query(replacement, k=1)
	np.testing.assert_array_equal(found[:, 0], [0, 5, 10])
	np.testing.assert_allclose(found_d, 0, atol=1e-3)
	indices, _ = index.neighbor_graph
	assert _recall(indices, _brute(changed, changed, 15)) >= 0.95


def test_update_with_fresh_and_replaced_points_together(data):
	index = _build(data)
	rng = np.random.RandomState(33)
	fresh = rng.normal(size=(5, 8)).astype(np.float32)
	replacement = rng.normal(size=(2, 8)).astype(np.float32) + 4
	index.update(xs_fresh=fresh, xs_updated=replacement, updated_indices=np.array([3, 4]))
	changed = np.vstack([data, fresh])
	changed[[3, 4]] = replacement
	assert index.neighbor_graph[0].shape == (305, 15)
	found, _ = index.query(changed[:20], k=1)
	np.testing.assert_array_equal(found[:, 0], np.arange(20))


def test_update_without_arguments_keeps_size(data):
	index = _build(data)
	index.update()
	assert index.neighbor_graph[0].shape == (300, 15)
	assert _graph_recall(index, data) >= 0.95


def test_update_updated_indices_without_data_warns(data):
	index = _build(data)
	with pytest.warns(UserWarning):
		index.update(updated_indices=[0])
	assert index.neighbor_graph[0].shape == (300, 15)


def test_update_error_paths(data):
	rows = np.zeros((3, 8), dtype=np.float32)
	index = _build(data)
	with pytest.raises(ValueError):
		index.update(xs_updated=rows)
	with pytest.raises(ValueError):
		index.update(xs_updated=rows, updated_indices=[0, 1])
	with pytest.raises(ValueError):
		index.update(xs_updated=rows, updated_indices=["a", "b", "c"])
	with pytest.raises(ValueError):
		index.update(xs_updated=np.zeros((3, 2), dtype=np.float32), updated_indices=[0, 1, 2])
	with pytest.raises(ValueError):
		index.update(xs_fresh=np.zeros((2, 3), dtype=np.float32))


# --------------------------------------------------------------------- pickling


def test_pickle_round_trip_preserves_graph_and_queries(data, queries):
	index = _build(data)
	payload = pickle.dumps(index)
	before = index.query(queries, k=5, epsilon=0.3)
	clone_index = pickle.loads(payload)
	np.testing.assert_array_equal(
		clone_index.neighbor_graph[0], index.neighbor_graph[0]
	)
	after = clone_index.query(queries, k=5, epsilon=0.3)
	assert _recall(after[0], before[0]) >= 0.95
	true = cdist(queries, data)
	np.testing.assert_allclose(
		after[1], np.take_along_axis(true, after[0], axis=1), rtol=1e-4, atol=1e-5
	)


def test_compressed_pickle_is_smaller_and_queries_work(data, queries):
	index = _build(data)
	full = pickle.dumps(index)
	index.compress_index()
	small = pickle.dumps(index)
	assert len(small) < len(full)
	restored = pickle.loads(small)
	assert restored.compressed is True
	found, _ = restored.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5)) >= 0.95


def test_unpickled_index_can_be_pickled_again(data, queries):
	restored = pickle.loads(pickle.dumps(_build(data)))
	again = pickle.loads(pickle.dumps(restored))
	found, _ = again.query(queries, k=5, epsilon=0.3)
	assert _recall(found, _brute(queries, data, 5)) >= 0.95


# ---------------------------------------------------------------- data validation


def test_nan_data_is_rejected():
	bad = np.full((20, 3), np.nan, dtype=np.float32)
	with pytest.raises(ValueError):
		NNDescent(bad)


def test_input_is_converted_to_float32(data):
	index = _build(data.astype(np.float64))
	assert index._raw_data.dtype == np.float32
	index = _build(np.asfortranarray(data))
	assert index._raw_data.flags["C_CONTIGUOUS"]
	assert index.dim == 8


# ------------------------------------------------------------------ transformer


@pytest.fixture
def captured_nndescent(monkeypatch):
	calls = []
	real = pynnd_module.NNDescent

	def spy(*args, **kwargs):
		calls.append(kwargs)
		return real(*args, **kwargs)

	monkeypatch.setattr(pynnd_module, "NNDescent", spy)
	return calls


def test_transformer_fit_transform_graph(data):
	transformer = PyNNDescentTransformer(n_neighbors=7, random_state=0, n_jobs=1)
	graph = transformer.fit_transform(data)
	assert graph.shape == (300, 300)
	assert graph.dtype == np.float32
	assert (np.diff(graph.indptr) == 8).all()
	np.testing.assert_array_equal(graph.diagonal(), 0)
	true = cdist(data, data)
	rows = [graph.indices[graph.indptr[i] : graph.indptr[i + 1]] for i in range(300)]
	assert _recall(rows, _brute(data, data, 8)) >= 0.95
	for i in range(0, 300, 37):
		np.testing.assert_allclose(
			graph.data[graph.indptr[i] : graph.indptr[i + 1]],
			true[i, rows[i]],
			rtol=1e-4,
			atol=1e-5,
		)
	assert transformer.index_.n_neighbors == 8
	assert transformer.n_samples_fit == 300
	assert transformer.index_.compressed is True


def test_transformer_transform_new_points(data, queries):
	transformer = PyNNDescentTransformer(n_neighbors=7, random_state=0, n_jobs=1).fit(data)
	graph = transformer.transform(queries)
	assert graph.shape == (60, 300)
	assert (np.diff(graph.indptr) == 7).all()
	rows = [graph.indices[graph.indptr[i] : graph.indptr[i + 1]] for i in range(60)]
	assert _recall(rows, _brute(queries, data, 7)) >= 0.9
	true = cdist(queries, data)
	for i in range(60):
		np.testing.assert_allclose(
			graph.data[graph.indptr[i] : graph.indptr[i + 1]], true[i, rows[i]],
			rtol=1e-4, atol=1e-5,
		)


def test_transformer_fit_compress_index_flag(data):
	kept = PyNNDescentTransformer(n_neighbors=5, random_state=0, n_jobs=1).fit(
		data, compress_index=False
	)
	squeezed = PyNNDescentTransformer(n_neighbors=5, random_state=0, n_jobs=1).fit(data)
	assert kept.index_.compressed is False
	assert squeezed.index_.compressed is True


def test_transformer_passes_options_to_index(data, captured_nndescent):
	"""Each transformer option reaches the NNDescent constructor (n_neighbors + 1, delta from early_termination_value); pins the current output."""
	transformer = PyNNDescentTransformer(
		n_neighbors=6,
		metric="manhattan",
		metric_kwds={},
		n_trees=4,
		leaf_size=40,
		pruning_degree_multiplier=2.0,
		diversify_prob=0.5,
		tree_init=True,
		random_state=3,
		n_jobs=1,
		low_memory=False,
		max_candidates=12,
		n_iters=6,
		early_termination_value=0.01,
		parallel_batch_queries=True,
		verbose=False,
	)
	transformer.fit(data)
	kwargs = captured_nndescent[-1]
	assert kwargs["n_neighbors"] == 7
	assert kwargs["metric"] == "manhattan"
	assert kwargs["n_trees"] == 4
	assert kwargs["leaf_size"] == 40
	assert kwargs["pruning_degree_multiplier"] == 2.0
	assert kwargs["diversify_prob"] == 0.5
	assert kwargs["random_state"] == 3
	assert kwargs["n_jobs"] == 1
	assert kwargs["low_memory"] is False
	assert kwargs["max_candidates"] == 12
	assert kwargs["n_iters"] == 6
	assert kwargs["delta"] == 0.01
	assert kwargs["parallel_batch_queries"] is True
	index = transformer.index_
	assert index.n_trees == 4 and index.n_iters == 6 and index.delta == 0.01


def test_transformer_search_epsilon_reaches_query(data, queries, monkeypatch):
	transformer = PyNNDescentTransformer(
		n_neighbors=5, search_epsilon=0.37, random_state=0, n_jobs=1
	).fit(data)
	seen = {}
	real_query = transformer.index_.query

	def spy(X, k=10, epsilon=0.1, **kwargs):
		seen["k"], seen["epsilon"] = k, epsilon
		return real_query(X, k=k, epsilon=epsilon, **kwargs)

	monkeypatch.setattr(transformer.index_, "query", spy)
	transformer.transform(queries)
	assert seen == {"k": 5, "epsilon": 0.37}


def test_transformer_is_a_cloneable_estimator():
	transformer = PyNNDescentTransformer(n_neighbors=9, search_epsilon=0.2, random_state=1)
	params = transformer.get_params()
	assert params["n_neighbors"] == 9 and params["search_epsilon"] == 0.2
	assert clone(transformer).get_params() == params
	transformer.set_params(n_neighbors=11)
	assert transformer.n_neighbors == 11


def test_transformer_default_options():
	params = PyNNDescentTransformer().get_params()
	assert params["n_neighbors"] == 30
	assert params["metric"] == "euclidean"
	assert params["early_termination_value"] == 0.001
	assert params["tree_init"] is True


def test_transformer_requires_fit():
	"""transform before fit raises AttributeError; pins the current output (the type is not part of a documented contract)."""
	with pytest.raises(AttributeError):
		PyNNDescentTransformer().transform(None)
	with pytest.raises(AttributeError):
		PyNNDescentTransformer().transform(np.zeros((2, 3), dtype=np.float32))


def test_transformer_fit_transform_leaves_compressed_index(data, queries):
	transformer = PyNNDescentTransformer(n_neighbors=5, random_state=0, n_jobs=1)
	transformer.fit_transform(data)
	assert not hasattr(transformer.index_, "_neighbor_graph")
	graph = transformer.transform(queries)
	assert graph.shape == (60, 300)


def test_transformer_pickle_round_trip(data, queries):
	transformer = PyNNDescentTransformer(n_neighbors=5, random_state=0, n_jobs=1).fit(data)
	restored = pickle.loads(pickle.dumps(transformer))
	a = transformer.transform(queries)
	b = restored.transform(queries)
	assert a.shape == b.shape
	rows_a = [set(a.indices[a.indptr[i] : a.indptr[i + 1]]) for i in range(60)]
	rows_b = [set(b.indices[b.indptr[i] : b.indptr[i + 1]]) for i in range(60)]
	assert _recall(rows_a, rows_b) >= 0.95


# ---------------------------------------------------- other storage and metrics


@pytest.mark.parametrize("scheme", ["uint4", "binary"])
def test_low_bit_quantization_returns_exact_reranked_distances(data, queries, scheme):
	index = _build(data, quantization=scheme)
	found, distances = index.query(queries, k=5, epsilon=0.3, candidate_width=60)
	assert found.shape == (60, 5)
	assert (np.diff(distances, axis=1) >= 0).all()
	true = cdist(queries, data)
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, found, axis=1), rtol=1e-4, atol=1e-5
	)


def test_bit_hamming_on_packed_bits():
	rng = np.random.RandomState(8)
	bits = rng.randint(0, 2, size=(200, 64)).astype(np.uint8)
	packed = np.packbits(bits, axis=1)
	index = _build(packed, metric="bit_hamming", n_neighbors=10)
	indices, distances = index.neighbor_graph
	true = cdist(bits, bits, "hamming") * 64
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, indices, axis=1), atol=1e-4
	)
	assert _graph_recall(index, bits, "hamming") >= 0.8
	found, found_d = index.query(packed[:20], k=1)
	assert (found[:, 0] == np.arange(20)).mean() >= 0.8
	true_q = cdist(bits[:20], bits, "hamming") * 64
	np.testing.assert_allclose(
		found_d, np.take_along_axis(true_q, found, axis=1), atol=1e-4
	)


def test_sparse_metric_kwds_and_metric_choice(sparse_pair):
	train, test = sparse_pair
	index = _build(train, metric="minkowski", metric_kwds={"p": 3})
	indices, distances = index.neighbor_graph
	true = cdist(train.toarray(), train.toarray(), "minkowski", p=3)
	np.testing.assert_allclose(
		distances, np.take_along_axis(true, indices, axis=1), rtol=1e-3, atol=1e-4
	)
	cosine = _build(train, metric="cosine")
	found, found_d = cosine.query(test, k=5, epsilon=0.3)
	true_q = cdist(test.toarray(), train.toarray(), "cosine")
	assert _recall(found, np.argsort(true_q, axis=1)[:, :5]) >= 0.9
	np.testing.assert_allclose(
		found_d, np.take_along_axis(true_q, found, axis=1), rtol=1e-3, atol=1e-4
	)
