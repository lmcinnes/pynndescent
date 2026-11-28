"""Tests for pynndescent.optimal_transport (network simplex and Sinkhorn).

Expected values come from hand-solved 2x2 problems, closed-form 1-D
Wasserstein distances (scipy.stats.wasserstein_distance), scipy's linear
programming solver, the marginal constraints of a transport plan, and a plain
numpy Sinkhorn loop.
"""

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance

import pynndescent.optimal_transport as ot


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _problem(rs, n, m):
	a = rs.rand(n) + 0.05
	b = rs.rand(m) + 0.05
	return a / a.sum(), b / b.sum(), rs.rand(n, m)


def _setup(a, b, cost, mixing=False):
	n, m = len(a), len(b)
	node_arc_data, spanning_tree, graph = ot.allocate_graph_structures(n, m, mixing)
	ot.initialize_supply(a, -b, graph, node_arc_data.supply)
	ot.initialize_cost(cost, graph, node_arc_data.cost)
	ok = ot.initialize_graph_structures(graph, node_arc_data, spanning_tree)
	return ok, node_arc_data, spanning_tree, graph


def _solve(a, b, cost, mixing=False, max_iter=100000):
	ok, nad, st, graph = _setup(a, b, cost, mixing)
	assert ok
	status = ot.network_simplex_core(nad, st, graph, max_iter)
	return status, nad, st, graph


def _flow_matrix(nad, graph):
	n, m = graph.n, graph.m
	F = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			F[i, j] = nad.flow[ot.arc_id(i * m + j, graph)]
	return F


def _linprog_cost(a, b, cost):
	n, m = cost.shape
	A_eq = np.zeros((n + m, n * m))
	for i in range(n):
		A_eq[i, i * m : (i + 1) * m] = 1
	for j in range(m):
		A_eq[n + j, j::m] = 1
	res = linprog(cost.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")
	assert res.status == 0
	return res.fun


def _conservation_error(nad, graph):
	n_total = graph.n_arcs + graph.n_nodes
	balance = np.zeros(graph.n_nodes + 1)
	for e in range(n_total):
		balance[nad.source[e]] += nad.flow[e]
		balance[nad.target[e]] -= nad.flow[e]
	return np.abs(balance - nad.supply[: graph.n_nodes + 1]).max()


def _check_tree(st, nad, graph):
	N = graph.n_nodes + 1
	root = st.root
	assert root == graph.n_nodes
	parent, thread, rev = st.parent, st.thread, st.rev_thread
	assert parent[root] == -1
	children = {u: [] for u in range(N)}
	for u in range(N):
		if u != root:
			assert 0 <= parent[u] < N
			children[int(parent[u])].append(u)

	def subtree(u):
		out = [u]
		for c in children[u]:
			out.extend(subtree(c))
		return out

	subtrees = {u: set(subtree(u)) for u in range(N)}
	assert subtrees[root] == set(range(N))
	order = [root]
	u = int(thread[root])
	while u != root:
		order.append(u)
		assert len(order) <= N
		u = int(thread[u])
	assert sorted(order) == list(range(N))
	pos = {u: i for i, u in enumerate(order)}
	for u in range(N):
		assert rev[thread[u]] == u
		size = len(subtrees[u])
		assert st.succ_num[u] == size
		assert set(order[pos[u] : pos[u] + size]) == subtrees[u]
		assert st.last_succ[u] == order[pos[u] + size - 1]
	n_total = graph.n_arcs + graph.n_nodes
	tree_arcs = [e for e in range(n_total) if st.state[e] == ot.ArcState.STATE_TREE]
	assert len(tree_arcs) == N - 1
	for u in range(N):
		if u == root:
			continue
		e = int(st.pred[u])
		assert st.state[e] == ot.ArcState.STATE_TREE
		assert {int(nad.source[e]), int(nad.target[e])} == {u, int(parent[u])}
		assert bool(st.forward[u]) == (int(nad.source[e]) == u)
		reduced = nad.cost[e] + nad.pi[nad.source[e]] - nad.pi[nad.target[e]]
		assert abs(reduced) <= 1e-8 * max(1.0, abs(nad.cost[e]))


def _lca(parent, u, v):
	ancestors = set()
	x = u
	while x != -1:
		ancestors.add(x)
		x = int(parent[x])
	x = v
	while x not in ancestors:
		x = int(parent[x])
	return x


def _expected_delta(join, in_arc, nad, st):
	assert st.state[in_arc] == ot.ArcState.STATE_LOWER
	delta = np.inf
	u = int(nad.target[in_arc])
	while u != join:
		if not st.forward[u]:
			delta = min(delta, nad.flow[st.pred[u]])
		u = int(st.parent[u])
	u = int(nad.source[in_arc])
	while u != join:
		if st.forward[u]:
			delta = min(delta, nad.flow[st.pred[u]])
		u = int(st.parent[u])
	return delta


def _reduced_costs(nad, st, graph):
	e = np.arange(graph.n_arcs)
	return st.state[e] * (nad.cost[e] + nad.pi[nad.source[e]] - nad.pi[nad.target[e]])


def _drive(a, b, cost, mixing):
	"""Run the simplex loop one pivot at a time, checking every invariant."""
	ok, nad, st, graph = _setup(a, b, cost, mixing)
	assert ok
	assert _conservation_error(nad, graph) < 1e-12
	bounded, in_arc = ot.construct_initial_pivots(graph, nad, st)
	assert bounded
	assert _conservation_error(nad, graph) < 1e-12
	_check_tree(st, nad, graph)
	block = max(int(np.sqrt(graph.n_arcs)), 10)
	in_arc, next_arc = ot.find_entering_arc(block, 0, graph.n_arcs, st.state, nad, in_arc)
	n_pivots = 0
	last_cost = ot.total_cost(nad.flow, nad.cost)
	while in_arc >= 0:
		n_pivots += 1
		assert n_pivots < 5000
		assert _reduced_costs(nad, st, graph)[in_arc] < 0
		join = ot.find_join_node(nad.source, nad.target, st.succ_num, st.parent, in_arc)
		assert join == _lca(st.parent, int(nad.source[in_arc]), int(nad.target[in_arc]))
		leaving = ot.find_leaving_arc(join, in_arc, nad, st)
		assert leaving.delta == pytest.approx(_expected_delta(join, in_arc, nad, st), abs=1e-14)
		assert leaving.change == (leaving.delta < ot.MAX)
		ot.update_flow(join, leaving, nad, st, in_arc)
		assert _conservation_error(nad, graph) < 1e-10
		assert (nad.flow[: graph.n_arcs + graph.n_nodes] >= -1e-12).all()
		new_cost = ot.total_cost(nad.flow, nad.cost)
		assert new_cost <= last_cost + 1e-9
		last_cost = new_cost
		if leaving.change:
			assert st.state[in_arc] == ot.ArcState.STATE_TREE
			ot.update_spanning_tree(st, leaving, join, in_arc, nad.source)
			ot.update_potential(leaving, nad.pi, nad.cost, st)
			_check_tree(st, nad, graph)
		in_arc, next_arc = ot.find_entering_arc(block, next_arc, graph.n_arcs, st.state, nad, in_arc)
	assert (_reduced_costs(nad, st, graph) >= -1e-9).all()
	assert (nad.flow[graph.n_arcs : graph.n_arcs + graph.n_nodes] <= 1e-12).all()
	return nad, graph, n_pivots


# ---------------------------------------------------------------------------
# constants and small helpers
# ---------------------------------------------------------------------------


def test_enums_and_constants():
	"""Pins the current output: ProblemStatus and ArcState values, INVALID and the infinity constants."""
	assert [s.value for s in ot.ProblemStatus] == [0, 1, 2, 3]
	assert ot.ProblemStatus.OPTIMAL.value == 0
	assert ot.ProblemStatus.MAX_ITER_REACHED.value == 1
	assert ot.ProblemStatus.UNBOUNDED.value == 2
	assert ot.ProblemStatus.INFEASIBLE.value == 3
	assert (ot.ArcState.STATE_UPPER, ot.ArcState.STATE_TREE, ot.ArcState.STATE_LOWER) == (-1, 0, 1)
	assert isinstance(ot.ArcState.STATE_LOWER, int)
	assert ot.INVALID == -1
	assert ot.INFINITY == ot.MAX == np.finfo(np.float64).max


def test_isclose_follows_the_documented_tolerance_rule():
	"""Check isclose follows the documented tolerance rule."""
	assert ot.isclose(1.0, 1.0)
	assert ot.isclose(1.0, 1.0 + 1e-7)
	assert not ot.isclose(1.0, 1.001)
	assert ot.isclose(0.0, 1e-16)
	assert not ot.isclose(1e-14, 0.0)
	assert ot.isclose(0.9, 1.0, 0.2)
	assert not ot.isclose(0.7, 1.0, 0.2)
	assert ot.isclose(0.0, 0.05, 0.0, 0.1)
	assert not ot.isclose(0.0, 0.2, 0.0, 0.1)
	assert ot.isclose(110.0, 100.0, 0.2, 0.0)
	assert not ot.isclose(100.0, 110.0, 0.05, 0.0)


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------


def test_allocate_graph_structures_without_mixing_hand_case():
	"""Check the 2x2 layout by hand; the uint16 arc endpoint dtype pins the current output."""
	nad, st, graph = ot.allocate_graph_structures(2, 2, False)
	assert (graph.n_nodes, graph.n_arcs, graph.n, graph.m) == (4, 4, 2, 2)
	assert graph.use_arc_mixing is False or not graph.use_arc_mixing
	assert graph.mixing_coeff == 0 and graph.subsequence_length == 0
	np.testing.assert_array_equal(nad.source[:4], [2, 2, 3, 3])
	np.testing.assert_array_equal(nad.target[:4], [0, 1, 0, 1])
	max_arc_num = 4 + 2 * 4
	for arr in (nad.source, nad.target, nad.cost, nad.flow, st.state):
		assert arr.shape == (max_arc_num,)
	for arr in (nad.supply, nad.pi, st.parent, st.pred, st.thread, st.rev_thread, st.succ_num, st.last_succ, st.forward):
		assert arr.shape == (5,)
	assert st.root == 4
	assert nad.cost.dtype == np.float64 and (nad.cost == 1.0).all()
	assert (nad.flow == 0).all() and (nad.supply == 0).all() and (nad.pi == 0).all()
	assert nad.source.dtype == np.uint16 and nad.target.dtype == np.uint16


@pytest.mark.parametrize("n,m", [(1, 1), (2, 3), (3, 2), (5, 7), (12, 12)])
def test_allocate_graph_structures_mixing_fields_and_sizes(n, m):
	"""Check allocate graph structures mixing fields and sizes."""
	nad, st, graph = ot.allocate_graph_structures(n, m, True)
	assert graph.n_nodes == n + m and graph.n_arcs == n * m
	assert graph.use_arc_mixing
	k = max(int(np.sqrt(n * m)), 10)
	assert graph.mixing_coeff == k
	assert graph.subsequence_length == (n * m) // k + 1
	assert graph.num_big_subsequences == (n * m) % k
	assert graph.num_total_big_subsequence_numbers == graph.subsequence_length * graph.num_big_subsequences
	assert nad.source.shape[0] == n * m + 2 * (n + m)
	arcs = set(zip(nad.source[: n * m].tolist(), nad.target[: n * m].tolist()))
	assert arcs == {(n + m - 1 - i, n + m - 1 - n - j) for i in range(n) for j in range(m)}


@pytest.mark.parametrize("mixing", [False, True])
def test_arc_id_is_a_bijection_consistent_with_source_and_target(mixing):
	"""Check arc id is a bijection consistent with source and target."""
	for n in range(1, 9):
		for m in range(1, 9):
			nad, _, graph = ot.allocate_graph_structures(n, m, mixing)
			ids = [ot.arc_id(arc, graph) for arc in range(n * m)]
			assert sorted(ids) == list(range(n * m))
			for arc, pos in enumerate(ids):
				assert nad.source[pos] == graph.n_nodes - arc // m - 1
				assert nad.target[pos] == graph.n_nodes - (arc % m + n) - 1


def test_arc_id_without_mixing_reverses_the_order():
	"""Check arc id without mixing reverses the order."""
	_, _, graph = ot.allocate_graph_structures(2, 3, False)
	assert [ot.arc_id(a, graph) for a in range(6)] == [5, 4, 3, 2, 1, 0]


def test_initialize_supply_places_left_then_right_nodes_in_reverse():
	"""Check initialize supply places left then right nodes in reverse."""
	_, _, graph = ot.allocate_graph_structures(2, 3, False)
	supply = np.zeros(6)
	ot.initialize_supply(np.array([0.5, 0.5]), np.array([-0.2, -0.3, -0.5]), graph, supply)
	np.testing.assert_array_equal(supply[:5], [-0.5, -0.3, -0.2, 0.5, 0.5])
	assert supply[5] == 0


@pytest.mark.parametrize("mixing", [False, True])
def test_initialize_cost_and_set_cost_address_arcs_through_arc_id(mixing):
	"""Check initialize cost and set cost address arcs through arc id."""
	rs = np.random.RandomState(3)
	C = rs.rand(4, 5)
	nad, _, graph = ot.allocate_graph_structures(4, 5, mixing)
	ot.initialize_cost(C, graph, nad.cost)
	for i in range(4):
		for j in range(5):
			assert nad.cost[ot.arc_id(i * 5 + j, graph)] == C[i, j]
	assert (nad.cost[20:] == 1.0).all()
	ot.set_cost(7, 123.5, nad.cost, graph)
	assert nad.cost[ot.arc_id(7, graph)] == 123.5


def test_total_cost_is_the_dot_product():
	"""Check total_cost equals the flow-cost dot product on hand values."""
	flow = np.array([0.5, 0.0, 2.0, 1.5])
	cost = np.array([2.0, 7.0, 0.25, 4.0])
	assert ot.total_cost(flow, cost) == pytest.approx(1.0 + 0.5 + 6.0)
	assert ot.total_cost(np.zeros(3), np.ones(3)) == 0.0


def test_initialize_graph_structures_builds_the_artificial_star_tree():
	"""Check initialize graph structures builds the artificial star tree."""
	a = np.array([0.4, 0.6])
	b = np.array([0.5, 0.5])
	C = np.array([[1.0, 3.0], [2.0, 1.0]])
	ok, nad, st, graph = _setup(a, b, C)
	assert ok
	n_nodes, n_arcs = graph.n_nodes, graph.n_arcs
	artificial = (3.0 + 1) * n_nodes
	assert st.root == n_nodes and st.parent[n_nodes] == -1
	assert st.succ_num[n_nodes] == n_nodes + 1
	assert st.last_succ[n_nodes] == n_nodes - 1
	assert (st.state[:n_arcs] == ot.ArcState.STATE_LOWER).all()
	assert (nad.flow[:n_arcs] == 0).all()
	for u in range(n_nodes):
		e = n_arcs + u
		assert st.parent[u] == n_nodes and st.pred[u] == e
		assert st.state[e] == ot.ArcState.STATE_TREE
		assert st.succ_num[u] == 1 and st.last_succ[u] == u and st.thread[u] == u + 1
		assert nad.flow[e] == pytest.approx(abs(nad.supply[u]))
		if nad.supply[u] >= 0:
			assert st.forward[u] and nad.pi[u] == 0 and nad.cost[e] == 0
			assert (nad.source[e], nad.target[e]) == (u, n_nodes)
		else:
			assert not st.forward[u] and nad.pi[u] == artificial and nad.cost[e] == artificial
			assert (nad.source[e], nad.target[e]) == (n_nodes, u)
	assert nad.supply[n_nodes] == pytest.approx(0.0, abs=1e-15)
	_check_tree(st, nad, graph)
	assert _conservation_error(nad, graph) < 1e-15


def test_initialize_graph_structures_rejects_unbalanced_supply():
	"""Check initialize graph structures rejects unbalanced supply."""
	a = np.array([0.5, 0.5])
	b = np.array([0.5, 0.6])
	ok, _, _, _ = _setup(a, b, np.ones((2, 2)))
	assert not ok


def test_initialize_graph_structures_rejects_empty_graph():
	"""Check initialize graph structures rejects empty graph."""
	nad, st, graph = ot.allocate_graph_structures(0, 0, False)
	assert not ot.initialize_graph_structures(graph, nad, st)


def test_initialize_graph_structures_resets_flow_and_state_on_reuse():
	"""Check initialize graph structures resets flow and state on reuse."""
	a = np.array([0.4, 0.6])
	b = np.array([0.5, 0.5])
	C = np.array([[1.0, 3.0], [2.0, 1.0]])
	_, nad, st, graph = _setup(a, b, C)
	nad.flow[:4] = 9.0
	st.state[:4] = ot.ArcState.STATE_UPPER
	assert ot.initialize_graph_structures(graph, nad, st)
	assert (nad.flow[:4] == 0).all() and (st.state[:4] == ot.ArcState.STATE_LOWER).all()


# ---------------------------------------------------------------------------
# network simplex
# ---------------------------------------------------------------------------


def test_network_simplex_hand_solved_2x2_problems():
	"""Check network simplex hand solved 2x2 problems."""
	status, nad, _, graph = _solve(np.array([0.5, 0.5]), np.array([0.25, 0.75]), np.array([[0.0, 1.0], [1.0, 0.0]]))
	assert status == ot.ProblemStatus.OPTIMAL
	np.testing.assert_allclose(_flow_matrix(nad, graph), [[0.25, 0.25], [0.0, 0.5]], atol=1e-12)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(0.25)

	status, nad, _, graph = _solve(np.array([0.4, 0.6]), np.array([0.5, 0.5]), np.array([[1.0, 3.0], [2.0, 1.0]]))
	assert status == ot.ProblemStatus.OPTIMAL
	np.testing.assert_allclose(_flow_matrix(nad, graph), [[0.4, 0.0], [0.1, 0.5]], atol=1e-12)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(1.1)

	status, nad, _, _ = _solve(np.array([0.3, 0.7]), np.array([0.3, 0.7]), np.array([[0.0, 5.0], [5.0, 0.0]]))
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(0.0, abs=1e-14)


def test_network_simplex_single_source_and_single_sink():
	"""Check network simplex single source and single sink."""
	status, nad, _, graph = _solve(np.array([1.0]), np.array([0.2, 0.3, 0.5]), np.array([[4.0, 2.0, 1.0]]))
	assert status == ot.ProblemStatus.OPTIMAL
	np.testing.assert_allclose(_flow_matrix(nad, graph), [[0.2, 0.3, 0.5]], atol=1e-12)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(0.8 + 0.6 + 0.5)
	status, nad, _, graph = _solve(np.array([0.5, 0.5]), np.array([1.0]), np.array([[3.0], [5.0]]))
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(4.0)
	status, nad, _, graph = _solve(np.array([1.0]), np.array([1.0]), np.array([[2.5]]))
	assert status == ot.ProblemStatus.OPTIMAL
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(2.5)


@pytest.mark.parametrize("mixing", [False, True])
@pytest.mark.parametrize("n,m", [(2, 3), (4, 4), (6, 5), (7, 9), (12, 3)])
def test_network_simplex_matches_linear_programming(n, m, mixing):
	"""Check network simplex matches linear programming."""
	rs = np.random.RandomState(100 * n + m)
	a, b, C = _problem(rs, n, m)
	status, nad, _, graph = _solve(a, b, C, mixing)
	assert status == ot.ProblemStatus.OPTIMAL
	F = _flow_matrix(nad, graph)
	assert (F >= -1e-12).all()
	np.testing.assert_allclose(F.sum(axis=1), a, atol=1e-10)
	np.testing.assert_allclose(F.sum(axis=0), b, atol=1e-10)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx((F * C).sum(), rel=1e-10)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(_linprog_cost(a, b, C), rel=1e-8)
	assert (nad.flow[graph.n_arcs : graph.n_arcs + graph.n_nodes] == 0).all()
	assert nad.pi[: graph.n_nodes].max() <= 0


@pytest.mark.parametrize("n,m", [(5, 5), (8, 6), (10, 10)])
def test_network_simplex_equals_one_dimensional_wasserstein(n, m):
	"""Check network simplex equals one dimensional wasserstein."""
	rs = np.random.RandomState(n * 31 + m)
	pos_a = np.sort(rs.rand(n)) * 10
	pos_b = np.sort(rs.rand(m)) * 10
	a = rs.rand(n) + 0.1
	a /= a.sum()
	b = rs.rand(m) + 0.1
	b /= b.sum()
	C = np.abs(pos_a[:, None] - pos_b[None, :])
	status, nad, _, _ = _solve(a, b, C)
	assert status == ot.ProblemStatus.OPTIMAL
	expected = wasserstein_distance(pos_a, pos_b, a, b)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(expected, rel=1e-9)


def test_network_simplex_identical_distributions_with_metric_cost_cost_zero():
	"""Check network simplex identical distributions with metric cost cost zero."""
	rs = np.random.RandomState(5)
	pos = np.sort(rs.rand(6))
	a = rs.rand(6) + 0.1
	a /= a.sum()
	C = np.abs(pos[:, None] - pos[None, :])
	status, nad, _, _ = _solve(a, a.copy(), C)
	assert status == ot.ProblemStatus.OPTIMAL
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(0.0, abs=1e-12)


def test_network_simplex_scales_linearly_in_the_cost():
	"""Check network simplex scales linearly in the cost."""
	rs = np.random.RandomState(8)
	a, b, C = _problem(rs, 5, 6)
	base = ot.total_cost(*(lambda r: (r[1].flow, r[1].cost))(_solve(a, b, C)))
	scaled = ot.total_cost(*(lambda r: (r[1].flow, r[1].cost))(_solve(a, b, 3.0 * C)))
	shifted = ot.total_cost(*(lambda r: (r[1].flow, r[1].cost))(_solve(a, b, C + 2.0)))
	assert scaled == pytest.approx(3.0 * base, rel=1e-10)
	assert shifted == pytest.approx(base + 2.0, rel=1e-10)


def test_network_simplex_max_iter_limits_pivots():
	"""Check network simplex max iter limits pivots."""
	rs = np.random.RandomState(9)
	a, b, C = _problem(rs, 8, 8)
	status, nad, _, graph = _solve(a, b, C, max_iter=1)
	assert status == ot.ProblemStatus.MAX_ITER_REACHED
	status, nad, _, graph = _solve(a, b, C, max_iter=0)
	assert status == ot.ProblemStatus.OPTIMAL
	status, nad, _, graph = _solve(a, b, C, max_iter=10**6)
	assert status == ot.ProblemStatus.OPTIMAL


@pytest.mark.parametrize("mixing", [False, True])
@pytest.mark.parametrize("n,m", [(1, 1), (1, 4), (4, 1), (2, 2), (3, 3), (5, 4), (7, 6)])
def test_simplex_steps_keep_tree_flow_and_potential_invariants(n, m, mixing):
	"""Check simplex steps keep tree flow and potential invariants."""
	rs = np.random.RandomState(7 * n + m)
	a, b, C = _problem(rs, n, m)
	nad, graph, n_pivots = _drive(a, b, C, mixing)
	F = _flow_matrix(nad, graph)
	np.testing.assert_allclose(F.sum(axis=1), a, atol=1e-10)
	np.testing.assert_allclose(F.sum(axis=0), b, atol=1e-10)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(_linprog_cost(a, b, C), rel=1e-8)
	assert n_pivots < 5000


def test_simplex_steps_on_integer_costs_with_degenerate_ties():
	"""Check simplex steps on integer costs with degenerate ties."""
	a = np.array([0.25, 0.25, 0.25, 0.25])
	b = np.array([0.5, 0.5])
	C = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]])
	nad, graph, _ = _drive(a, b, C, False)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(1.0)
	C2 = np.ones((3, 3))
	nad, graph, _ = _drive(np.ones(3) / 3, np.ones(3) / 3, C2, True)
	assert ot.total_cost(nad.flow, nad.cost) == pytest.approx(1.0)


def test_find_entering_arc_picks_the_most_negative_reduced_cost_in_a_small_problem():
	"""Check find entering arc picks the most negative reduced cost in a small problem."""
	for seed in range(5):
		rs = np.random.RandomState(seed)
		a, b, C = _problem(rs, 3, 3)
		ok, nad, st, graph = _setup(a, b, C)
		rc = _reduced_costs(nad, st, graph)
		assert rc.min() < 0
		in_arc, next_arc = ot.find_entering_arc(10, 0, graph.n_arcs, st.state, nad, -1)
		assert in_arc == int(np.argmin(rc))


def test_find_entering_arc_reports_no_candidate_at_optimality():
	"""Check find entering arc reports no candidate at optimality."""
	rs = np.random.RandomState(4)
	a, b, C = _problem(rs, 4, 5)
	status, nad, st, graph = _solve(a, b, C)
	assert status == ot.ProblemStatus.OPTIMAL
	in_arc, next_arc = ot.find_entering_arc(10, 0, graph.n_arcs, st.state, nad, 0)
	assert in_arc == -1 and next_arc == 0


def test_find_join_node_is_the_lowest_common_ancestor_in_the_star_tree():
	"""Check find join node is the lowest common ancestor in the star tree."""
	a = np.array([0.4, 0.6])
	b = np.array([0.5, 0.5])
	_, nad, st, graph = _setup(a, b, np.array([[1.0, 3.0], [2.0, 1.0]]))
	for arc in range(graph.n_arcs):
		join = ot.find_join_node(nad.source, nad.target, st.succ_num, st.parent, arc)
		assert join == graph.n_nodes


def test_construct_initial_pivots_gives_a_feasible_tree_with_no_worse_cost():
	"""Check construct initial pivots gives a feasible tree with no worse cost."""
	rs = np.random.RandomState(12)
	for n, m in [(1, 3), (3, 1), (4, 4), (6, 3)]:
		a, b, C = _problem(rs, n, m)
		ok, nad, st, graph = _setup(a, b, C)
		before = ot.total_cost(nad.flow, nad.cost)
		bounded, in_arc = ot.construct_initial_pivots(graph, nad, st)
		assert bounded
		assert _conservation_error(nad, graph) < 1e-12
		_check_tree(st, nad, graph)
		assert ot.total_cost(nad.flow, nad.cost) <= before + 1e-12


def test_update_potential_restores_zero_reduced_cost_on_the_new_tree_arc():
	"""Check update potential restores zero reduced cost on the new tree arc."""
	rs = np.random.RandomState(2)
	a, b, C = _problem(rs, 4, 4)
	ok, nad, st, graph = _setup(a, b, C)
	in_arc, _ = ot.find_entering_arc(10, 0, graph.n_arcs, st.state, nad, -1)
	join = ot.find_join_node(nad.source, nad.target, st.succ_num, st.parent, in_arc)
	leaving = ot.find_leaving_arc(join, in_arc, nad, st)
	assert leaving.change
	pi_before = nad.pi.copy()
	ot.update_flow(join, leaving, nad, st, in_arc)
	ot.update_spanning_tree(st, leaving, join, in_arc, nad.source)
	ot.update_potential(leaving, nad.pi, nad.cost, st)
	reduced = nad.cost[in_arc] + nad.pi[nad.source[in_arc]] - nad.pi[nad.target[in_arc]]
	assert reduced == pytest.approx(0.0, abs=1e-10)
	assert not np.array_equal(pi_before, nad.pi)
	subtree = set()
	u = int(leaving.u_in)
	end = int(st.thread[st.last_succ[u]])
	while u != end:
		subtree.add(u)
		u = int(st.thread[u])
	for v in range(graph.n_nodes + 1):
		if v not in subtree:
			assert nad.pi[v] == pi_before[v]


# ---------------------------------------------------------------------------
# Sinkhorn
# ---------------------------------------------------------------------------


def _numpy_sinkhorn_plan(x, y, cost, reg, iterations=5000):
	K = np.exp(-cost / reg)
	u = np.ones(len(x))
	for _ in range(iterations):
		v = y / (K.T @ u)
		u = x / (K @ v)
	return u[:, None] * K * v[None, :]


def _grid_cost(n, m):
	return np.abs(np.arange(n)[:, None] / (n - 1) - np.arange(m)[None, :] / (m - 1))


def _dist(rs, n):
	v = rs.rand(n) + 0.2
	return v / v.sum()


def test_k_from_cost():
	"""Check K_from_cost equals exp(-cost / regularization) with the dtype and shape preserved."""
	cost = np.array([[0.0, 1.0], [2.0, 4.0]])
	K = ot.K_from_cost(cost, 2.0)
	np.testing.assert_allclose(K, np.exp(-cost / 2.0), rtol=1e-12)
	assert K.shape == cost.shape and K.dtype == cost.dtype
	rs = np.random.RandomState(0)
	C = rs.rand(5, 7)
	np.testing.assert_allclose(ot.K_from_cost(C, 0.3), np.exp(-C / 0.3), rtol=1e-12)


def test_precompute_k_prime_divides_rows_by_x():
	"""Check precompute k prime divides rows by x."""
	rs = np.random.RandomState(1)
	K = rs.rand(4, 3)
	x = rs.rand(4) + 0.1
	np.testing.assert_allclose(ot.precompute_K_prime(K, x), K / x[:, None], rtol=1e-12)


def test_precompute_k_prime_zero_weight_rows_become_huge():
	"""Pins the current output: rows with zero weight are divided by a tiny epsilon and become huge."""
	K = np.full((2, 3), 0.5)
	out = ot.precompute_K_prime(K, np.array([0.0, 0.5]))
	assert (out[0] > 1e300).all()
	np.testing.assert_allclose(out[1], 1.0)


def test_transport_plan_scales_rows_and_columns():
	"""Check transport plan scales rows and columns."""
	rs = np.random.RandomState(2)
	K = rs.rand(4, 5)
	u = rs.rand(4)
	v = rs.rand(5)
	np.testing.assert_allclose(ot.transport_plan(K, u, v), u[:, None] * K * v[None, :], rtol=1e-12)


def test_right_marginal_error_matches_numpy_formula():
	"""Check right marginal error matches numpy formula."""
	rs = np.random.RandomState(3)
	K = rs.rand(4, 6)
	u = rs.rand(4)
	v = rs.rand(6)
	y = rs.rand(6)
	expected = np.sqrt(((y - (u @ K) * v) ** 2).sum())
	assert ot.right_marginal_error(u, K, v, y) == pytest.approx(expected, rel=1e-6)
	v_exact = y / (u @ K)
	assert ot.right_marginal_error(u, K, v_exact, y) == pytest.approx(0.0, abs=1e-6)


def test_right_marginal_error_batch_matches_numpy_formula():
	"""Check right marginal error batch matches numpy formula."""
	rs = np.random.RandomState(4)
	K = rs.rand(4, 6)
	u = rs.rand(4, 3)
	v = rs.rand(6, 3)
	y = rs.rand(3, 6)
	expected = np.sqrt(((y.T - (K.T @ u) * v) ** 2).sum())
	assert ot.right_marginal_error_batch(u, K, v, y) == pytest.approx(expected, rel=1e-6)


def test_relative_change_in_plan():
	"""Check the mean relative change of u*v against numpy and the hand value 1.0 for doubling u."""
	rs = np.random.RandomState(5)
	old_u, new_u = rs.rand(4) + 0.5, rs.rand(4) + 0.5
	old_v, new_v = rs.rand(5) + 0.5, rs.rand(5) + 0.5
	assert ot.relative_change_in_plan(old_u, old_v, old_u, old_v) == pytest.approx(0.0, abs=1e-6)
	old_uv = old_u[:, None] * old_v[None, :]
	expected = (np.abs(old_uv - new_u[:, None] * new_v[None, :]) / old_uv).mean()
	assert ot.relative_change_in_plan(old_u, old_v, new_u, new_v) == pytest.approx(expected, rel=1e-6)
	assert ot.relative_change_in_plan(np.ones(2), np.ones(2), 2 * np.ones(2), np.ones(2)) == pytest.approx(1.0)


@pytest.mark.parametrize("reg", [1.0, 0.3])
def test_sinkhorn_iterations_reach_the_prescribed_marginals(reg):
	"""Check sinkhorn iterations reach the prescribed marginals."""
	rs = np.random.RandomState(6)
	n, m = 6, 8
	x, y = _dist(rs, n), _dist(rs, m)
	cost = _grid_cost(n, m)
	K = np.exp(-cost / reg)
	u, v = ot.sinkhorn_iterations(x, y, np.full(n, 1.0 / n), np.full(m, 1.0 / m), K)
	plan = u[:, None] * K * v[None, :]
	np.testing.assert_allclose(plan.sum(axis=0), y, atol=1e-6)
	np.testing.assert_allclose(plan.sum(axis=1), x, atol=1e-6)
	np.testing.assert_allclose(plan, _numpy_sinkhorn_plan(x, y, cost, reg), atol=1e-6)


def test_sinkhorn_iterations_with_zero_iterations_return_the_start_values():
	"""Check sinkhorn iterations with zero iterations return the start values."""
	x, y = np.full(3, 1 / 3), np.full(4, 0.25)
	u0, v0 = np.full(3, 0.7), np.full(4, 0.2)
	u, v = ot.sinkhorn_iterations(x, y, u0, v0, np.ones((3, 4)), 0)
	np.testing.assert_array_equal(u, u0)
	np.testing.assert_array_equal(v, v0)


def test_sinkhorn_iterations_batch_matches_single_problem_marginals():
	"""Check sinkhorn iterations batch matches single problem marginals."""
	rs = np.random.RandomState(7)
	n, m, batch = 5, 6, 4
	x = _dist(rs, n)
	y = np.stack([_dist(rs, m) for _ in range(batch)])
	cost = _grid_cost(n, m)
	K = np.exp(-cost / 0.5)
	u, v = ot.sinkhorn_iterations_batch(x, y, np.full((n, batch), 1.0 / n), np.full((m, batch), 1.0 / m), K)
	assert u.shape == (n, batch) and v.shape == (m, batch)
	for b in range(batch):
		plan = u[:, b][:, None] * K * v[:, b][None, :]
		np.testing.assert_allclose(plan.sum(axis=0), y[b], atol=1e-6)
		np.testing.assert_allclose(plan.sum(axis=1), x, atol=1e-6)
		np.testing.assert_allclose(plan, _numpy_sinkhorn_plan(x, y[b], cost, 0.5), atol=1e-6)


@pytest.mark.parametrize("reg", [1.0, 0.25])
def test_sinkhorn_transport_plan_has_the_right_marginals(reg):
	"""Check sinkhorn transport plan has the right marginals."""
	rs = np.random.RandomState(8)
	n, m = 7, 5
	x, y = _dist(rs, n), _dist(rs, m)
	cost = _grid_cost(n, m)
	plan = ot.sinkhorn_transport_plan(x, y, cost=cost, regularization=reg)
	assert plan.shape == (n, m)
	assert (plan > 0).all()
	np.testing.assert_allclose(plan.sum(axis=1), x, atol=1e-6)
	np.testing.assert_allclose(plan.sum(axis=0), y, atol=1e-6)
	np.testing.assert_allclose(plan, _numpy_sinkhorn_plan(x, y, cost, reg), atol=1e-6)


def test_sinkhorn_transport_plan_zero_cost_is_the_product_measure():
	"""Check sinkhorn transport plan zero cost is the product measure."""
	x = np.array([0.3, 0.7])
	y = np.array([0.6, 0.4])
	plan = ot.sinkhorn_transport_plan(x, y)
	np.testing.assert_allclose(plan, np.outer(x, y), atol=1e-12)
	plan = ot.sinkhorn_transport_plan(x, y, cost=np.zeros((2, 2)), regularization=5.0)
	np.testing.assert_allclose(plan, np.outer(x, y), atol=1e-12)


def test_sinkhorn_transport_plan_max_iter_zero_gives_the_initial_scaling():
	"""Check sinkhorn transport plan max iter zero gives the initial scaling."""
	x = np.array([0.3, 0.7])
	y = np.array([0.6, 0.4])
	cost = np.array([[0.0, 1.0], [1.0, 0.0]])
	plan = ot.sinkhorn_transport_plan(x, y, cost=cost, regularization=1.0, max_iter=0)
	K = np.exp(-cost)
	np.testing.assert_allclose(plan, 0.25 * K, rtol=1e-12)


def test_sinkhorn_distance_default_arguments():
	"""Pins the current output: with no cost the default distance is 0.0."""
	x = np.array([0.3, 0.7])
	y = np.array([0.6, 0.4])
	assert ot.sinkhorn_distance(x, y) == 0.0


def test_sinkhorn_distance_is_the_cost_of_the_plan():
	"""Check sinkhorn distance is the cost of the plan."""
	rs = np.random.RandomState(9)
	n, m = 6, 6
	x, y = _dist(rs, n), _dist(rs, m)
	cost = rs.rand(n, m)
	plan = _numpy_sinkhorn_plan(x, y, cost, 0.5)
	assert ot.sinkhorn_distance(x, y, cost, 0.5) == pytest.approx((plan * cost).sum(), rel=1e-6)


@pytest.mark.parametrize("n", [4, 6])
def test_sinkhorn_distance_brackets_the_exact_transport_cost(n):
	"""Check sinkhorn distance brackets the exact transport cost."""
	rs = np.random.RandomState(n)
	x, y = _dist(rs, n), _dist(rs, n)
	cost = _grid_cost(n, n)
	status, nad, _, _ = _solve(x, y, cost)
	exact = ot.total_cost(nad.flow, nad.cost)
	reg = 0.1
	value = ot.sinkhorn_distance(x, y, cost, reg)
	assert exact - 1e-6 <= value <= exact + reg * np.log(n * n) + 1e-6


def test_sinkhorn_distance_decreases_with_regularization_and_hits_the_product_limit():
	"""Check sinkhorn distance decreases with regularization and hits the product limit."""
	rs = np.random.RandomState(11)
	n = 5
	x, y = _dist(rs, n), _dist(rs, n)
	cost = _grid_cost(n, n)
	values = [ot.sinkhorn_distance(x, y, cost, reg) for reg in (1e4, 1.0, 0.3, 0.1)]
	assert values[0] == pytest.approx(x @ cost @ y, abs=1e-4)
	assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))
	assert values[-1] >= wasserstein_distance(np.arange(n) / (n - 1), np.arange(n) / (n - 1), x, y) - 1e-6


def test_sinkhorn_distance_batch_matches_row_by_row_distance():
	"""Check sinkhorn distance batch matches row by row distance."""
	rs = np.random.RandomState(12)
	n = 6
	x = _dist(rs, n)
	y = np.stack([_dist(rs, n) for _ in range(n)])
	cost = _grid_cost(n, n)
	got = ot.sinkhorn_distance_batch(x, y, cost, 0.5)
	assert got.shape == (n,)
	expected = [ot.sinkhorn_distance(x, y[i], cost, 0.5) for i in range(n)]
	np.testing.assert_allclose(got, expected, rtol=1e-6)
	for i in range(n):
		plan = _numpy_sinkhorn_plan(x, y[i], cost, 0.5)
		assert got[i] == pytest.approx((plan * cost).sum(), rel=1e-6)


def test_make_fixed_cost_sinkhorn_distance_matches_sinkhorn_distance():
	"""Check make fixed cost sinkhorn distance matches sinkhorn distance."""
	rs = np.random.RandomState(13)
	n = 6
	pos = np.sort(rs.rand(n))
	cost = np.abs(pos[:, None] - pos[None, :])
	closure = ot.make_fixed_cost_sinkhorn_distance(cost, 0.2)
	for _ in range(3):
		x, y = _dist(rs, n), _dist(rs, n)
		assert closure(x, y) == pytest.approx(ot.sinkhorn_distance(x, y, cost, 0.2), rel=1e-9)
		assert closure(x, y) == pytest.approx(closure(y, x), rel=1e-6)
