"""Search indexes adapted to a sample of the queries they will answer.

When the queries come from a different distribution than the indexed data (text
queries against image embeddings, say), much of a search's effort can go into
reaching the region of the graph that holds the query's neighbours, and graph
edges built from the data alone may not connect the points a query's neighbours
lie among. Given example queries and their nearest neighbours in the data,
``NNDescent.prepare(query_sample=...)`` adapts the index to them:

* query-induced edges: each sample query's few nearest data points (its
  "pivots") get edges to the query's other near neighbours, pruned among
  themselves as relative neighbourhood graph edges are, so that the points
  queries' results lie among are linked to each other (as in RoarGraph's
  projection of a query-data bipartite graph). The edges join the diversified
  k-NN edges before the search graph is symmetrised and degree pruned;
* a query search tree: searches start from the leaves of a random projection
  tree built over the sample queries rather than the data, each leaf listing
  the nearest data points of the sample queries that fall in it.
"""

import numba
import numpy as np

from pynndescent.rp_trees import FlatTree, convert_tree_format, make_forest


@numba.njit(parallel=True)
def _pivot_edges(
    data, dist, pivots, starts, ends, queries, candidates, out_indices, out_distances
):
    max_edges = out_indices.shape[1]
    for u in numba.prange(pivots.shape[0]):
        p = pivots[u]
        n_candidates = (ends[u] - starts[u]) * candidates.shape[1]
        cand = np.empty(n_candidates, dtype=np.int32)
        cand_d = np.empty(n_candidates, dtype=np.float32)
        m = 0
        for t in range(starts[u], ends[u]):
            q = queries[t]
            for j in range(candidates.shape[1]):
                c = candidates[q, j]
                if c < 0 or c == p:
                    continue
                cand[m] = c
                cand_d[m] = dist(data[p], data[c])
                m += 1
        order = np.argsort(cand_d[:m])
        kept = 0
        last = -1
        for a in order:
            if kept >= max_edges:
                break
            c = cand[a]
            if c == last:
                continue
            last = c
            d = cand_d[a]
            keep = True
            for r in range(kept):
                kept_c = out_indices[p, r]
                if kept_c == c or dist(data[kept_c], data[c]) < d:
                    keep = False
                    break
            if keep:
                out_indices[p, kept] = c
                out_distances[p, kept] = d
                kept += 1


def query_induced_edges(data, dist, query_neighbors, n_candidates=64, max_edges=16, n_pivots=4):
    """Edges among the data points the sample queries' neighbours lie among.

    Each of a query's ``n_pivots`` nearest data points is offered edges to the
    query's other neighbours among its first ``n_candidates``; a point keeps, of
    all the edges it is offered, at most ``max_edges``, closest first, dropping
    any that a kept edge's endpoint is closer to (relative neighbourhood
    pruning).

    Returns ``(indices, distances)``: (n_points, max_edges) rows, -1 / inf padded.
    """
    query_neighbors = np.asarray(query_neighbors)
    n_queries = query_neighbors.shape[0]
    n_pivots = min(n_pivots, query_neighbors.shape[1])
    candidates = np.ascontiguousarray(query_neighbors[:, :n_candidates], dtype=np.int32)
    pivot_of = query_neighbors[:, :n_pivots].T.ravel().astype(np.int64)
    query_of = np.tile(np.arange(n_queries, dtype=np.int64), n_pivots)
    valid = pivot_of >= 0
    pivot_of, query_of = pivot_of[valid], query_of[valid]
    order = np.argsort(pivot_of, kind="stable")
    pivots, starts, counts = np.unique(pivot_of[order], return_index=True, return_counts=True)
    indices = np.full((data.shape[0], max_edges), -1, dtype=np.int32)
    distances = np.full((data.shape[0], max_edges), np.inf, dtype=np.float32)
    _pivot_edges(
        data, dist, pivots.astype(np.int32), starts, starts + counts,
        np.ascontiguousarray(query_of[order]), candidates, indices, distances,
    )
    return indices, distances


@numba.njit(parallel=True)
def merge_edge_rows(indices, distances, extra_indices, extra_distances):
    """Append each row's extra edges (those not already present) to its edges."""
    n, width = indices.shape
    out_indices = np.full((n, width + extra_indices.shape[1]), -1, dtype=np.int32)
    out_distances = np.full((n, width + extra_indices.shape[1]), np.inf, dtype=np.float32)
    for i in numba.prange(n):
        for j in range(width):
            out_indices[i, j] = indices[i, j]
            out_distances[i, j] = distances[i, j]
        pos = width
        for j in range(extra_indices.shape[1]):
            c = extra_indices[i, j]
            if c < 0:
                break
            present = False
            for t in range(width):
                if indices[i, t] == c:
                    present = True
                    break
            if not present:
                out_indices[i, pos] = c
                out_distances[i, pos] = extra_distances[i, j]
                pos += 1
    return out_indices, out_distances


def query_search_tree(
    queries, query_neighbors, n_points, leaf_size, max_entries, n_ranks, angular,
    rng_state, random_state, n_jobs=None,
):
    """A random projection tree over the sample queries for finding search entry points.

    Each leaf lists (up to ``max_entries`` of) the nearest data points of the
    queries in it, taking every query's nearest first, then every query's second
    nearest, and so on, from each query's ``n_ranks`` nearest; the leaves list at
    most ``n_points`` points in all, as the search trees index arrays do.
    """
    forest = make_forest(queries, leaf_size, 1, leaf_size, rng_state, random_state, n_jobs, angular)
    flat = convert_tree_format(forest[0], queries.shape[0], queries.shape[1])
    children = flat.children.copy()
    leaves = np.flatnonzero(children[:, 0] <= 0)
    per_leaf = int(max(1, min(max_entries, n_points // len(leaves))))
    ranked = np.asarray(query_neighbors)[:, :n_ranks]
    fallback = ranked[ranked >= 0][:1]
    indices = np.empty(n_points, dtype=np.int32)
    pos = 0
    for node in leaves:
        members = flat.indices[-children[node, 0] : -children[node, 1]]
        entries = ranked[members].T.ravel()
        entries = entries[entries >= 0]
        _, first = np.unique(entries, return_index=True)
        chosen = entries[np.sort(first)][:per_leaf]
        if chosen.shape[0] == 0:
            chosen = fallback
        indices[pos : pos + chosen.shape[0]] = chosen
        children[node, 0] = -pos
        children[node, 1] = -(pos + chosen.shape[0])
        pos += chosen.shape[0]
    # The search trees' index arrays are as long as the data; the rest is unused.
    indices[pos:] = indices[0]
    return FlatTree(flat.hyperplanes, flat.offsets, children, indices, per_leaf)
