Release Notes
=============

0.8.0
-----

PyNNDescent 0.8 adds fast quantized search, maximum inner product search,
adaptation of the index to an out-of-distribution query workload, and a
faster progressive build for high dimensional data. See
:doc:`quantization_and_query_adaptation` for a guide to the new options.

New features
~~~~~~~~~~~~

* ``quantization="sq8"`` and ``quantization="sq4"``: per-feature 8- and 4-bit
  scalar quantization for ``euclidean``, ``cosine`` and ``dot`` metrics (and
  ``mips``) on dense data. The float32 query is scored against the codes with
  integer dot products (using AVX-512 VNNI where available) and the best
  candidates are reranked exactly within the same compiled call. This is the
  fastest search PyNNDescent offers; ``candidate_width`` acts as the beam
  width.
* ``rerank_precision="float16"`` keeps a half precision copy of the data and
  reranks sq8/sq4 candidates from it, halving the memory traffic of the
  reranking step.
* ``quantization_dims`` searches sq8/sq4 codes of the data projected onto its
  top principal components (a number of components, or a fraction of variance
  to keep), which is faster for high dimensional data whose variance is
  concentrated in fewer directions, such as many text embeddings.
* ``metric="mips"``: maximum inner product search on unnormalized dense data,
  via a euclidean index of the data augmented with one extra coordinate. Query
  distances are negative inner products. ``mips`` searches need much smaller
  ``epsilon`` values (typically 0.002 to 0.05) than other metrics.
* ``prepare(query_sample=..., query_neighbors=...)`` adapts the index to a
  query distribution that differs from the data's (text queries against image
  embeddings, for example): the search graph gains edges among the sample
  queries' nearest data points and searches start from a tree built over the
  sample queries. Queries like the sample then reach a given recall with less
  work. ``update()`` accepts the same arguments to keep the adaptation.
* ``progressive_build`` (default ``"auto"``): for dense float32 data with
  euclidean, cosine or dot metrics from 512 features up, the RP-tree leaf join
  and the first NN-descent iterations run on 8-bit codes before refining with
  exact distances, making index construction around 25-45% faster at equal or
  better graph quality. From 1024 features the initialisation forest is built
  on a random projection of the data.
* ``PyNNDescentTransformer`` accepts ``quantization``, ``rerank_precision``,
  ``quantization_dims`` and ``progressive_build``.

Performance
~~~~~~~~~~~

* Graph search bounds candidate distance computations early and prefetches
  the next candidates' vectors and adjacency rows, and the search graph is
  stored as fixed-width, cache-line aligned adjacency rows.
* The data and search graph are reordered with reverse Cuthill-McKee ordering
  for better memory locality during search.
* The NN-descent builds for ``euclidean``, ``cosine``, ``dot``,
  ``inner_product`` and ``proxy_inner_product`` are compiled once and cached on
  disk, as is the search function for ``proxy_inner_product``.
* ``update()`` with only fresh data skips scanning the graph for edges to
  updated rows, and int8 quantization needs one fewer copy of the data.

Behaviour changes
~~~~~~~~~~~~~~~~~

* ``true_angular`` now returns the angle between vectors as a fraction of
  pi, ``θ/π``, for every pair of vectors; in 0.7 it returned ``1 - θ/π`` (a
  similarity), and pairs at 90 degrees or more were treated as infinitely
  far apart. A zero vector is at distance 0.5 (orthogonal) from any non-zero
  vector.
* ``bit_jaccard`` indexes now report standard Jaccard distances, like
  ``jaccard``; the ``bit_jaccard`` distance function itself uses ``log2``
  rather than the natural logarithm.
* Dense ``cosine``, ``dot`` and ``hellinger`` distances are clamped at 0, so
  floating point rounding can no longer make them negative (or NaN).
* ``proxy_inner_product`` search results are ranked by inner product, and a
  zero inner product is treated as maximally distant.
* ``circular_kantorovich`` centers the CDF difference with its median rather
  than the median of its ``p``-th powers, and ``sinkhorn`` ignores zero-mass
  bins and normalizes the marginals, so both give different (correct) values
  from 0.7.
* ``NNDescent`` validates its inputs (``n_neighbors`` and the data) up front,
  and ``transform(None)`` on a compressed ``PyNNDescentTransformer`` raises a
  clear error.
* Sparse training and query data are canonicalized (sorted indices, no
  duplicates) without modifying the caller's arrays, and ``update()`` on a
  sparse index raises before changing any index state.
* New ``NNDescent`` parameters (``rerank_precision``, ``quantization_dims``
  and ``progressive_build``) are added after the existing ones, so positional
  arguments keep their meaning.
* Python 3.11 or later is required (3.10 is no longer supported), along with
  numpy 1.23, scipy 1.9, scikit-learn 1.1 and numba 0.57 or later. Python 3.15
  is supported.

Bug fixes
~~~~~~~~~

* ``update()`` on an index prepared with ``quantization="binary"`` failed.
* Packed-bit queries lost their dtype, and query shapes were not validated.
* A spurious ``n_neighbors`` warning was raised when ``n_neighbors`` equalled
  the number of samples.

0.7.0
-----

PyNNDescent 0.7 is primarily a performance release. Index construction,
index preparation and querying are all substantially faster, especially on
multicore machines and for sparse data, and ``prepare()`` no longer compiles
anything for the most common dense metrics. See :doc:`performance_0_7` for
benchmarks against 0.6.0 and 0.5.13.

Performance
~~~~~~~~~~~

**Index construction**

* The random projection forest used to initialise NN-descent is now built
  breadth-first across all trees at once, keeping only the leaves. Work is
  parallelised within trees as well as across them, so it scales to many more
  cores.
* The NN-descent local join compares one point against four candidates at a
  time for squared euclidean and dot-product distances, and candidate
  construction and graph updates are partitioned by owning thread so that no
  thread scans the whole graph.
* Dense ``cosine`` data is normalised once at construction and searched with
  the (cheaper) dot-product distance, giving cosine the same fast kernels as
  ``dot``.
* Sparse distances (euclidean, manhattan, chebyshev, minkowski, cosine,
  hellinger, canberra, bray-curtis, correlation and their variants) no longer
  allocate temporary vectors on every call, which makes sparse index
  construction several times faster, and up to two orders of magnitude faster
  for sparse random projection trees on many cores.
* Bit-packed data uses hardware popcount for random projection splits.

**Index preparation** (``prepare()``)

* The graph-informed search tree is built in parallel, level by level, instead
  of by a single-threaded recursion, and is written directly in its final flat
  format.
* The search graph is assembled with parallel kernels rather than a chain of
  single-threaded scipy sparse operations, and graph diversification works in
  place without allocation.
* The search function for dense float32 data with euclidean, cosine or dot
  metrics is compiled once and cached on disk, so ``prepare()`` no longer
  compiles a new search function for every index. Graph diversification for
  these metrics is cached too.

**Queries**

* Graph traversal computes distances to four neighbours at a time for squared
  euclidean and dot-product distances.
* Search trees are better balanced (see below), which removes very large
  leaves that previously made some queries start from hundreds or thousands of
  points.
* Sparse search trees store their hyperplanes without padding, making sparse
  indexes much smaller and their queries faster (with the fix described under
  bug fixes, the search tree for 20 Newsgroups TF-IDF shrinks from about
  900 MiB in 0.6 to 5 MiB).

New features
~~~~~~~~~~~~

* ``quantization="int8"`` for ``cosine``, ``direct_cosine`` and ``dot``
  metrics, alongside the existing ``"binary"``, ``"uint8"`` and ``"uint4"``
  options.
* New ``query()`` options: ``candidate_width`` and ``rerank_width`` control
  how many quantised (proxy) candidates are kept and reranked;
  ``return_distances=False`` skips computing distances; and
  ``pre_normalized=True`` skips normalising queries that are already unit
  length (``cosine``, ``direct_cosine`` and ``dot``).
* ``NNDescent.query_indices()`` returns only neighbour indices.
* ``direct_cosine`` metric: cosine distance on unit-normalised data without the
  log transform used by ``cosine``.
* ``prepare(compile_search=True)`` compiles a search function specialised to
  the index, as earlier versions did, instead of using the cached one.
* ``search_tree_leaf_budget`` lets a query seed its search from several search
  tree leaves, chosen best-first by hyperplane margin.
* ``bit_metric`` and ``angular_trees`` let custom (callable) metrics use
  bit-packed data and angular random projection trees.
* Support for ``scipy.sparse`` sparse arrays (``csr_array``) as well as
  sparse matrices.
* Python 3.14 is supported (tested on Linux and Windows).

Behaviour changes
~~~~~~~~~~~~~~~~~

* Search trees now always choose the most balanced of their candidate hub
  splits, and fall back to splitting at the median margin when no candidate is
  balanced enough, instead of making an arbitrarily large leaf (dense data) or
  choosing by edge cuts without a balance check (sparse and bit-packed data).
  Query results can differ slightly from 0.6; accuracy at a given speed is the
  same or better.
* The "reverse diversification" pass of ``prepare()`` has been removed. As
  implemented it re-pruned the forward graph, which forward diversification
  already leaves unchanged at the default ``diversify_prob=1.0``, so search
  graphs are unchanged in that case. With ``diversify_prob < 1`` it compounded
  the pruning probability, so such settings now prune less; diversification
  with ``diversify_prob < 1`` is now also deterministic for a given
  ``random_state``, independent of the number of threads.
* ``n_search_trees`` has had no effect since 0.5.5; passing it now raises a
  ``FutureWarning``, and it will be removed in the next version.
* Dense queries must be 2-D with the same number of features as the indexed
  data; a ``ValueError`` is raised otherwise.
* All-zero queries with angular metrics (which have no defined direction) are
  no longer searched.
* Distance functions are compiled on first use rather than at import, so
  ``import pynndescent`` is faster.
* New ``NNDescent`` parameters (``bit_metric``, ``angular_trees``,
  ``search_tree_leaf_budget``) are added after the existing ones, so
  positional arguments keep their meaning.

Bug fixes
~~~~~~~~~

* Unpickling a sparse index failed on its first query.
* Unpickling a quantised index lost its quantisation, so queries skipped the
  exact reranking step.
* ``update()`` did not normalise new or replaced rows for ``dot``-style
  metrics.
* Sparse search trees padded every hyperplane to the number of rows in the
  dataset, using gigabytes of memory for moderately sized sparse data, and
  failed with a ``ValueError`` on small, high-dimensional sparse datasets.
* Index construction could corrupt the graph on datasets with more than
  16,777,216 points (vertex ids were stored as float32 values).
* Sparse correlation distance was wrong when a shared entry equalled its row
  mean.
* TS-SS distance used twice the defined sector area.
* Graph diversification with ``diversify_prob < 1`` raced on a shared random
  number generator state.
