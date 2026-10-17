Performance improvements in 0.7
===============================

PyNNDescent 0.7 is substantially faster than 0.6 and 0.5 at every stage of
its use: building the approximate k-nearest neighbor graph (``NNDescent(...)``),
preparing the index for search (``prepare()``), and querying it
(``query()``). This page summarises the main changes behind that, and shows
benchmarks comparing 0.7.0 with 0.6.0 and 0.5.13.

In short, on the benchmarks below, compared with 0.6.0 (and 0.5.13):

* **index construction** is 1.5 to 2.4 times faster for dense data and 3 to 11
  times faster for sparse data (2.7 to 12 times and 5 to 87 times faster than
  0.5.13), with equal or slightly better graph accuracy;
* **preparing an index** is 5 to 9 times faster for dense data and 24 to 30
  times faster for sparse data (15 to 79 times and 3 to 11 times faster than
  0.5.13);
* **queries** are about 1.4 times faster on GloVe and 3 to 4 times faster on
  sparse text data at the same accuracy, while on MNIST 0.7 recovers a
  slowdown in 0.6 and is on par with or slightly faster than 0.5.13.

The gains are largest on many cores: much more of the work now runs in
parallel, so 0.7 continues to speed up with more threads where earlier
versions often did not (0.5.13 was slower with 64 threads than with 8 on all
three datasets).

How the benchmarks were run
---------------------------

* **Datasets:**

  * MNIST: 60,000 points, 784 dimensions, euclidean distance;
  * GloVe-100: 1,183,514 points, 100 dimensions, cosine distance;
  * 20 Newsgroups: TF-IDF vectors of the 18,846 documents (scikit-learn's
    ``TfidfVectorizer(min_df=2)``, 75,875 features, sparse), cosine distance.

  MNIST and GloVe-100 use their standard query sets (10,000 points each) and
  ground truth from `ann-benchmarks <https://github.com/erikbern/ann-benchmarks>`_.
  For 20 Newsgroups the last 1,000 documents are held out as queries and the
  ground truth is computed exactly.
* **Parameters:** library defaults, with ``n_neighbors=30``.
* **Hardware:** a dual-socket Intel Xeon Platinum 8370C (2 x 16 cores, 64
  hardware threads). Each benchmark was run with 64 threads, and with 8
  threads pinned to 8 physical cores, to represent a large server and a typical
  workstation.
* **Software:** every version ran in the same Python 3.12 environment with
  numba 0.62.1 and numpy 2.3.5.
* **Timing:** all times are measured after just-in-time compilation has been
  warmed up on a smaller index, so they reflect steady-state performance.
  Construction and preparation times are the median of several runs; query
  throughput is the best of three runs over the full query set, averaged over
  the 8 and 64 thread runs. Queries are run one batch at a time on a single
  thread, which is the library default.

Index construction
------------------

.. image:: perf07_construction.png
    :alt: Speedup of index construction in 0.7.0 over 0.6.0 and 0.5.13

Index construction has two phases: building a random projection forest whose
leaves give each point some initial candidate neighbors, then refining the
graph with nearest neighbor descent.

* **A breadth-first forest.** The random projection forest used to initialise
  NN-descent used to be built one tree per thread, so it could use at most as
  many cores as there are trees (typically 6 to 12), and every tree was built
  depth-first. 0.7 builds all trees together, level by level: each level
  computes which side of its node's hyperplane every point falls on in a single
  parallel pass over the data (comparing each point against several trees'
  hyperplanes at once), then partitions all nodes in parallel. Small nodes are
  finished depth-first, in parallel across all trees. Since NN-descent only
  needs the leaves, the hyperplanes are not kept.
* **Batched distance kernels.** In the NN-descent local join, each point is
  compared with four candidates at a time for squared euclidean and dot-product
  distances, sharing the loads of the first point. The same kernels are used
  for queries.
* **Less contention between threads.** Building the candidate lists and
  applying graph updates are partitioned by the thread that owns each vertex,
  so threads no longer scan the whole graph or each other's updates.
* **Cheaper cosine distance.** Dense data for the ``cosine`` metric is
  normalised once at construction, so cosine distance becomes a dot product and
  uses the batched kernels above.
* **Sparse data.** Sparse distance functions no longer build temporary vectors
  on every call, and sparse random projection splits no longer allocate. For
  sparse data on many cores this made the initialisation forest up to two orders
  of magnitude faster, as threads no longer contend for the memory allocator.

Construction times (median of three runs; two for 0.5.13 on GloVe-100) and
the fraction of each point's true 30 nearest neighbors found (graph recall):

=============  =======  ==========  =========  =========  =============  ============  ============
Dataset        Threads  0.5.13 (s)  0.6.0 (s)  0.7.0 (s)  0.5.13 recall  0.6.0 recall  0.7.0 recall
=============  =======  ==========  =========  =========  =============  ============  ============
MNIST          8        3.99        2.17       1.46       0.997          0.997         0.997
MNIST          64       5.47        1.39       0.58       0.997          0.997         0.997
GloVe-100      8        90.39       46.91      25.70      0.780          0.781         0.786
GloVe-100      64       126.63      24.32      10.17      0.782          0.781         0.786
20 Newsgroups  8        62.73       34.80      11.58      0.951          0.955         0.955
20 Newsgroups  64       225.40      28.89      2.60       0.949          0.954         0.955
=============  =======  ==========  =========  =========  =============  ============  ============

Index preparation
-----------------

.. image:: perf07_prepare.png
    :alt: Speedup of prepare() in 0.7.0 over 0.6.0 and 0.5.13

``prepare()`` turns the k-NN graph into a search index: it builds a search
tree used to find starting points for each query, and a search graph (the k-NN
graph, diversified to remove redundant edges, made undirected and degree
pruned), and it sets up the search function.

* **A parallel search tree.** The graph-informed search tree, which chooses
  splits through highly connected "hub" points of the k-NN graph, was built by
  a single-threaded recursion. It is now built level by level in parallel, in
  the same way as the initialisation forest, and written directly in the flat
  format used for searching. Splits now also always keep at least 10% of a
  node's points on each side, which keeps the tree shallow (this also speeds up
  queries; see below).
* **Parallel search graph assembly.** Diversification now works in place
  without allocating, and making the graph undirected, pruning it and
  reordering it to match the search tree are done with parallel kernels rather
  than a chain of single-threaded scipy sparse matrix operations.
* **No per-index compilation.** Earlier versions compiled a new search function
  for every index, which typically took one to two seconds. For dense data with
  euclidean, cosine or dot metrics, 0.7 uses a search function that is compiled
  once and cached on disk, and is as fast as the specialised one. (Pass
  ``prepare(compile_search=True)`` to compile a specialised search function as
  before.)

Times for ``prepare()``:

=============  =======  ==========  =========  =========
Dataset        Threads  0.5.13 (s)  0.6.0 (s)  0.7.0 (s)
=============  =======  ==========  =========  =========
MNIST          8        3.24        1.51       0.21
MNIST          64       3.24        1.45       0.17
GloVe-100      8        71.85       9.10       1.80
GloVe-100      64       75.61       8.78       0.96
20 Newsgroups  8        4.11        31.44      1.29
20 Newsgroups  64       11.03       30.40      1.03
=============  =======  ==========  =========  =========

Queries
-------

.. image:: perf07_query.png
    :alt: Query throughput of 0.7.0 relative to 0.6.0 and 0.5.13 at equal recall

Query speed has to be compared at equal accuracy: the ``epsilon`` parameter
trades accuracy for speed, so the chart above compares the number of queries
per second each version achieves at the same recall@10 (the fraction of the
true 10 nearest neighbors found), interpolated from the curves below.

.. image:: perf07_query_curves.png
    :alt: Queries per second against recall@10 for 0.5.13, 0.6.0 and 0.7.0

* **Batched distances.** As in construction, graph traversal computes
  distances to four unvisited neighbors at a time.
* **Better search trees.** Each query starts from the leaf of the search tree
  it falls into. Previously, nodes that could not be split evenly became leaves
  regardless of size, and on some data a few leaves held thousands of points
  that every query landing in them had to examine. 0.7 splits such nodes at the
  median, so leaves stay small; on MNIST with cosine distance this alone made
  queries about 2.4 times faster at equal recall.
* **Cheaper cosine distance,** as for construction.

Queries per second at equal recall@10 (single thread):

=============  =========  ==================  =================  =================
Dataset        recall@10  0.5.13 (queries/s)  0.6.0 (queries/s)  0.7.0 (queries/s)
=============  =========  ==================  =================  =================
MNIST          0.95       19,904              17,341             20,935
MNIST          0.99       12,922              11,436             13,108
GloVe-100      0.75       4,974               4,961              7,061
GloVe-100      0.85       2,416               2,387              3,458
20 Newsgroups  0.85       289                 250                1,066
20 Newsgroups  0.9        208                 188                635
=============  =========  ==================  =================  =================

Sparse data
-----------

Sparse data benefits most from 0.7. Sparse distance functions and random
projection splits used to allocate temporary arrays on every call, and on many
cores threads spent most of their time waiting on the memory allocator: 0.5.13
took longer to build the 20 Newsgroups index with 64 threads than with 8. In
0.7 they allocate nothing, so sparse construction scales with the number of
cores. Preparing a sparse index in 0.6 was slow because the search tree chose
splits that minimised edge cuts with no regard for balance, producing very deep
trees (on 20 Newsgroups, as deep as the 200 level limit allowed); 0.7's balanced
splits keep the tree shallow, which also makes it a much better starting point
for queries. Sparse search trees also no longer pad every hyperplane to the
size of the dataset, which made sparse indexes far larger than necessary.
