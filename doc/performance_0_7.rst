Performance Improvements in PyNNDescent 0.7
===========================================

The big theme for PyNNDescent 0.7 is speed. Pretty much every stage of using
PyNNDescent has gotten faster: building the approximate k-nearest neighbor
graph when you create an ``NNDescent`` object, preparing that index for
searching with ``prepare()``, and actually querying it with ``query()``. This
page walks through what changed in each of those stages, and shows some
benchmarks comparing 0.7.0 with the previous releases, 0.6.0 and 0.5.13.

If you just want the headline numbers: compared with 0.6.0, building an index
is 1.5 to 2.4 times faster for dense data and 3 to 11 times faster for sparse
data, preparing an index is 5 to 9 times faster for dense data and 24 to 30
times faster for sparse data, and queries are about 1.4 times faster on GloVe
and 3 to 4 times faster on sparse text data, all at the same (or slightly
better) accuracy. Compared with 0.5.13 the differences are larger still, with
index construction and preparation up to 87 and 79 times faster respectively.
The one place the story is less dramatic is querying MNIST, where 0.7 makes up
for a slowdown in 0.6, but is only on par with (or slightly faster than)
0.5.13.

The gains are biggest on machines with lots of cores. Much more of the work is
now done in parallel, so 0.7 keeps getting faster as you give it more threads.
That wasn't always true of earlier versions: on all three of the datasets
below, 0.5.13 was actually *slower* building an index with 64 threads than with
8.

How we benchmarked
------------------

We used three datasets: MNIST (60,000 points in 784 dimensions, with euclidean
distance), GloVe-100 (1,183,514 word vectors in 100 dimensions, with cosine
distance), and TF-IDF vectors of the 18,846 documents in the 20 Newsgroups
dataset (built with scikit-learn's ``TfidfVectorizer(min_df=2)``, giving
75,875 sparse features, with cosine distance). For MNIST and GloVe-100 we used
the standard query sets (10,000 points each) and ground truth from
`ann-benchmarks <https://github.com/erikbern/ann-benchmarks>`_. For 20
Newsgroups we held out the last 1,000 documents as queries and computed the
true nearest neighbors exactly.

Everything was run with the library defaults and ``n_neighbors=30``, on a
dual-socket Intel Xeon Platinum 8370C machine (2 x 16 cores, 64 hardware
threads). We ran every benchmark twice: once with all 64 threads, and once with
8 threads pinned to 8 physical cores, to get a feel for both a large server and
a more typical workstation. All three versions ran in the same Python 3.12
environment, with numba 0.62.1 and numpy 2.3.5, so the only thing that differs
is PyNNDescent itself.

Since PyNNDescent compiles a lot of code just-in-time with numba, we warmed up
each version on a smaller index first, so the timings below reflect the
steady-state cost of the work rather than compilation. Construction and
preparation times are the median of several runs. Query throughput is the best
of three runs over the full query set, averaged over the 8 and 64 thread runs.
Queries were run as a single batch on a single thread, which is the library's
default.

Index construction
------------------

.. image:: perf07_construction.png
    :alt: Speedup of index construction in 0.7.0 over 0.6.0 and 0.5.13

Building an index happens in two phases. First PyNNDescent builds a forest of
random projection trees; points that end up in the same leaf of a tree are
likely to be near each other, so the leaves give every point a decent set of
initial candidate neighbors. Then nearest neighbor descent refines that
initial graph, repeatedly checking whether the neighbors of a point's neighbors
are closer than the neighbors it already has.

In earlier versions the random projection forest was built one tree per
thread, so it could only ever use as many cores as there were trees (typically
6 to 12), and each tree was built depth-first by a single thread. In 0.7 we
build all the trees together, one level at a time: for each level we work out,
in a single parallel pass over the data, which side of its node's hyperplane
every point falls on (comparing each point against several trees' hyperplanes
at once), and then split all the nodes in parallel. Once nodes get small enough
they are finished off depth-first, again in parallel across all the trees.
Since nearest neighbor descent only needs the leaves, we don't bother keeping
the hyperplanes at all.

Nearest neighbor descent itself got a number of improvements. The core
operation is computing the distances between candidate pairs of points, and
for squared euclidean and dot-product distances we now compare one point
against four candidates at a time, so the first point only needs to be loaded
from memory once. Building the lists of candidates and applying updates to the
graph are now split up by which thread "owns" each point, so threads no longer
need to scan the whole graph, or each other's updates, to find their work.
Finally, for dense data with the ``cosine`` metric, we now normalise the data
once when the index is built. Cosine distance then becomes a simple dot
product, and gets to use the same fast batched distance computations.

Sparse data gets an extra boost. Sparse distance functions used to build
temporary vectors every time they were called, and the sparse random projection
splits allocated memory for every point. On a machine with lots of cores the
threads spent most of their time waiting on each other for the memory
allocator. None of that allocation happens any more, which made building the
initialisation forest for sparse data as much as a hundred times faster on many
cores.

Here are the construction times we measured (the median of three runs, or two
for 0.5.13 on GloVe-100), along with the graph recall: the fraction of each
point's true 30 nearest neighbors that the index found.

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

Before you can search an index it needs to be prepared (this happens
automatically the first time you query, or you can call ``prepare()``
yourself). Preparation builds two things: a search tree, which gives each
query a good set of points to start searching from, and a search graph, which
is the k-nearest neighbor graph with redundant edges pruned away
("diversified"), made undirected, and with the number of neighbors of any one
point capped. It also sets up the function that does the actual searching.

The search tree is "graph-informed": it chooses where to split by drawing
hyperplanes between highly connected "hub" points in the nearest neighbor
graph. It used to be built by a single-threaded recursion. In 0.7 it is built
level by level in parallel, in the same way as the initialisation forest, and
written straight into the flat format used for searching. We also made sure
splits always keep at least 10% of a node's points on each side, which keeps
the tree shallow (and, as we'll see below, also makes queries faster).

Building the search graph used to involve a chain of single-threaded scipy
sparse matrix operations. Diversification now works in place, without
allocating any memory, and making the graph undirected, pruning it, and
reordering it to match the search tree are all done with parallel code
instead.

Finally, earlier versions compiled a brand new search function for every
index, specialised to that index's data. That typically took a second or two
every time you prepared an index. For dense data with euclidean, cosine, or
dot metrics, 0.7 instead uses a search function that is compiled once and
cached on disk, and that is just as fast as the specialised version. If you
want the old behaviour you can still get it with
``prepare(compile_search=True)``.

Here are the times we measured for ``prepare()``:

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

Comparing query speed is a little more subtle than comparing build times,
because there is a trade-off between speed and accuracy. The ``epsilon``
parameter to ``query()`` controls how hard the search works: larger values
search more of the graph, and find more of the true nearest neighbors, but take
longer. To make a fair comparison we ran each version at a range of
``epsilon`` values, and compared the number of queries per second each version
manages at the same recall@10 (the fraction of the true 10 nearest neighbors
found). The full curves of speed against accuracy look like this:

.. image:: perf07_query_curves.png
    :alt: Queries per second against recall@10 for 0.5.13, 0.6.0 and 0.7.0

Some of the query speedup comes from the same places as for index
construction: searching the graph now computes distances to four neighbors at
a time, and cosine distance on dense data is now just a dot product.

The rest comes from better search trees. Each query starts its search from the
points in the leaf of the search tree that it lands in. Previously, a node of
the tree that couldn't be split evenly simply became a leaf, however large it
was, and on some data a handful of leaves ended up holding thousands of points,
all of which had to be checked by any query that landed there. In 0.7 such
nodes are split at the median instead, so leaves stay small. On MNIST with
cosine distance that change alone made queries about 2.4 times faster at the
same accuracy.

Here are the numbers of queries per second (on a single thread) at equal
recall@10:

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

As you may have noticed, sparse data is where 0.7 really shines. Some of that
is the end of memory allocation in the sparse distance functions and random
projection splits: with many threads, 0.5.13 spent so much time waiting on the
memory allocator that it took longer to build the 20 Newsgroups index with 64
threads than with 8, while 0.7 just keeps getting faster as you add cores.

Preparing a sparse index in 0.6 was slow because, for sparse data, the search
tree chose its splits to cut as few nearest neighbor graph edges as possible,
with no regard for balance. That produced extremely lopsided trees: on 20
Newsgroups the tree went as deep as the 200 level limit allowed. The balanced
splits in 0.7 keep the tree shallow, which makes it both quick to build and a
much better starting point for queries. Sparse search trees also no longer pad
every hyperplane out to the size of the dataset, which used to make sparse
indexes far larger than they needed to be.

As always, if you run into any problems, or find cases where 0.7 is slower
than earlier versions, please
`open an issue <https://github.com/lmcinnes/pynndescent/issues/new>`_ and let us
know.
