Faster Searching with Quantization and Query Adaptation
=======================================================

How can we make queries faster without giving up much accuracy? And what can
we do when the queries we need to answer don't look much like the data we
indexed? PyNNDescent 0.8 adds a number of new options aimed at exactly these
questions. In this guide we'll walk through each of them: quantized search,
which answers queries using compact codes in place of the full data; maximum
inner product search, for when the size of a vector matters as well as its
direction; adapting an index to the queries it will actually see; and a faster
way of building indexes for high dimensional data. For each we'll look at what
it does, why it helps, and how to use it.

All of these options work with dense data. Sparse data is indexed and searched
just as before.

To have something to work with, let's make some random data to index, and some
queries to search it with.

.. code:: python

    import numpy as np
    from pynndescent import NNDescent

    rng = np.random.default_rng(42)
    data = rng.normal(size=(20000, 256)).astype(np.float32)
    queries = rng.normal(size=(1000, 256)).astype(np.float32)

Quantized search
----------------

Searching an index involves computing distances between a query and a great
many candidate points. For high dimensional data most of the time spent doing
that isn't the arithmetic, it's waiting for each candidate's vector to arrive
from memory. Quantization tackles this by storing a compact code for every
point, using 8 or even 4 bits per feature instead of the 32 bits of a float.
The search then works with distances computed from the codes. These are only
approximate, so once the search has found its best candidates we *rerank*
them, recomputing their distances with the full precision data. The neighbors
and distances you get back are therefore exact for the points returned; the
approximation only affects which points the search manages to find.

To use quantization you simply pass the ``quantization`` parameter when you
create an index.

.. code:: python

    index = NNDescent(data, metric="euclidean", quantization="sq8")
    indices, distances = index.query(queries, k=10)

The ``"sq8"`` and ``"sq4"`` options use per-feature scalar quantization to 8
and 4 bits respectively (with two 4-bit codes packed into each byte), and work
with Euclidean, cosine and dot distances, as well as ``mips`` (more on that
below). The query itself is not quantized; instead it is scored directly
against the codes using integer dot products, which on CPUs with AVX-512 VNNI
instructions can process 64 features in a single instruction. The best
candidates are then reranked within the same compiled function. This is the
fastest search PyNNDescent offers, and it's the one we suggest trying first.

There are some other quantization options as well. ``"int8"`` uses signed
8-bit codes with a scale for each row, and works with cosine, direct cosine
and dot distances. ``"binary"`` keeps just the sign of each feature, and
``"uint8"`` and ``"uint4"`` use codes based on the quantiles of the data; the
``NNDescent`` documentation lists which metrics each of these supports.

The data is quantized when the index is prepared, either by calling
``prepare()`` or with the first query. It's worth being clear that the full
precision data is still kept, since it's needed for reranking, so quantization
makes queries faster but doesn't reduce the memory an index uses.

Trading speed for accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~

For an ordinary index the ``epsilon`` parameter to ``query()`` controls how
hard the search works. With ``"sq8"`` and ``"sq4"`` quantization the most
useful control is instead ``candidate_width``: the number of candidates the
search keeps track of, and then reranks. You can think of it as the width of
the search's beam. It defaults to twice ``k``, and larger values find more of
the true nearest neighbors at the cost of more work.

.. code:: python

    for width in (20, 40, 80):
        indices, distances = index.query(queries, k=10, candidate_width=width)

If reranking all those candidates is more work than you need you can also set
``rerank_width``, which can be anywhere from ``k`` up to ``candidate_width``,
to rerank only the best of them. Since 4-bit codes are coarser than 8-bit ones,
``"sq4"`` will usually need a larger ``candidate_width`` than ``"sq8"`` to
reach the same accuracy, but in return each step of the search is cheaper.

Reranking in half precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the search itself is fast, reranking can become the expensive part,
especially for high dimensional data, since it reads the full vector of every
candidate it reranks. Setting ``rerank_precision="float16"`` keeps an extra
half precision copy of the data (2 bytes per feature) and reranks from that,
halving the amount of memory that reranking has to read. The distances you get
back are then accurate to about three significant digits, which is usually
more than enough to put the neighbors in the right order.

.. code:: python

    index = NNDescent(data, quantization="sq8", rerank_precision="float16")

Searching fewer dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Many high dimensional datasets, text embeddings being a common example, have
most of their variance concentrated in far fewer directions than they have
features. For data like this we can search using codes of the data projected
onto its top principal components, rather than codes of every feature, which
makes each step of the search cheaper still. Reranking still uses every
feature, so the final distances are unaffected. The ``quantization_dims``
parameter controls this: an integer gives the number of components to keep,
while a float between 0 and 1 keeps however many components are needed to
explain that fraction of the variance.

.. code:: python

    index = NNDescent(data, quantization="sq8", quantization_dims=0.9)

Since the search distances are now a little less exact, you may need a larger
``candidate_width`` to reach the same accuracy. If the projection would keep
more than two thirds of the features it isn't worth it, and no projection is
used.

Maximum inner product search
----------------------------

Both the ``dot`` and ``cosine`` metrics normalize the data, so they find the
points whose direction is closest to the query's, regardless of how long the
vectors are. Sometimes, though, the length matters. Many recommendation
models, for example, score items by the inner product of a user vector and an
item vector, and we want the items with the *largest* inner product, with
longer item vectors scoring higher. This is maximum inner product search, and
PyNNDescent supports it with ``metric="mips"``.

.. code:: python

    index = NNDescent(data, metric="mips")
    indices, distances = index.query(queries, k=10, epsilon=0.02)

Inner products don't behave like a distance (a point need not even be its own
nearest neighbor), so a ``mips`` index uses a well known trick. If ``M`` is the
largest norm of any point in the data, we append an extra coordinate,
``sqrt(M**2 - |x|**2)``, to every point, so that they all end up with the same
norm. If we pad each query with a zero, Euclidean distance on this augmented
data orders points in exactly the same way as their inner product with the
query. Under the hood a ``mips`` index is then just a Euclidean index of the
augmented data. The distances returned by ``query()`` are the negated inner
products, so that, as usual, smaller is better.

There is one catch. All of the augmented distances share a large constant
term, which makes the differences between them relatively small, and the
search has to be told to be much more careful in turn. ``mips`` searches need
much smaller ``epsilon`` values than other metrics to reach the same accuracy,
typically somewhere between 0.002 and 0.05; the default of 0.1 can end up
close to an exhaustive search. ``mips`` indexes support ``"sq8"`` and
``"sq4"`` quantization too. If you add data to a ``mips`` index with
``update()``, the new points can't have a larger norm than the largest norm
in the data the index was built with.

Adapting the index to the queries
---------------------------------

Approximate nearest neighbor search works best when the queries come from the
same distribution as the indexed data. This isn't always the case. Multimodal
models, for instance, embed text and images in a shared space, but text
queries searching a collection of image embeddings tend to land in quite
different parts of that space from the images themselves. The search graph and
search tree are built to navigate the data, so such searches have to work
considerably harder to find the true nearest neighbors.

If you have a sample of the kinds of queries you expect, you can prepare the
index for them.

.. code:: python

    sample_queries = ...  # e.g. a few tens of thousands of example queries
    index = NNDescent(data)
    index.prepare(query_sample=sample_queries)
    indices, distances = index.query(queries, k=10)

This does two things. First, the search graph gains extra edges among each
sample query's nearest data points, so that searches can move around more
easily in the regions where queries tend to arrive. Second, searches start
from a random projection tree built over the sample queries, rather than over
the data, so they begin close to where the query's neighbors actually are. If
you are also using ``quantization_dims``, the projection is fitted to the
sample queries as well as the data. The result is that queries like the
sample reach a given accuracy with less work. A few tens of thousands of sample
queries or more tends to work best.

Preparing the index this way needs to know the nearest neighbors of each sample
query in the data. If you already have them (ideally 64 or more for each
query) you can pass their indices with ``query_neighbors``; otherwise they are
found by searching the index, which adds to the time ``prepare()`` takes.

The sample has to be given before the search index is built, so you need to
call ``prepare(query_sample=...)`` before you first query the index. This is
supported for dense data with either no quantization or ``"sq8"`` or ``"sq4"``
quantization. To keep the index small the sample isn't kept once it has been
used (it isn't pickled with the index, for example). That means if you
``update()`` the index you should pass the sample to ``update()`` as well;
otherwise the updated index is prepared without it, and you'll get a warning
letting you know.

Building indexes progressively
------------------------------

So far we've looked at making queries faster, but the same ideas can speed up
building an index too. For dense data with many features most of the cost of
building the nearest neighbor graph comes from computing distances in the
early rounds of nearest neighbor descent, and those rounds only need to find
roughly the right neighbors. With a *progressive* build those early rounds use
8-bit codes and integer dot products, while keeping somewhat longer lists of
candidate neighbors so that the true neighbors aren't lost along the way. The
build then switches to exact distances to refine the graph. The final graph
has exact distances, and typically recall that is as good or better.

You don't need to do anything to make use of this. With the default
``progressive_build="auto"`` it is used for Euclidean, cosine and dot
distances on data with 512 or more features, which is where it pays off. It
makes building an index about 25% faster for 784 dimensional data, and about
45% faster at 4096 dimensions. From 1024 features up the random projection
forest used to initialise the graph is also built on a lower dimensional
random projection of the data, which is several times faster with no loss of
graph quality. The progressive build does need about half as much memory again
as the data itself while the index is being built, so if memory is tight, or
you'd simply like the original build, pass ``progressive_build=False``.

Using these options in pipelines
--------------------------------

``PyNNDescentTransformer`` accepts ``quantization``, ``rerank_precision``,
``quantization_dims`` and ``progressive_build`` as well, and passes them on to
the index it builds. Bear in mind that quantization only speeds up the
searches made when you ``transform`` new data. The nearest neighbor graph of
the training data that ``fit_transform`` returns comes straight from building
the index, so it is unaffected (although the progressive build can make it
faster to compute).

That covers the new options in PyNNDescent 0.8. How much each of them helps
depends a great deal on your data, so it's worth trying a few settings and
measuring the speed and accuracy you get. If you find cases where these
options don't behave as you'd expect, please
`open an issue <https://github.com/lmcinnes/pynndescent/issues/new>`_ and let
us know.
