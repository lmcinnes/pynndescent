.. pynndescent documentation master file, created by
   sphinx-quickstart on Sat Sep 12 12:01:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyNNDescent for fast Approximate Nearest Neighbors
==================================================

PyNNDescent is a Python nearest neighbor descent for approximate nearest neighbors.
It provides a python implementation of Nearest Neighbor
Descent for k-neighbor-graph construction and approximate nearest neighbor
search, as per the paper:

Dong, Wei, Charikar Moses, and Kai Li.
*"Efficient k-nearest neighbor graph construction for generic similarity
measures."*
Proceedings of the 20th international conference on World wide web. ACM, 2011.

This library supplements that approach with the use of random projection trees for
initialisation. This can be particularly useful for the metrics that are
amenable to such approaches (euclidean, minkowski, angular, cosine, etc.). Graph
diversification is also performed, pruning the longest edges of any triangles in the
graph.

Currently this library targets relatively high accuracy
(80%-100% accuracy rate) approximate nearest neighbor searches.

Why use PyNNDescent?
--------------------

PyNNDescent provides fast approximate nearest neighbor queries. The
`ann-benchmarks <https://github.com/erikbern/ann-benchmarks>`_ system puts it
solidly in the mix of top performing ANN libraries:

**GIST-960 Euclidean**

.. image:: https://camo.githubusercontent.com/142a48c992ba689b8ea9e62636b5281a97322f74/68747470733a2f2f7261772e6769746875622e636f6d2f6572696b6265726e2f616e6e2d62656e63686d61726b732f6d61737465722f726573756c74732f676973742d3936302d6575636c696465616e2e706e67
    :alt: ANN benchmark performance for GIST 960 dataset

**NYTimes-256 Angular**

.. image:: https://camo.githubusercontent.com/6120a35a9db64104eaa1c95cb4803c2fc4cd2679/68747470733a2f2f7261772e6769746875622e636f6d2f6572696b6265726e2f616e6e2d62656e63686d61726b732f6d61737465722f726573756c74732f6e7974696d65732d3235362d616e67756c61722e706e67
    :alt: ANN benchmark performance for NYTimes 256 dataset

While PyNNDescent is among fastest ANN library, it is also both easy to install (pip
and conda installable) with no platform or compilation issues, and is very flexible,
supporting a wide variety of distance metrics by default:

**Minkowski style metrics**

- euclidean
- manhattan
- chebyshev
- minkowski

**Miscellaneous spatial metrics**

- canberra
- braycurtis
- haversine

**Normalized spatial metrics**

- mahalanobis
- wminkowski
- seuclidean

**Angular and correlation metrics**

- cosine
- correlation
- spearmanr

**Probability metrics

- hellinger
- wasserstein

**Metrics for binary data**

- hamming
- jaccard
- dice
- russelrao
- kulsinski
- rogerstanimoto
- sokalmichener
- sokalsneath
- yule

and also custom user defined distance metrics while still retaining performance.

PyNNDescent also integrates well with Scikit-learn, including providing support
for the KNeighborTransformer as a drop in replacement for algorithms
that make use of nearest neighbor computations.

Installing
----------

PyNNDescent is designed to be easy to install being a pure python module with
relatively light requirements:

* numpy
* scipy
* scikit-learn >= 0.22
* numba >= 0.51

all of which should be pip or conda installable. The easiest way to install should be
via conda:

.. code:: bash

    conda install -c conda-forge pynndescent

or via pip:

.. code:: bash

    pip install pynndescent


.. toctree::
   :maxdepth: 2
   :caption: User Guide / Tutorial:

   how_to_use_pynndescent
   pynndescent_metrics
   sparse_data_with_pynndescent
   pynndescent_in_pipelines

.. toctree::
   :caption: API Reference:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
