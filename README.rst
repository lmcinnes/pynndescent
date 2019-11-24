.. image:: https://travis-ci.org/lmcinnes/pynndescent.svg
    :target: https://travis-ci.org/lmcinnes/pynndescent
    :alt: Travis Build Status
.. image:: https://ci.appveyor.com/api/projects/status/github/lmcinnes/pynndescent?branch=master&svg=true
    :target: https://ci.appveyor.com/project/lmcinnes/pynndescent
    :alt: AppVeyor Build Status
.. image:: https://coveralls.io/repos/github/lmcinnes/pynndescent/badge.svg
    :target: https://coveralls.io/github/lmcinnes/pynndescent
    :alt: Test Coverage Status
.. image:: https://img.shields.io/lgtm/alerts/g/lmcinnes/pynndescent.svg
    :target: https://lgtm.com/projects/g/lmcinnes/pynndescent/alerts
    :alt: LGTM Alerts
.. image:: https://img.shields.io/lgtm/grade/python/g/lmcinnes/pynndescent.svg
    :target: https://lgtm.com/projects/g/lmcinnes/pynndescent/context:python
    :alt: LGTM Grade

===========
PyNNDescent
===========

A Python nearest neighbor descent for approximate nearest neighbors. This is
a relatively straightforward python implementation of Nearest Neighbor
Descent for k-neighbor-graph construction and approximate nearest neighbor
search, as per the paper:

Dong, Wei, Charikar Moses, and Kai Li.
*"Efficient k-nearest neighbor graph construction for generic similarity
measures."*
Proceedings of the 20th international conference on World wide web. ACM, 2011.

This library supplements that approach with the use of random projection
trees for initialisation. This can be particularly useful for the metrics
that are amenable to such approaches (euclidean, minkowski, angular, cosine,
etc.).

Currently this library targets relatively high accuracy 
(90%-99% accuracy rate) approximate nearest neighbor searches.

--------------------
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

While PyNNDescent is not the fastest ANN library, it is both easy to install (pip installable)
with no platform or compilation issues, and very flexible, supporting a wide variety of
distance metrics by default:

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
for the upcoming KNeighborTransformer as a drop in replacement for algorithms
that make use of nearest neighbor computations.

----------------------
How to use PyNNDescent
----------------------

PyNNDescent aims to have a very simple interface. It is similar to (but more
limited than) KDTrees and BallTrees in ``sklearn``. In practice there are
only two operations -- index construction, and querying an index for nearest
neighbors.

To build a new search index on some training data ``data`` you can do something
like

.. code:: python

    from pynndescent import NNDescent
    index = NNDescent(data)

You can then use the index for searching (and can pickle it to disk if you
wish). To search a pynndescent index for the 15 nearest neighbors of a test data
set ``query_data`` you can do something like

.. code:: python

    index.query(query_data, k=15)

and that is pretty much all there is to it.

----------
Installing
----------

PyNNDescent is designed to be easy to install being a pure python module with
relatively light requirements:

* numpy
* scipy
* scikit-learn >= 0.18
* numba >= 0.37

all of which should be pip installable. The easiest way to install should be

.. code:: bash

    pip install pynndescent

To manually install this package:

.. code:: bash

    wget https://github.com/lmcinnes/pynndescent/archive/master.zip
    unzip master.zip
    rm master.zip
    cd pynndescent-master
    python setup.py install

----------------
Help and Support
----------------

This project is still very young. I am currently trying to get example
notebooks and documentation prepared, but it may be a while before those are
available. In the meantime please `open an issue <https://github.com/lmcinnes/pynndescent/issues/new>`_
and I will try to provide any help and guidance that I can. Please also check
the docstrings on the code, which provide some descriptions of the parameters.

-------
License
-------

The pynndescent package is 2-clause BSD licensed. Enjoy.

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. To contribute please `fork the project <https://github.com/lmcinnes/pynndescent/issues#fork-destination-box>`_ make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.


