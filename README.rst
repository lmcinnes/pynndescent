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


