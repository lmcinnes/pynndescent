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
