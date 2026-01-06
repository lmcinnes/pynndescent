# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import numba

from pynndescent.optimal_transport import (
    allocate_graph_structures,
    initialize_graph_structures,
    initialize_supply,
    initialize_cost,
    network_simplex_core,
    total_cost,
    ProblemStatus,
    sinkhorn_transport_plan,
)

from numba import types
from numba.extending import intrinsic
from numba.core import cgutils
from llvmlite import ir as llvm_ir

_mock_identity = np.eye(2, dtype=np.float32)
_mock_ones = np.ones(2, dtype=np.float32)
_dummy_cost = np.zeros((2, 2), dtype=np.float64)

FLOAT32_EPS = np.finfo(np.float32).eps
FLOAT32_MAX = np.finfo(np.float32).max


@intrinsic
def popcnt_u8(typingctx, val):
    """Hardware popcount for uint8 using LLVM intrinsic."""
    sig = types.uint8(types.uint8)

    def popcnt_u8_impl(context, builder, sig, args):
        [val] = args
        # Declare LLVM's ctpop intrinsic for i8
        llvm_i8 = val.type
        fnty = llvm_ir.FunctionType(llvm_i8, [llvm_i8])
        llvm_ctpop = cgutils.get_or_insert_function(
            builder.module, fnty, "llvm.ctpop.i8"
        )
        result = builder.call(llvm_ctpop, [val])
        return result

    return sig, popcnt_u8_impl


@numba.njit(fastmath=True)
def euclidean(x, y):
    r"""Standard euclidean distance.

    .. math::
        D(x, y) = \\sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def squared_euclidean(x, y):
    r"""Squared euclidean distance.

    .. math::
        D(x, y) = \sum_i (x_i - y_i)^2
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(fastmath=True)
def standardised_euclidean(x, y, sigma=_mock_ones):
    r"""Euclidean distance standardised against a vector of standard
    deviations per coordinate.

    .. math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += ((x[i] - y[i]) ** 2) / sigma[i]

    return np.sqrt(result)


@numba.njit(fastmath=True)
def manhattan(x, y):
    r"""Manhattan, taxicab, or l1 distance.

    .. math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])

    return result


@numba.njit(fastmath=True)
def chebyshev(x, y):
    r"""Chebyshev or l-infinity distance.

    .. math::
        D(x, y) = \max_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result = max(result, np.abs(x[i] - y[i]))

    return result


@numba.njit(fastmath=True)
def minkowski(x, y, p=2):
    r"""Minkowski distance.

    .. math::
        D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    This is a general distance. For p=1 it is equivalent to
    manhattan distance, for p=2 it is Euclidean distance, and
    for p=infinity it is Chebyshev distance. In general it is better
    to use the more specialised functions for those distances.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit(fastmath=True)
def weighted_minkowski(x, y, w=_mock_ones, p=2):
    r"""A weighted version of Minkowski distance.

    .. math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    If weights w_i are inverse standard deviations of graph_data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += w[i] * np.abs(x[i] - y[i]) ** p

    return result ** (1.0 / p)


@numba.njit(fastmath=True)
def mahalanobis(x, y, vinv=_mock_identity):
    r"""Mahalanobis distance.

    .. math::
        D(x, y) = \sqrt{(x - y)^T V^{-1} (x - y)}

    where V is the covariance matrix. This is equivalent to Euclidean distance
    after transforming the space by the inverse square root of the covariance.
    """
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float32)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += tmp * diff[i]

    return np.sqrt(result)


@numba.njit(fastmath=True)
def hamming(x, y):
    r"""Hamming distance.

    The proportion of elements that differ between two vectors.

    .. math::
        D(x, y) = \frac{1}{n} \sum_i \mathbf{1}_{x_i \neq y_i}
    """
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            result += 1.0

    return float(result) / x.shape[0]


@numba.njit(fastmath=True)
def canberra(x, y):
    r"""Canberra distance.

    A weighted version of Manhattan distance where each term is divided
    by the sum of absolute values.

    .. math::
        D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
    """
    result = 0.0
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator

    return result


@numba.njit(fastmath=True)
def bray_curtis(x, y):
    r"""Bray-Curtis distance.

    A distance measure commonly used in ecology to quantify the compositional
    dissimilarity between two samples.

    .. math::
        D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i |x_i + y_i|}
    """
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        return float(numerator) / denominator
    else:
        return 0.0


@numba.njit(fastmath=True)
def jaccard(x, y):
    r"""Jaccard distance.

    One minus the Jaccard similarity coefficient. For binary vectors this is
    the size of the symmetric difference divided by the size of the union.

    .. math::
        D(x, y) = 1 - \frac{|x \cap y|}{|x \cup y|}

    For continuous vectors, non-zero values are treated as set membership.
    """
    num_non_zero = 0.0
    num_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "num_non_zero": numba.types.float32,
        "num_equal": numba.types.float32,
        "x_true": numba.types.uint8,
        "y_true": numba.types.uint8,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_jaccard(x, y):
    r"""Alternative Jaccard distance using log transform.

    A transformed version of Jaccard distance suitable for the bounded-radius
    search algorithm. Uses negative log of the Jaccard similarity coefficient.

    .. math::
        D_{alt}(x, y) = -\log_2\left(\frac{|x \cap y|}{|x \cup y|}\right)

    Use `correct_alternative_jaccard` to convert back to standard Jaccard distance.
    """
    num_non_zero = 0.0
    num_equal = 0.0
    dim = x.shape[0]
    for i in range(dim):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return -np.log2(num_equal / num_non_zero)


@numba.vectorize(fastmath=True)
def correct_alternative_jaccard(v):
    r"""Convert alternative Jaccard distance back to standard Jaccard distance.

    .. math::
        D(x, y) = 1 - 2^{-D_{alt}(x, y)}
    """
    return 1.0 - pow(2.0, -v)


@numba.njit(fastmath=True)
def matching(x, y):
    r"""Matching distance (simple matching dissimilarity).

    The proportion of elements that differ in their boolean state.
    For binary vectors, counts positions where one is non-zero and
    the other is zero.

    .. math::
        D(x, y) = \frac{1}{n} \sum_i \mathbf{1}_{(x_i \neq 0) \neq (y_i \neq 0)}
    """
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return float(num_not_equal) / x.shape[0]


@numba.njit(fastmath=True)
def dice(x, y):
    r"""Dice distance (Sørensen-Dice dissimilarity).

    One minus twice the intersection divided by the sum of cardinalities.
    Commonly used for comparing the similarity of two samples.

    .. math::
        D(x, y) = \frac{|x \oplus y|}{2|x \cap y| + |x \oplus y|}

    where :math:`\oplus` denotes symmetric difference.
    """
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit(fastmath=True)
def kulsinski(x, y):
    r"""Kulsinski distance.

    A variant of Jaccard distance that includes a count of all dimensions.
    For binary vectors, gives more weight to dimensions where both are false.

    .. math::
        D(x, y) = \frac{|x \oplus y| - |x \cap y| + n}{|x \oplus y| + n}

    where n is the number of dimensions.
    """
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + x.shape[0]) / (
            num_not_equal + x.shape[0]
        )


@numba.njit(fastmath=True)
def rogers_tanimoto(x, y):
    r"""Rogers-Tanimoto distance.

    A distance measure for binary vectors that gives double weight to
    disagreements.

    .. math::
        D(x, y) = \frac{2|x \oplus y|}{n + |x \oplus y|}

    where n is the number of dimensions.
    """
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit(fastmath=True)
def russellrao(x, y):
    r"""Russell-Rao distance.

    The proportion of dimensions where at least one vector has a false value.

    .. math::
        D(x, y) = \frac{n - |x \cap y|}{n}

    where n is the number of dimensions.
    """
    num_true_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true

    if num_true_true == np.sum(x != 0) and num_true_true == np.sum(y != 0):
        return 0.0
    else:
        return float(x.shape[0] - num_true_true) / (x.shape[0])


@numba.njit(fastmath=True)
def sokal_michener(x, y):
    r"""Sokal-Michener distance.

    Equivalent to Rogers-Tanimoto distance. A distance measure for binary
    vectors that gives double weight to disagreements.

    .. math::
        D(x, y) = \frac{2|x \oplus y|}{n + |x \oplus y|}

    where n is the number of dimensions.
    """
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit(fastmath=True)
def sokal_sneath(x, y):
    r"""Sokal-Sneath distance.

    A binary distance that gives double weight to agreements (both true).

    .. math::
        D(x, y) = \frac{|x \oplus y|}{0.5|x \cap y| + |x \oplus y|}

    where :math:`\oplus` denotes symmetric difference.
    """
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit(fastmath=True)
def haversine(x, y):
    r"""Haversine (great circle) distance.

    The angular distance between two points on a sphere, given their
    latitudes and longitudes in radians. Only valid for 2D data where
    x[0], y[0] are latitudes and x[1], y[1] are longitudes.

    .. math::
        D(x, y) = 2 \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_1 - \phi_2}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_1 - \lambda_2}{2}\right)}\right)

    where :math:`\phi` is latitude and :math:`\lambda` is longitude.
    """
    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional graph_data")
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    sin_long = np.sin(0.5 * (x[1] - y[1]))
    result = np.sqrt(sin_lat**2 + np.cos(x[0]) * np.cos(y[0]) * sin_long**2)
    return 2.0 * np.arcsin(result)


@numba.njit(fastmath=True)
def yule(x, y):
    r"""Yule distance.

    A binary distance based on the Yule Q coefficient of association.

    .. math::
        D(x, y) = \frac{2 \cdot n_{TF} \cdot n_{FT}}{n_{TT} \cdot n_{FF} + n_{TF} \cdot n_{FT}}

    where :math:`n_{TF}` is the count of positions where x is true and y is false, etc.
    """
    num_true_true = 0.0
    num_true_false = 0.0
    num_false_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_true_false += x_true and (not y_true)
        num_false_true += (not x_true) and y_true

    num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

    if num_true_false == 0.0 or num_false_true == 0.0:
        return 0.0
    else:
        return (2.0 * num_true_false * num_false_true) / (
            num_true_true * num_false_false + num_true_false * num_false_true
        )


@numba.njit(fastmath=True)
def cosine(x, y):
    r"""Cosine distance.

    One minus the cosine of the angle between two vectors. Measures the
    angular difference between vectors, independent of their magnitudes.

    .. math::
        D(x, y) = 1 - \frac{\langle x, y \rangle}{\|x\| \|y\|}

    Returns 0 if both vectors are zero, 1 if one is zero.
    """
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        return 1.0 - (result / np.sqrt(norm_x * norm_y))


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "norm_x": numba.types.float32,
        "norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_cosine(x, y):
    r"""Alternative cosine distance using log transform.

    A transformed version of cosine distance suitable for the bounded-radius
    search algorithm. Uses negative log of the cosine similarity.

    .. math::
        D_{alt}(x, y) = \log_2\left(\frac{\|x\| \|y\|}{\langle x, y \rangle}\right)

    Returns FLOAT32_MAX for non-positive cosine similarities (treating them
    as infinitely far). Use `correct_alternative_cosine` to convert back
    to standard cosine distance.
    """
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return FLOAT32_MAX
    elif result <= 0.0:
        return FLOAT32_MAX
    else:
        result = np.sqrt(norm_x * norm_y) / result
        return np.log2(result)


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def dot(x, y):
    r"""Dot product distance for normalized vectors.

    One minus the dot product. This is equivalent to cosine distance when
    vectors are normalized to unit length. For unnormalized vectors, use
    `inner_product` distance instead.

    .. math::
        D(x, y) = 1 - \langle x, y \rangle

    Returns 1.0 for non-positive dot products.
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    if result <= 0.0:
        return 1.0
    else:
        return 1.0 - result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_dot(x, y):
    r"""Alternative dot product distance using log transform.

    A transformed version of dot product distance suitable for the bounded-radius
    search algorithm. Uses negative log of the dot product.

    .. math::
        D_{alt}(x, y) = -\log_2(\langle x, y \rangle)

    Returns FLOAT32_MAX for non-positive dot products (treating them as
    infinitely far). Use `correct_alternative_cosine` to convert back
    to standard dot distance.
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    if result <= 0.0:
        return FLOAT32_MAX
    else:
        return -np.log2(result)


@numba.vectorize(fastmath=True)
def correct_alternative_cosine(d):
    r"""Convert alternative cosine/dot distance back to standard form.

    .. math::
        D(x, y) = 1 - 2^{-D_{alt}(x, y)}
    """
    return 1.0 - pow(2.0, -d)


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def inner_product(x, y):
    r"""Inner product distance (negative inner product).

    This is useful for retrieval tasks where the inner product represents
    similarity (higher = more similar). The distance is simply the negation
    of the inner product, so that higher similarity becomes lower distance.

    Note: Unlike dot product distance, this does NOT assume normalized vectors.
    For normalized vectors, use the `dot` distance instead which is bounded [0, 1].

    .. math::
        D(x, y) = -\sum_i x_i y_i
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    return -result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_inner_product(x, y):
    r"""Alternative inner product distance using reciprocal transform.

    This transforms the inner product into a positive distance suitable for
    the bounded-radius search algorithm. The transform is:

    .. math::
        D_{alt}(x, y) = \frac{1}{\langle x, y \rangle}

    This maps positive inner products to positive distances:
    - High inner product → small positive distance
    - Low positive inner product → large positive distance
    - Non-positive inner product → FLOAT32_MAX (treated as infinitely far)

    In high-dimensional nearest neighbor search, we expect true neighbors
    to have positive inner products. Pairs with non-positive inner products
    are treated as maximally distant, similar to how alternative_cosine
    handles negative cosine similarities.

    The correction function `correct_alternative_inner_product` converts
    back to the negative inner product.
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    if result <= 0.0:
        return FLOAT32_MAX
    else:
        return 1.0 / result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "ip_result": numba.types.float32,
        "norm_x": numba.types.float32,
        "norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_inner_product(x, y):
    r"""A proxy for inner product distance (negative inner product).

    Inner product distance has undesirable properties for nearest neighbor
    graph based search, and NNDescent in general. This is a proxy function
    that behaves similarly to inner product distance for ranking neighbors,
    but avoids some of the pitfalls.

    This is to be used internally, and results should use reranking with true
    inner product distance.
    """
    ip_result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        ip_result += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]

    if norm_x == 0 or norm_y == 0:
        return FLOAT32_MAX

    cosine_result = -np.log2(ip_result / np.sqrt(norm_x * norm_y))
    if ip_result >= 0:
        return cosine_result + 1.0 / np.sqrt(ip_result)
    else:
        return FLOAT32_MAX


@numba.vectorize(fastmath=True)
def correct_alternative_inner_product(d):
    r"""Convert alternative inner product distance back to negative inner product.

    .. math::
        D(x, y) = -\langle x, y \rangle = -\frac{1}{D_{alt}(x, y)}

    For d = FLOAT32_MAX (non-positive inner products), returns 0.0 as the
    negative inner product (representing orthogonal or dissimilar vectors).
    """
    if d >= FLOAT32_MAX:
        return 0.0
    return -1.0 / d


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "cdf_x": numba.types.float32,
        "cdf_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_wasserstein_1d(x, y):
    r"""A proxy for 1D Wasserstein distance.

    Uses L1 distance on the cumulative distribution functions, which is
    exactly equal to Wasserstein-1 distance for 1D distributions. This
    avoids the more expensive Minkowski computation with allocation.

    For Wasserstein-p with p > 1, this is a lower bound and correlates
    well for nearest neighbor search.

    .. math::
        D_{proxy}(x, y) = \sum_i |F_x(i) - F_y(i)|

    where :math:`F_x, F_y` are the cumulative distribution functions.

    Results should be reranked with true wasserstein_1d distance if p > 1.
    """
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0.0 or l1_norm_y == 0.0:
        return FLOAT32_MAX

    # Compute CDF difference inline to avoid allocation
    cdf_x = 0.0
    cdf_y = 0.0
    result = 0.0

    for i in range(dim):
        cdf_x += x[i] / l1_norm_x
        cdf_y += y[i] / l1_norm_y
        result += np.abs(cdf_x - cdf_y)

    return result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "tv_result": numba.types.float32,
        "hellinger_result": numba.types.float32,
        "px": numba.types.float32,
        "py": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_kantorovich(x, y):
    r"""A proxy for Kantorovich (Earth Mover's) distance.

    The full Kantorovich distance requires solving an optimal transport
    problem via network simplex, which is expensive. This proxy uses a
    combination of:
    1. Total variation distance (L1 on normalized distributions)
    2. Hellinger-like term for better correlation

    This is much cheaper to compute and correlates reasonably well with
    true optimal transport distance for nearest neighbor search.

    Results should be reranked with true kantorovich distance.
    """
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0.0 or l1_norm_y == 0.0:
        return FLOAT32_MAX

    # Total variation distance + Hellinger-like term
    tv_result = 0.0
    hellinger_result = 0.0

    for i in range(dim):
        px = x[i] / l1_norm_x
        py = y[i] / l1_norm_y
        tv_result += np.abs(px - py)
        hellinger_result += np.sqrt(px * py)

    # Combine: TV captures mass difference, Hellinger captures shape similarity
    return 0.5 * tv_result + (1.0 - hellinger_result)


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "cdf_x": numba.types.float32,
        "cdf_y": numba.types.float32,
        "mu": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_circular_kantorovich(x, y):
    r"""A proxy for circular Kantorovich distance.

    Uses mean-shifted CDF L1 distance instead of the more expensive
    median-shifted Minkowski distance. The mean is a reasonable
    approximation to the median for most distributions and avoids
    the expensive median computation.

    Results should be reranked with true circular_kantorovich distance.
    """
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0.0 or l1_norm_y == 0.0:
        return FLOAT32_MAX

    # Compute CDF differences and their mean
    cdf_x = 0.0
    cdf_y = 0.0
    mu = 0.0

    for i in range(dim):
        cdf_x += x[i] / l1_norm_x
        cdf_y += y[i] / l1_norm_y
        mu += cdf_x - cdf_y

    mu /= dim

    # L1 on shifted CDFs
    cdf_x = 0.0
    cdf_y = 0.0
    result = 0.0

    for i in range(dim):
        cdf_x += x[i] / l1_norm_x
        cdf_y += y[i] / l1_norm_y
        result += np.abs(cdf_x - cdf_y - mu)

    return result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "bc": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_jensen_shannon(x, y):
    r"""A proxy for Jensen-Shannon divergence.

    Jensen-Shannon requires computing logs and the mixture distribution,
    which is expensive. This proxy uses squared Hellinger distance, which
    is also a proper divergence on probability distributions and much
    cheaper to compute (no logs required).

    .. math::
        D_{proxy}(x, y) = 1 - \left(\sum_i \sqrt{p_i q_i}\right)^2

    where p, q are the normalized distributions.

    Results should be reranked with true jensen_shannon_divergence.
    """
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0.0 or l1_norm_y == 0.0:
        return FLOAT32_MAX

    # Bhattacharyya coefficient
    bc = 0.0
    for i in range(dim):
        bc += np.sqrt((x[i] / l1_norm_x) * (y[i] / l1_norm_y))

    # Squared Hellinger-like distance: 1 - BC^2
    # This spreads values more than standard Hellinger and correlates
    # well with Jensen-Shannon divergence
    return 1.0 - bc * bc


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "px": numba.types.float32,
        "py": numba.types.float32,
        "denom": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_symmetric_kl(x, y):
    r"""A proxy for symmetric KL divergence.

    Symmetric KL requires computing logs which is expensive. This proxy
    uses triangular discrimination (symmetric chi-squared divergence),
    which is a second-order approximation to KL divergence and much cheaper.

    .. math::
        D_{proxy}(x, y) = \sum_i \frac{(p_i - q_i)^2}{p_i + q_i}

    Results should be reranked with true symmetric_kl_divergence.
    """
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0.0 or l1_norm_y == 0.0:
        return FLOAT32_MAX

    # Triangular discrimination / symmetric chi-squared
    result = 0.0
    for i in range(dim):
        px = x[i] / l1_norm_x
        py = y[i] / l1_norm_y
        denom = px + py
        if denom > 0:
            diff = px - py
            result += (diff * diff) / denom

    return result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "tv_result": numba.types.float32,
        "hellinger_result": numba.types.float32,
        "px": numba.types.float32,
        "py": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def proxy_sinkhorn(x, y):
    r"""A proxy for Sinkhorn (entropy-regularized optimal transport) distance.

    Sinkhorn distance requires iterative matrix scaling which is expensive.
    This proxy uses the same combination as proxy_kantorovich since Sinkhorn
    approximates Kantorovich.

    Results should be reranked with true sinkhorn distance.
    """
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0.0 or l1_norm_y == 0.0:
        return FLOAT32_MAX

    # Total variation distance + Hellinger-like term
    tv_result = 0.0
    hellinger_result = 0.0

    for i in range(dim):
        px = x[i] / l1_norm_x
        py = y[i] / l1_norm_y
        tv_result += np.abs(px - py)
        hellinger_result += np.sqrt(px * py)

    return 0.5 * tv_result + (1.0 - hellinger_result)


@numba.njit(fastmath=True)
def tsss(x, y):
    r"""Triangle Area Similarity - Sector Area Similarity (TS-SS) distance.

    A distance metric that combines both magnitude and angular information.
    It multiplies a triangle area (capturing angular difference) by a sector
    area (capturing both angular and magnitude differences).

    Useful when both the direction and magnitude of vectors are important.
    """
    d_euc_squared = 0.0
    d_cos = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        diff = x[i] - y[i]
        d_euc_squared += diff * diff
        d_cos += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]

    norm_x = np.sqrt(norm_x)
    norm_y = np.sqrt(norm_y)
    magnitude_difference = np.abs(norm_x - norm_y)
    d_cos /= norm_x * norm_y
    theta = np.arccos(d_cos) + np.radians(10)  # Add 10 degrees as an "epsilon" to
    # avoid problems
    sector = ((np.sqrt(d_euc_squared) + magnitude_difference) ** 2) * theta
    triangle = norm_x * norm_y * np.sin(theta) / 2.0
    return triangle * sector


@numba.njit(fastmath=True)
def true_angular(x, y):
    r"""True angular distance.

    The actual angle between two vectors, normalized to [0, 1].
    Unlike cosine distance which uses 1 - cos(θ), this returns 1 - θ/π.

    .. math::
        D(x, y) = 1 - \frac{\arccos\left(\frac{\langle x, y \rangle}{\|x\| \|y\|}\right)}{\pi}

    Returns 0 for identical directions, approaches 1 for opposite directions.
    """
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return FLOAT32_MAX
    elif result <= 0.0:
        return FLOAT32_MAX
    else:
        result = result / np.sqrt(norm_x * norm_y)
        return 1.0 - (np.arccos(result) / np.pi)


@numba.vectorize(fastmath=True)
def true_angular_from_alt_cosine(d):
    r"""Convert alternative cosine distance to true angular distance.

    .. math::
        D_{angular}(x, y) = 1 - \frac{\arccos(2^{-D_{alt}})}{\pi}
    """
    return 1.0 - (np.arccos(pow(2.0, -d)) / np.pi)


@numba.njit(fastmath=True)
def correlation(x, y):
    r"""Correlation distance.

    One minus the Pearson correlation coefficient. Measures how linearly
    related two vectors are after centering (subtracting their means).

    .. math::
        D(x, y) = 1 - \frac{\langle x - \bar{x}, y - \bar{y} \rangle}{\|x - \bar{x}\| \|y - \bar{y}\|}

    Equivalent to cosine distance on mean-centered data.
    """
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x**2
        norm_y += shifted_y**2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def hellinger(x, y):
    r"""Hellinger distance.

    A distance for probability distributions, based on the Bhattacharyya
    coefficient. Input vectors are treated as (unnormalized) probability
    distributions.

    .. math::
        D(x, y) = \sqrt{1 - \frac{\sum_i \sqrt{x_i y_i}}{\sqrt{\sum_i x_i \cdot \sum_i y_i}}}

    Returns values in [0, 1].
    """
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        result += np.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return 1.0
    else:
        return np.sqrt(1 - result / np.sqrt(l1_norm_x * l1_norm_y))


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "l1_norm_x": numba.types.float32,
        "l1_norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_hellinger(x, y):
    r"""Alternative Hellinger distance using log transform.

    A transformed version of Hellinger distance suitable for the bounded-radius
    search algorithm.

    .. math::
        D_{alt}(x, y) = \log_2\left(\frac{\sqrt{\sum_i x_i \cdot \sum_i y_i}}{\sum_i \sqrt{x_i y_i}}\right)

    Use `correct_alternative_hellinger` to convert back to standard Hellinger distance.
    """
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        result += np.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return FLOAT32_MAX
    elif result <= 0:
        return FLOAT32_MAX
    else:
        result = np.sqrt(l1_norm_x * l1_norm_y) / result
        return np.log2(result)


@numba.vectorize(fastmath=True)
def correct_alternative_hellinger(d):
    r"""Convert alternative Hellinger distance back to standard Hellinger distance.

    .. math::
        D(x, y) = \sqrt{1 - 2^{-D_{alt}(x, y)}}
    """
    return np.sqrt(1.0 - pow(2.0, -d))


@numba.njit()
def rankdata(a, method="average"):
    arr = np.ravel(np.asarray(a))
    if method == "ordinal":
        sorter = arr.argsort(kind="mergesort")
    else:
        sorter = arr.argsort(kind="quicksort")

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)

    if method == "ordinal":
        return (inv + 1).astype(np.float64)

    arr = arr[sorter]
    obs = np.ones(arr.size, np.bool_)
    obs[1:] = arr[1:] != arr[:-1]
    dense = obs.cumsum()[inv]

    if method == "dense":
        return dense.astype(np.float64)

    # cumulative counts of each unique value
    nonzero = np.nonzero(obs)[0]
    count = np.concatenate((nonzero, np.array([len(obs)], nonzero.dtype)))

    if method == "max":
        return count[dense].astype(np.float64)

    if method == "min":
        return (count[dense - 1] + 1).astype(np.float64)

    # average method
    return 0.5 * (count[dense] + count[dense - 1] + 1)


@numba.njit(fastmath=True)
def spearmanr(x, y):
    r"""Spearman rank correlation distance.

    One minus the Spearman rank correlation coefficient. Measures the monotonic
    relationship between two vectors by computing correlation on their ranks.

    .. math::
        D(x, y) = 1 - \rho(\text{rank}(x), \text{rank}(y))

    where :math:`\rho` is Pearson correlation.
    """
    x_rank = rankdata(x)
    y_rank = rankdata(y)

    return correlation(x_rank, y_rank)


@numba.njit(nogil=True)
def kantorovich(x, y, cost=_dummy_cost, max_iter=100000):
    r"""Kantorovich distance (Earth Mover's Distance / Wasserstein distance).

    The optimal transport distance between two probability distributions.
    Computes the minimum cost to transform one distribution into another,
    given a cost matrix.

    Parameters
    ----------
    x, y : array-like
        Input vectors treated as probability distributions (will be normalized).
    cost : array-like
        Cost matrix where cost[i,j] is the cost of moving mass from bin i to bin j.
    max_iter : int
        Maximum number of iterations for the network simplex algorithm.

    Returns
    -------
    float
        The optimal transport distance.
    """

    row_mask = x != 0
    col_mask = y != 0

    a = x[row_mask].astype(np.float64)
    b = y[col_mask].astype(np.float64)

    a_sum = a.sum()
    b_sum = b.sum()

    # if not isclose(a_sum, b_sum):
    #     raise ValueError(
    #         "Kantorovich distance inputs must be valid probability distributions."
    #     )

    a /= a_sum
    b /= b_sum

    sub_cost = cost[row_mask, :][:, col_mask]

    node_arc_data, spanning_tree, graph = allocate_graph_structures(
        a.shape[0], b.shape[0], False
    )
    initialize_supply(a, -b, graph, node_arc_data.supply)
    initialize_cost(sub_cost, graph, node_arc_data.cost)
    # initialize_cost(cost, graph, node_arc_data.cost)
    init_status = initialize_graph_structures(graph, node_arc_data, spanning_tree)
    if init_status == False:
        raise ValueError(
            "Kantorovich distance inputs must be valid probability distributions."
        )
    solve_status = network_simplex_core(node_arc_data, spanning_tree, graph, max_iter)
    # if solve_status == ProblemStatus.MAX_ITER_REACHED:
    #     print("WARNING: RESULT MIGHT BE INACCURATE\nMax number of iteration reached!")
    if solve_status == ProblemStatus.INFEASIBLE:
        raise ValueError(
            "Optimal transport problem was INFEASIBLE. Please check inputs."
        )
    elif solve_status == ProblemStatus.UNBOUNDED:
        raise ValueError(
            "Optimal transport problem was UNBOUNDED. Please check inputs."
        )
    result = total_cost(node_arc_data.flow, node_arc_data.cost)

    return result


@numba.njit(fastmath=True)
def sinkhorn(x, y, cost=_dummy_cost, regularization=1.0):
    r"""Sinkhorn distance (entropy-regularized optimal transport).

    An approximation to the Kantorovich distance using entropy regularization.
    Faster to compute than exact optimal transport for large distributions.

    Parameters
    ----------
    x, y : array-like
        Input vectors treated as probability distributions (will be normalized).
    cost : array-like
        Cost matrix where cost[i,j] is the cost of moving mass from bin i to bin j.
    regularization : float
        Entropy regularization parameter. Smaller values give results closer
        to exact Kantorovich distance but may be less stable.

    Returns
    -------
    float
        The entropy-regularized optimal transport distance.
    """
    row_mask = x != 0
    col_mask = y != 0

    a = x[row_mask].astype(np.float64)
    b = y[col_mask].astype(np.float64)

    a_sum = a.sum()
    b_sum = b.sum()

    a /= a_sum
    b /= b_sum

    sub_cost = cost[row_mask, :][:, col_mask]

    transport_plan = sinkhorn_transport_plan(
        x, y, cost=sub_cost, regularization=regularization
    )
    dim_i = transport_plan.shape[0]
    dim_j = transport_plan.shape[1]
    result = 0.0
    for i in range(dim_i):
        for j in range(dim_j):
            result += transport_plan[i, j] * cost[i, j]

    return result


@numba.njit()
def jensen_shannon_divergence(x, y):
    r"""Jensen-Shannon divergence.

    A symmetrized and smoothed version of KL divergence. Measures the
    similarity between two probability distributions.

    .. math::
        D(x, y) = \frac{1}{2} \left( D_{KL}(x \| m) + D_{KL}(y \| m) \right)

    where :math:`m = \frac{1}{2}(x + y)` and :math:`D_{KL}` is KL divergence.
    Input vectors are normalized to probability distributions.
    """
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    l1_norm_x += FLOAT32_EPS * dim
    l1_norm_y += FLOAT32_EPS * dim

    pdf_x = (x + FLOAT32_EPS) / l1_norm_x
    pdf_y = (y + FLOAT32_EPS) / l1_norm_y
    m = 0.5 * (pdf_x + pdf_y)

    for i in range(dim):
        result += 0.5 * (
            pdf_x[i] * np.log(pdf_x[i] / m[i]) + pdf_y[i] * np.log(pdf_y[i] / m[i])
        )

    return result


@numba.njit()
def wasserstein_1d(x, y, p=1):
    r"""1-dimensional Wasserstein distance.

    The p-Wasserstein distance for 1D distributions, computed efficiently
    via the CDF. Input vectors are treated as histograms over ordered bins.

    .. math::
        W_p(x, y) = \left( \sum_i |F_x(i) - F_y(i)|^p \right)^{1/p}

    where :math:`F_x, F_y` are the cumulative distribution functions.

    Parameters
    ----------
    x, y : array-like
        Input vectors treated as probability distributions (will be normalized).
    p : int
        The order of the Wasserstein distance (default 1).
    """
    x_sum = 0.0
    y_sum = 0.0
    for i in range(x.shape[0]):
        x_sum += x[i]
        y_sum += y[i]

    x_cdf = x / x_sum
    y_cdf = y / y_sum

    for i in range(1, x_cdf.shape[0]):
        x_cdf[i] += x_cdf[i - 1]
        y_cdf[i] += y_cdf[i - 1]

    return minkowski(x_cdf, y_cdf, p)


@numba.njit()
def circular_kantorovich(x, y, p=1):
    r"""Circular Kantorovich distance.

    The Wasserstein distance for distributions on a circle (periodic domain).
    Useful for cyclic data like angles, time of day, or periodic histograms.

    Parameters
    ----------
    x, y : array-like
        Input vectors treated as probability distributions (will be normalized).
    p : int
        The order of the Wasserstein distance (default 1).
    """
    x_sum = 0.0
    y_sum = 0.0
    for i in range(x.shape[0]):
        x_sum += x[i]
        y_sum += y[i]

    x_cdf = x / x_sum
    y_cdf = y / y_sum

    for i in range(1, x_cdf.shape[0]):
        x_cdf[i] += x_cdf[i - 1]
        y_cdf[i] += y_cdf[i - 1]

    mu = np.median((x_cdf - y_cdf) ** p)

    # Now we just want minkowski distance on the CDFs shifted by mu
    result = 0.0
    if p > 2:
        for i in range(x_cdf.shape[0]):
            result += np.abs(x_cdf[i] - y_cdf[i] - mu) ** p

        return result ** (1.0 / p)

    elif p == 2:
        for i in range(x_cdf.shape[0]):
            val = x_cdf[i] - y_cdf[i] - mu
            result += val * val

        return np.sqrt(result)

    elif p == 1:
        for i in range(x_cdf.shape[0]):
            result += np.abs(x_cdf[i] - y_cdf[i] - mu)

        return result

    else:
        raise ValueError("Invalid p supplied to Kantorvich distance")


@numba.njit()
def symmetric_kl_divergence(x, y):
    r"""Symmetric Kullback-Leibler divergence.

    The sum of KL divergences in both directions, making it symmetric.

    .. math::
        D(x, y) = D_{KL}(x \| y) + D_{KL}(y \| x)

    where :math:`D_{KL}(p \| q) = \sum_i p_i \log(p_i / q_i)`.
    Input vectors are normalized to probability distributions.
    """
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    l1_norm_x += FLOAT32_EPS * dim
    l1_norm_y += FLOAT32_EPS * dim

    pdf_x = (x + FLOAT32_EPS) / l1_norm_x
    pdf_y = (y + FLOAT32_EPS) / l1_norm_y

    for i in range(dim):
        result += pdf_x[i] * np.log(pdf_x[i] / pdf_y[i]) + pdf_y[i] * np.log(
            pdf_y[i] / pdf_x[i]
        )

    return result


@numba.njit(
    [
        "f4(u1[::1],u1[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    nogil=True,
    boundscheck=False,
    locals={
        "result": numba.types.int32,
        "intersection": numba.types.uint8,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def bit_hamming(x, y):
    r"""Hamming distance for bit-packed binary vectors.

    Counts the number of differing bits between two uint8 arrays, where each
    byte contains 8 packed binary features. More efficient than standard
    Hamming for binary data.

    .. math::
        D(x, y) = \sum_i \text{popcount}(x_i \oplus y_i)

    Returns the total count of differing bits (not normalized).
    """
    result = 0
    dim = x.shape[0]

    for i in range(dim):
        intersection = x[i] ^ y[i]
        result += popcnt_u8(intersection)

    return np.float32(result)


@numba.njit(
    [
        "f4(u1[::1],u1[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    nogil=True,
    boundscheck=False,
    locals={
        "result": numba.types.int32,
        "denom": numba.types.int32,
        "and_": numba.types.uint8,
        "or_": numba.types.uint8,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def bit_jaccard(x, y):
    r"""Jaccard distance for bit-packed binary vectors.

    Computes Jaccard distance for uint8 arrays where each byte contains
    8 packed binary features. Uses negative log transform for compatibility
    with the bounded-radius search algorithm.

    .. math::
        D(x, y) = -\log\left(\frac{\text{popcount}(x \land y)}{\text{popcount}(x \lor y)}\right)

    More efficient than standard Jaccard for binary data.
    """
    result = 0
    denom = 0
    dim = x.shape[0]

    for i in range(dim):
        and_ = x[i] & y[i]
        or_ = x[i] | y[i]
        result += popcnt_u8(and_)
        denom += popcnt_u8(or_)

    if denom == 0:
        return 0.0
    else:
        return -np.log(np.float32(result) / np.float32(denom))


@numba.njit(
    [
        "f4(f4[::1],u1[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    nogil=True,
    boundscheck=False,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
        "y_i": numba.types.float32,
    },
)
def quantized_uint8_sq_euclidean(x, y, quantized_values):
    r"""Squared Euclidean distance between a float vector ``x`` and
    a quantized uint8 vector ``y``. The uint8 values in ``y`` are mapped
    back to floats using the provided ``quantized_values`` array.
    """
    result = 0.0
    dim = x.shape[0]

    for i in range(dim):
        y_i = quantized_values[y[i]]
        diff = x[i] - y_i
        result += diff * diff

    return result


@numba.njit(
    [
        "f4(f4[::1],u1[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    nogil=True,
    boundscheck=False,
    locals={
        "result": numba.types.float32,
        "norm_x": numba.types.float32,
        "norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def quantized_uint8_alternative_cosine(x, y, quantized_values):
    r"""Alternative cosine distance between a float vector ``x`` and
    a quantized uint8 vector ``y``. The uint8 values in ``y`` are mapped
    back to floats using the provided ``quantized_values`` array.
    """
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        qy = quantized_values[y[i]]
        result += x[i] * qy
        norm_x += x[i] * x[i]
        norm_y += qy * qy

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return FLOAT32_MAX
    elif result <= 0.0:
        return FLOAT32_MAX
    else:
        result = result / np.sqrt(norm_x * norm_y)
        return -np.log2((result + 1.0) / 2.0)


@numba.njit(
    [
        "f4(f4[::1],u1[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def quantized_uint8_alternative_dot(x, y, quantized_values):
    r"""Alternative dot product distance between a float vector ``x`` and
    a quantized uint8 vector ``y``. The uint8 values in ``y`` are mapped
    back to floats using the provided ``quantized_values`` array. x and y
    are assumed to be normalized.
    """
    result = 0.0
    dim = x.shape[0]
    norm_y = 0.0

    for i in range(dim):
        qy = quantized_values[y[i]]
        result += x[i] * qy
        norm_y += qy * qy

    if result <= 0.0:
        return FLOAT32_MAX
    else:
        return -np.log2(result / np.sqrt(norm_y))


@numba.njit(
    [
        "f4(f4[::1],u1[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "quantized_index": numba.types.uint8,
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def quantized_uint4_sq_euclidean(x, y, quantized_values):
    r"""Squared Euclidean distance between a float vector ``x`` and
    a quantized uint8 vector ``y``. The uint8 values in ``y`` are mapped
    back to floats using upper and lower nibbles and via the provided
    ``quantized_values`` array.
    """
    result = 0.0
    dim = x.shape[0]

    for i in range(dim):
        byte = y[i // 2]
        if i % 2 == 0:
            quantized_index = byte & 0x0F  # Lower 4 bits
        else:
            quantized_index = (byte >> 4) & 0x0F  # Upper 4 bits

        diff = x[i] - quantized_values[quantized_index]
        result += diff * diff

    return result


@numba.njit(
    [
        "f4(f4[::1],u1[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "quantized_index": numba.types.uint8,
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def quantized_uint4_alternative_cosine(x, y, quantized_values):
    r"""Alternative cosine distance between a float vector ``x`` and
    a quantized uint8 vector ``y``. The uint8 values in ``y`` are mapped
    back to floats using upper and lower nibbles and via the provided
    ``quantized_values`` array.
    """
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        byte = y[i // 2]
        if i % 2 == 0:
            quantized_index = byte & 0x0F  # Lower 4 bits
        else:
            quantized_index = (byte >> 4) & 0x0F  # Upper 4 bits

        qy = quantized_values[quantized_index]
        result += x[i] * qy
        norm_x += x[i] * x[i]
        norm_y += qy * qy

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return FLOAT32_MAX
    elif result <= 0.0:
        return FLOAT32_MAX
    else:
        result = result / np.sqrt(norm_x * norm_y)
        return -np.log2((result + 1.0) / 2.0)


@numba.njit(
    [
        "f4(f4[::1],u1[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "quantized_index": numba.types.uint8,
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def quantized_uint4_alternative_dot(x, y, quantized_values):
    r"""Alternative dot product distance between a float vector ``x`` and
    a quantized uint8 vector ``y``. The uint8 values in ``y`` are mapped
    back to floats using upper and lower nibbles and via the provided
    ``quantized_values`` array. x and y are assumed to be normalized.
    """
    result = 0.0
    dim = x.shape[0]
    norm_y = 0.0

    for i in range(dim):
        byte = y[i // 2]
        if i % 2 == 0:
            quantized_index = byte & 0x0F  # Lower 4 bits
        else:
            quantized_index = (byte >> 4) & 0x0F  # Upper 4 bits

        qy = quantized_values[quantized_index]
        result += x[i] * qy
        norm_y += qy * qy

    if result <= 0.0:
        return FLOAT32_MAX
    else:
        return -np.log2(result / np.sqrt(norm_y))


named_distances = {
    # general minkowski distances
    "euclidean": euclidean,
    "l2": euclidean,
    "sqeuclidean": squared_euclidean,
    "manhattan": manhattan,
    "taxicab": manhattan,
    "l1": manhattan,
    "chebyshev": chebyshev,
    "linfinity": chebyshev,
    "linfty": chebyshev,
    "linf": chebyshev,
    "minkowski": minkowski,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean,
    "standardised_euclidean": standardised_euclidean,
    "wminkowski": weighted_minkowski,
    "weighted_minkowski": weighted_minkowski,
    "mahalanobis": mahalanobis,
    # Other distances
    "canberra": canberra,
    "cosine": cosine,
    "dot": dot,
    "inner_product": inner_product,
    "correlation": correlation,
    "haversine": haversine,
    "braycurtis": bray_curtis,
    "spearmanr": spearmanr,
    "tsss": tsss,
    "true_angular": true_angular,
    # Distribution distances
    "hellinger": hellinger,
    "kantorovich": kantorovich,
    "wasserstein": kantorovich,
    "wasserstein_1d": wasserstein_1d,
    "wasserstein-1d": wasserstein_1d,
    "kantorovich-1d": wasserstein_1d,
    "kantorovich_1d": wasserstein_1d,
    "circular_kantorovich": circular_kantorovich,
    "circular_wasserstein": circular_kantorovich,
    "sinkhorn": sinkhorn,
    "jensen-shannon": jensen_shannon_divergence,
    "jensen_shannon": jensen_shannon_divergence,
    "symmetric-kl": symmetric_kl_divergence,
    "symmetric_kl": symmetric_kl_divergence,
    "symmetric_kullback_liebler": symmetric_kl_divergence,
    # Binary distances
    "hamming": hamming,
    "jaccard": jaccard,
    "dice": dice,
    "matching": matching,
    "kulsinski": kulsinski,
    "rogerstanimoto": rogers_tanimoto,
    "russellrao": russellrao,
    "sokalsneath": sokal_sneath,
    "sokalmichener": sokal_michener,
    "yule": yule,
    "bit_hamming": bit_hamming,
    "bit_jaccard": bit_jaccard,
}

# Some distances have a faster to compute alternative that
# retains the same ordering of distances. We can compute with
# this instead, and then correct the final distances when complete.
# This provides a list of distances that have such an alternative
# along with the alternative distance function and the correction
# function to be applied.
fast_distance_alternatives = {
    "euclidean": {"dist": squared_euclidean, "correction": np.sqrt},
    "l2": {"dist": squared_euclidean, "correction": np.sqrt},
    "cosine": {"dist": alternative_cosine, "correction": correct_alternative_cosine},
    "dot": {"dist": alternative_dot, "correction": correct_alternative_cosine},
    "inner_product": {
        "dist": alternative_inner_product,
        "correction": correct_alternative_inner_product,
    },
    "true_angular": {
        "dist": alternative_cosine,
        "correction": true_angular_from_alt_cosine,
    },
    "hellinger": {
        "dist": alternative_hellinger,
        "correction": correct_alternative_hellinger,
    },
    "jaccard": {"dist": alternative_jaccard, "correction": correct_alternative_jaccard},
}

proxy_distances = {
    "proxy_inner_product": {
        "proxy_dist": proxy_inner_product,
        "true_dist": inner_product,
    },
    "proxy_wasserstein_1d": {
        "proxy_dist": proxy_wasserstein_1d,
        "true_dist": wasserstein_1d,
    },
    "proxy_wasserstein-1d": {
        "proxy_dist": proxy_wasserstein_1d,
        "true_dist": wasserstein_1d,
    },
    "proxy_kantorovich": {
        "proxy_dist": proxy_kantorovich,
        "true_dist": kantorovich,
    },
    "proxy_wasserstein": {
        "proxy_dist": proxy_kantorovich,
        "true_dist": kantorovich,
    },
    "proxy_circular_kantorovich": {
        "proxy_dist": proxy_circular_kantorovich,
        "true_dist": circular_kantorovich,
    },
    "proxy_circular_wasserstein": {
        "proxy_dist": proxy_circular_kantorovich,
        "true_dist": circular_kantorovich,
    },
    "proxy_jensen_shannon": {
        "proxy_dist": proxy_jensen_shannon,
        "true_dist": jensen_shannon_divergence,
    },
    "proxy_jensen-shannon": {
        "proxy_dist": proxy_jensen_shannon,
        "true_dist": jensen_shannon_divergence,
    },
    "proxy_symmetric_kl": {
        "proxy_dist": proxy_symmetric_kl,
        "true_dist": symmetric_kl_divergence,
    },
    "proxy_symmetric-kl": {
        "proxy_dist": proxy_symmetric_kl,
        "true_dist": symmetric_kl_divergence,
    },
    "proxy_sinkhorn": {
        "proxy_dist": proxy_sinkhorn,
        "true_dist": sinkhorn,
    },
}

quantized_distances = {
    "binary": {
        "euclidean": bit_hamming,
        "l2": bit_hamming,
        "cosine": bit_jaccard,
        "dot": bit_jaccard,
        "hamming": bit_hamming,
        "jaccard": bit_jaccard,
    },
    "uint8": {
        "euclidean": quantized_uint8_sq_euclidean,
        "l2": quantized_uint8_sq_euclidean,
        "cosine": quantized_uint8_alternative_cosine,
        "dot": quantized_uint8_alternative_dot,
    },
    "uint4": {
        "euclidean": quantized_uint4_sq_euclidean,
        "l2": quantized_uint4_sq_euclidean,
        "cosine": quantized_uint4_alternative_cosine,
        "dot": quantized_uint4_alternative_dot,
    },
}
