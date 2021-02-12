import pytest
import numpy as np
from numpy.testing import assert_array_equal

from pynndescent.distances import rankdata


def test_empty():
    """rankdata([]) should return an empty array."""
    a = np.array([], dtype=int)
    r = rankdata(a)
    assert_array_equal(r, np.array([], dtype=np.float64))


def test_one():
    """Check rankdata with an array of length 1."""
    data = [100]
    a = np.array(data, dtype=int)
    r = rankdata(a)
    assert_array_equal(r, np.array([1.0], dtype=np.float64))


def test_basic():
    """Basic tests of rankdata."""
    data = [100, 10, 50]
    expected = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    a = np.array(data, dtype=int)
    r = rankdata(a)
    assert_array_equal(r, expected)

    data = [40, 10, 30, 10, 50]
    expected = np.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=np.float64)
    a = np.array(data, dtype=int)
    r = rankdata(a)
    assert_array_equal(r, expected)

    data = [20, 20, 20, 10, 10, 10]
    expected = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
    a = np.array(data, dtype=int)
    r = rankdata(a)
    assert_array_equal(r, expected)
    # The docstring states explicitly that the argument is flattened.
    a2d = a.reshape(2, 3)
    r = rankdata(a2d)
    assert_array_equal(r, expected)


def test_rankdata_object_string():
    min_rank = lambda a: [1 + sum(i < j for i in a) for j in a]
    max_rank = lambda a: [sum(i <= j for i in a) for j in a]
    ordinal_rank = lambda a: min_rank([(x, i) for i, x in enumerate(a)])

    def average_rank(a):
        return np.array([(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))])

    def dense_rank(a):
        b = np.unique(a)
        return np.array([1 + sum(i < j for i in b) for j in a])

    rankf = dict(
        min=min_rank,
        max=max_rank,
        ordinal=ordinal_rank,
        average=average_rank,
        dense=dense_rank,
    )

    def check_ranks(a):
        for method in "min", "max", "dense", "ordinal", "average":
            out = rankdata(a, method=method)
            assert_array_equal(out, rankf[method](a))

    check_ranks(np.random.uniform(size=[200]))


def test_large_int():
    data = np.array([2 ** 60, 2 ** 60 + 1], dtype=np.uint64)
    r = rankdata(data)
    assert_array_equal(r, [1.0, 2.0])

    data = np.array([2 ** 60, 2 ** 60 + 1], dtype=np.int64)
    r = rankdata(data)
    assert_array_equal(r, [1.0, 2.0])

    data = np.array([2 ** 60, -(2 ** 60) + 1], dtype=np.int64)
    r = rankdata(data)
    assert_array_equal(r, [2.0, 1.0])


def test_big_tie():
    for n in [10000, 100000, 1000000]:
        data = np.ones(n, dtype=int)
        r = rankdata(data)
        expected_rank = 0.5 * (n + 1)
        assert_array_equal(r, expected_rank * data, "test failed with n=%d" % n)


@pytest.mark.parametrize(
    "values,method,expected",
    [  # values, method, expected
        (np.array([], np.float64), "average", np.array([], np.float64)),
        (np.array([], np.float64), "min", np.array([], np.float64)),
        (np.array([], np.float64), "max", np.array([], np.float64)),
        (np.array([], np.float64), "dense", np.array([], np.float64)),
        (np.array([], np.float64), "ordinal", np.array([], np.float64)),
        #
        (np.array([100], np.float64), "average", np.array([1.0], np.float64)),
        (np.array([100], np.float64), "min", np.array([1.0], np.float64)),
        (np.array([100], np.float64), "max", np.array([1.0], np.float64)),
        (np.array([100], np.float64), "dense", np.array([1.0], np.float64)),
        (np.array([100], np.float64), "ordinal", np.array([1.0], np.float64)),
        # #
        (
            np.array([100, 100, 100], np.float64),
            "average",
            np.array([2.0, 2.0, 2.0], np.float64),
        ),
        (
            np.array([100, 100, 100], np.float64),
            "min",
            np.array([1.0, 1.0, 1.0], np.float64),
        ),
        (
            np.array([100, 100, 100], np.float64),
            "max",
            np.array([3.0, 3.0, 3.0], np.float64),
        ),
        (
            np.array([100, 100, 100], np.float64),
            "dense",
            np.array([1.0, 1.0, 1.0], np.float64),
        ),
        (
            np.array([100, 100, 100], np.float64),
            "ordinal",
            np.array([1.0, 2.0, 3.0], np.float64),
        ),
        #
        (
            np.array([100, 300, 200], np.float64),
            "average",
            np.array([1.0, 3.0, 2.0], np.float64),
        ),
        (
            np.array([100, 300, 200], np.float64),
            "min",
            np.array([1.0, 3.0, 2.0], np.float64),
        ),
        (
            np.array([100, 300, 200], np.float64),
            "max",
            np.array([1.0, 3.0, 2.0], np.float64),
        ),
        (
            np.array([100, 300, 200], np.float64),
            "dense",
            np.array([1.0, 3.0, 2.0], np.float64),
        ),
        (
            np.array([100, 300, 200], np.float64),
            "ordinal",
            np.array([1.0, 3.0, 2.0], np.float64),
        ),
        #
        (
            np.array([100, 200, 300, 200], np.float64),
            "average",
            np.array([1.0, 2.5, 4.0, 2.5], np.float64),
        ),
        (
            np.array([100, 200, 300, 200], np.float64),
            "min",
            np.array([1.0, 2.0, 4.0, 2.0], np.float64),
        ),
        (
            np.array([100, 200, 300, 200], np.float64),
            "max",
            np.array([1.0, 3.0, 4.0, 3.0], np.float64),
        ),
        (
            np.array([100, 200, 300, 200], np.float64),
            "dense",
            np.array([1.0, 2.0, 3.0, 2.0], np.float64),
        ),
        (
            np.array([100, 200, 300, 200], np.float64),
            "ordinal",
            np.array([1.0, 2.0, 4.0, 3.0], np.float64),
        ),
        #
        (
            np.array([100, 200, 300, 200, 100], np.float64),
            "average",
            np.array([1.5, 3.5, 5.0, 3.5, 1.5], np.float64),
        ),
        (
            np.array([100, 200, 300, 200, 100], np.float64),
            "min",
            np.array([1.0, 3.0, 5.0, 3.0, 1.0], np.float64),
        ),
        (
            np.array([100, 200, 300, 200, 100], np.float64),
            "max",
            np.array([2.0, 4.0, 5.0, 4.0, 2.0], np.float64),
        ),
        (
            np.array([100, 200, 300, 200, 100], np.float64),
            "dense",
            np.array([1.0, 2.0, 3.0, 2.0, 1.0], np.float64),
        ),
        (
            np.array([100, 200, 300, 200, 100], np.float64),
            "ordinal",
            np.array([1.0, 3.0, 5.0, 4.0, 2.0], np.float64),
        ),
        #
        (
            np.array([10] * 30, np.float64),
            "ordinal",
            np.arange(1.0, 31.0, dtype=np.float64),
        ),
    ],
)
def test_cases(values, method, expected):
    r = rankdata(values, method=method)
    assert_array_equal(r, expected)
