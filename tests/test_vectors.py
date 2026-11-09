import numpy as np

from modules.vectors import (
    add_vectors,
    subtract_vectors,
    vector_magnitude,
    dot_product
)


def test_add_vectors():
    a = np.array([3, 4])
    b = np.array([2, 1])

    result = add_vectors(a, b)

    assert np.array_equal(result, [5, 5])


def test_subtract_vectors():
    a = np.array([3, 4])
    b = np.array([2, 1])

    result = subtract_vectors(a, b)

    assert np.array_equal(result, [1, 3])


def test_vector_magnitude():
    a = np.array([3, 4])

    result = vector_magnitude(a)

    assert result == 5


def test_dot_product():
    a = np.array([3, 4])
    b = np.array([2, 1])

    result = dot_product(a, b)

    assert result == 10