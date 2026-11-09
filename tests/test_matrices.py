import numpy as np

from modules.matrices import (
    add_matrices,
    multiply_matrices,
    transpose_matrix,
    inverse_matrix
)


def test_add_matrices():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    result = add_matrices(a, b)

    assert np.array_equal(result, [[6, 8], [10, 12]])


def test_multiply_matrices():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    result = multiply_matrices(a, b)

    assert np.array_equal(result, [[19, 22], [43, 50]])


def test_transpose_matrix():
    a = np.array([[1, 2], [3, 4]])

    result = transpose_matrix(a)

    assert np.array_equal(result, [[1, 3], [2, 4]])


def test_inverse_matrix():
    a = np.array([[1, 2], [3, 4]])

    result = inverse_matrix(a)

    assert np.allclose(
        result,
        [[-2, 1], [1.5, -0.5]]
    )