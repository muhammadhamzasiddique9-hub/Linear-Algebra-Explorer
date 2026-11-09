"""Basic operations for finite real-valued matrices."""

import numpy as np

from modules.validation import as_finite_array, positive_tolerance


def _matrix(value, name="Matrix"):
    matrix = as_finite_array(value, name, dimensions=(2,))
    if 0 in matrix.shape:
        raise ValueError(f"{name} cannot have an empty dimension.")
    return matrix


def add_matrices(matrix_a, matrix_b):
    a = _matrix(matrix_a, "Matrix A")
    b = _matrix(matrix_b, "Matrix B")
    if a.shape != b.shape:
        raise ValueError("Matrices must have the same dimensions for addition.")
    return np.add(a, b)


def multiply_matrices(matrix_a, matrix_b):
    a = _matrix(matrix_a, "Matrix A")
    b = _matrix(matrix_b, "Matrix B")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            "The number of columns in Matrix A must equal the number of rows in Matrix B."
        )
    return np.matmul(a, b)


def transpose_matrix(matrix):
    return np.transpose(_matrix(matrix))


def inverse_matrix(matrix, tolerance=1e-12):
    matrix = _matrix(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Only square matrices can have an inverse.")
    tolerance = positive_tolerance(tolerance)
    if np.linalg.matrix_rank(matrix, tol=tolerance) < matrix.shape[0]:
        raise np.linalg.LinAlgError("Matrix is singular and has no inverse.")
    return np.linalg.inv(matrix)
