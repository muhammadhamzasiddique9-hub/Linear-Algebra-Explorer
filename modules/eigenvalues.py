"""Eigenvalue, eigenvector and diagonalization analysis for real matrices."""

import numpy as np

from modules.validation import as_finite_array, positive_tolerance


def validate_square_matrix(matrix):
    matrix = as_finite_array(matrix, "Input matrix", dimensions=(2,))
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Matrix cannot be empty.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    return matrix


def calculate_eigenvalues(matrix):
    return np.linalg.eigvals(validate_square_matrix(matrix))


def calculate_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(validate_square_matrix(matrix))
    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def calculate_characteristic_polynomial(matrix):
    return np.poly(validate_square_matrix(matrix))


def analyze_diagonalization(matrix, tolerance=1e-10):
    matrix = validate_square_matrix(matrix)
    tolerance = positive_tolerance(tolerance)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    size = matrix.shape[0]
    rank = int(np.linalg.matrix_rank(eigenvectors, tol=tolerance))
    complex_diagonalizable = rank == size

    eigenvalues_are_real = bool(np.all(np.abs(eigenvalues.imag) <= tolerance))
    real_diagonalizable = False
    if eigenvalues_are_real:
        real_vectors = np.real_if_close(eigenvectors, tol=1000)
        if np.isrealobj(real_vectors):
            real_rank = int(np.linalg.matrix_rank(real_vectors, tol=tolerance))
            real_diagonalizable = real_rank == size

    diagonal_matrix = np.diag(eigenvalues)
    inverse_eigenvector_matrix = (
        np.linalg.inv(eigenvectors) if complex_diagonalizable else None
    )

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rank": rank,
        "diagonalizable": complex_diagonalizable,
        "real_diagonalizable": real_diagonalizable,
        "diagonal_matrix": diagonal_matrix,
        "inverse_eigenvector_matrix": inverse_eigenvector_matrix,
    }


def analyze_eigenvalues(matrix):
    matrix = validate_square_matrix(matrix)
    eigen_result = calculate_eigenvectors(matrix)
    return {
        "eigenvalues": eigen_result["eigenvalues"],
        "eigenvectors": eigen_result["eigenvectors"],
        "characteristic_coefficients": calculate_characteristic_polynomial(matrix),
        "diagonalization": analyze_diagonalization(matrix),
    }
