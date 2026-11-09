"""Linear independence analysis for two real-valued vectors."""

import numpy as np

from modules.validation import as_finite_array


def analyze_independence(vector_a, vector_b, tolerance=None):
    a = as_finite_array(vector_a, "Vector A", dimensions=(1,))
    b = as_finite_array(vector_b, "Vector B", dimensions=(1,))

    if a.size == 0 or b.size == 0:
        raise ValueError("Vectors cannot be empty.")
    if a.shape != b.shape:
        raise ValueError("Both vectors must have the same dimensions.")

    matrix = np.column_stack((a, b))
    if tolerance is None:
        rank = int(np.linalg.matrix_rank(matrix))
    else:
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Tolerance must be a positive finite number.")
        rank = int(np.linalg.matrix_rank(matrix, tol=tolerance))

    determinant = float(np.linalg.det(matrix)) if a.size == 2 else None
    return {
        "rank": rank,
        "independent": rank == 2,
        "determinant": determinant,
    }


def describe_independence(rank):
    if rank == 2:
        return "Independent: neither vector can be formed as a scalar multiple of the other."
    if rank == 1:
        return "Dependent: the vectors lie on the same line through the origin."
    if rank == 0:
        return "Dependent: both vectors are zero vectors."
    return "Unsupported rank for two vectors."
