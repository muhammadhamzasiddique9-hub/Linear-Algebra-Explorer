"""Linear combinations and span analysis for two real vectors."""

import numpy as np

from modules.validation import as_finite_array


def _matching_vectors(vector_a, vector_b):
    a = as_finite_array(vector_a, "Vector A", dimensions=(1,))
    b = as_finite_array(vector_b, "Vector B", dimensions=(1,))
    if a.size == 0 or b.size == 0:
        raise ValueError("Vectors cannot be empty.")
    if a.shape != b.shape:
        raise ValueError("Both vectors must have the same dimensions.")
    return a, b


def calculate_linear_combination(vector_a, vector_b, coefficient_a, coefficient_b):
    a, b = _matching_vectors(vector_a, vector_b)
    try:
        alpha = float(coefficient_a)
        beta = float(coefficient_b)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Coefficients must be valid real numbers.") from error
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("Coefficients must be finite numbers.")
    return alpha * a + beta * b


def calculate_span_dimension(vector_a, vector_b, tolerance=None):
    a, b = _matching_vectors(vector_a, vector_b)
    matrix = np.column_stack((a, b))
    if tolerance is None:
        return int(np.linalg.matrix_rank(matrix))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be a positive finite number.")
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def describe_span(dimension):
    if dimension == 2:
        return "These vectors span a 2D plane. In R², this is the entire plane; in a higher-dimensional space, it is a 2D subspace."
    if dimension == 1:
        return "These vectors span a line through the origin."
    if dimension == 0:
        return "These vectors span only the zero vector."
    return "Unsupported span dimension for two vectors."
