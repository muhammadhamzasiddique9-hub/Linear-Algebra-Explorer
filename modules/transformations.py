"""Two-dimensional linear transformation operations."""

import numpy as np

from modules.validation import as_finite_array


def _transformation_matrix(value):
    matrix = as_finite_array(value, "Transformation matrix", dimensions=(2,))
    if matrix.shape != (2, 2):
        raise ValueError("Transformation matrix must be 2x2.")
    return matrix


def apply_transformation(matrix, vector):
    matrix = _transformation_matrix(matrix)
    vector = as_finite_array(vector, "Input vector", dimensions=(1,))
    if vector.shape != (2,):
        raise ValueError("Input vector must have two components.")
    return matrix @ vector


def calculate_determinant(matrix):
    return float(np.linalg.det(_transformation_matrix(matrix)))


def analyze_transformation(matrix):
    matrix = _transformation_matrix(matrix)
    determinant = float(np.linalg.det(matrix))
    singular = bool(np.linalg.matrix_rank(matrix) < 2)

    if singular:
        description = (
            "This transformation is singular. It collapses the plane into a line or a point."
        )
    else:
        description = (
            "This transformation is non-singular. It maps the plane onto the plane."
        )

    return {
        "determinant": determinant,
        "area_scale_factor": abs(determinant),
        "singular": singular,
        "description": description,
    }
