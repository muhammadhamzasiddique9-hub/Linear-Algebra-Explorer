"""Basic operations for finite real-valued vectors."""

import numpy as np

from modules.validation import as_finite_array


def _vector(value, name):
    vector = as_finite_array(value, name, dimensions=(1,))
    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return vector


def _matching_vectors(vector_a, vector_b):
    a = _vector(vector_a, "Vector A")
    b = _vector(vector_b, "Vector B")
    if a.shape != b.shape:
        raise ValueError("Both vectors must have the same dimensions.")
    return a, b


def add_vectors(vector_a, vector_b):
    a, b = _matching_vectors(vector_a, vector_b)
    return np.add(a, b)


def subtract_vectors(vector_a, vector_b):
    a, b = _matching_vectors(vector_a, vector_b)
    return np.subtract(a, b)


def vector_magnitude(vector):
    return float(np.linalg.norm(_vector(vector, "Vector")))


def dot_product(vector_a, vector_b):
    a, b = _matching_vectors(vector_a, vector_b)
    return float(np.dot(a, b))
