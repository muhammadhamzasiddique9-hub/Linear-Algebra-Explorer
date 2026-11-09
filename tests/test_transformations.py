
import numpy as np
import pytest

from modules.transformations import (
    apply_transformation,
    calculate_determinant,
    analyze_transformation
)


def test_identity_transformation():

    matrix = np.array([
        [1, 0],
        [0, 1]
    ])

    vector = np.array([3, 4])

    result = apply_transformation(matrix, vector)

    assert np.allclose(result, [3, 4])


def test_scaling_transformation():

    matrix = np.array([
        [2, 0],
        [0, 1]
    ])

    vector = np.array([1, 2])

    result = apply_transformation(matrix, vector)

    assert np.allclose(result, [2, 2])


def test_determinant():

    matrix = np.array([
        [2, 0],
        [0, 1]
    ])

    assert np.isclose(
        calculate_determinant(matrix),
        2
    )


def test_singular_transformation():

    matrix = np.array([
        [1, 2],
        [2, 4]
    ])

    result = analyze_transformation(matrix)

    assert result["singular"] is True
    assert np.isclose(result["area_scale_factor"], 0)


def test_non_singular_transformation():

    matrix = np.array([
        [2, 0],
        [0, 3]
    ])

    result = analyze_transformation(matrix)

    assert result["singular"] is False
    assert np.isclose(result["area_scale_factor"], 6)


def test_invalid_matrix():

    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    vector = np.array([1, 2])

    with pytest.raises(ValueError):
        apply_transformation(matrix, vector)


def test_invalid_vector():

    matrix = np.eye(2)
    vector = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        apply_transformation(matrix, vector)
