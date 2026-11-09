
import numpy as np
import pytest

from modules.basis_dimension import (
    analyze_basis,
    describe_basis
)


def test_full_rank_matrix():

    matrix = np.array([
        [1, 0],
        [0, 1]
    ])

    result = analyze_basis(matrix)

    assert result["rank"] == 2
    assert result["dimension"] == 2
    assert result["nullity"] == 0


def test_dependent_columns():

    matrix = np.array([
        [1, 2, 3],
        [0, 1, 1],
        [1, 3, 4]
    ])

    result = analyze_basis(matrix)

    assert result["rank"] == 2
    assert result["pivot_columns"] == [0, 1]
    assert result["nullity"] == 1


def test_zero_matrix():

    matrix = np.zeros((2, 3))

    result = analyze_basis(matrix)

    assert result["rank"] == 0
    assert result["dimension"] == 0
    assert result["nullity"] == 3


def test_basis_vectors_are_original_columns():

    matrix = np.array([
        [1, 2, 3],
        [0, 1, 1],
        [1, 3, 4]
    ])

    result = analyze_basis(matrix)

    assert np.array_equal(
        result["basis_vectors"][0],
        matrix[:, 0]
    )

    assert np.array_equal(
        result["basis_vectors"][1],
        matrix[:, 1]
    )


def test_rectangular_matrix():

    matrix = np.array([
        [1, 0, 2],
        [0, 1, 3]
    ])

    result = analyze_basis(matrix)

    assert result["rank"] == 2
    assert result["nullity"] == 1


def test_invalid_matrix():

    with pytest.raises(ValueError):
        analyze_basis(np.array([1, 2, 3]))


def test_describe_basis():

    assert "zero" in describe_basis(0, 2, 3)

    assert "entire R^2" in describe_basis(2, 2, 2)

    assert "dimension 1" in describe_basis(1, 2, 3)
