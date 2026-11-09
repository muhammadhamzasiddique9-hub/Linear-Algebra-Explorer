import numpy as np
import pytest

from modules.eigenvalues import (
    calculate_eigenvalues,
    calculate_eigenvectors,
    calculate_characteristic_polynomial,
    analyze_diagonalization,
    analyze_eigenvalues
)


def test_eigenvalues():

    matrix = np.array([
        [2, 0],
        [0, 3]
    ])

    result = calculate_eigenvalues(matrix)

    assert np.allclose(
        np.sort(result),
        [2, 3]
    )


def test_eigenvectors():

    matrix = np.array([
        [2, 0],
        [0, 3]
    ])

    result = calculate_eigenvectors(matrix)

    assert np.allclose(
        matrix @ result["eigenvectors"],
        result["eigenvectors"] @ np.diag(
            result["eigenvalues"]
        )
    )


def test_characteristic_polynomial():

    matrix = np.array([
        [2, 0],
        [0, 3]
    ])

    result = calculate_characteristic_polynomial(matrix)

    assert np.allclose(result, [1, -5, 6])


def test_diagonalizable_matrix():

    matrix = np.array([
        [2, 0],
        [0, 3]
    ])

    result = analyze_diagonalization(matrix)

    assert result["diagonalizable"] is True
    assert result["real_diagonalizable"] is True


def test_non_diagonalizable_matrix():

    matrix = np.array([
        [1, 1],
        [0, 1]
    ])

    result = analyze_diagonalization(matrix)

    assert result["diagonalizable"] is False
    assert result["real_diagonalizable"] is False


def test_complex_eigenvalues():

    matrix = np.array([
        [0, -1],
        [1, 0]
    ])

    result = analyze_diagonalization(matrix)

    assert np.allclose(
        np.sort_complex(result["eigenvalues"]),
        np.sort_complex([1j, -1j])
    )

    assert result["diagonalizable"] is True
    assert result["real_diagonalizable"] is False


def test_complete_analysis():

    matrix = np.array([
        [2, 0],
        [0, 3]
    ])

    result = analyze_eigenvalues(matrix)

    assert "eigenvalues" in result
    assert "eigenvectors" in result
    assert "characteristic_coefficients" in result
    assert "diagonalization" in result


def test_non_square_matrix():

    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    with pytest.raises(ValueError):
        calculate_eigenvalues(matrix)


def test_matrix_with_invalid_values():

    matrix = np.array([
        [1, np.nan],
        [2, 3]
    ])

    with pytest.raises(ValueError):
        calculate_eigenvalues(matrix)