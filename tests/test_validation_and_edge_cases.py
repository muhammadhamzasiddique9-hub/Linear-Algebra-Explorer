import numpy as np
import pytest

from modules.vectors import add_vectors, vector_magnitude, dot_product
from modules.matrices import add_matrices, multiply_matrices, inverse_matrix
from modules.linear_combinations import calculate_linear_combination, calculate_span_dimension
from modules.independence import analyze_independence
from modules.basis_dimension import analyze_basis
from modules.transformations import apply_transformation
from modules.eigenvalues import validate_square_matrix, analyze_diagonalization


def test_vector_addition_rejects_different_shapes():
    with pytest.raises(ValueError):
        add_vectors([1, 2], [1, 2, 3])


def test_vector_functions_reject_non_finite_values():
    with pytest.raises(ValueError):
        vector_magnitude([1, np.inf])
    with pytest.raises(ValueError):
        dot_product([1, np.nan], [2, 3])


def test_vector_functions_reject_empty_vectors():
    with pytest.raises(ValueError):
        vector_magnitude([])


def test_matrix_addition_rejects_different_shapes():
    with pytest.raises(ValueError):
        add_matrices([[1, 2]], [[1], [2]])


def test_matrix_multiplication_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        multiply_matrices([[1, 2]], [[1, 2]])


def test_matrix_inverse_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        inverse_matrix([[1, 2, 3], [4, 5, 6]])


def test_matrix_inverse_rejects_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        inverse_matrix([[1, 2], [2, 4]])


def test_linear_combination_rejects_non_finite_coefficient():
    with pytest.raises(ValueError):
        calculate_linear_combination([1, 0], [0, 1], np.nan, 1)


def test_span_dimension_for_vectors_in_three_dimensions():
    assert calculate_span_dimension([1, 0, 0], [0, 1, 0]) == 2


def test_independence_rejects_non_finite_values():
    with pytest.raises(ValueError):
        analyze_independence([1, np.nan], [0, 1])


def test_basis_handles_small_scaled_independent_matrix():
    matrix = np.array([[1e-14, 0.0], [0.0, 1e-14]])
    result = analyze_basis(matrix, tolerance=1e-16)
    assert result["rank"] == 2
    assert result["pivot_columns"] == [0, 1]


def test_basis_handles_large_scale_well_conditioned_matrix():
    result = analyze_basis(np.diag([1e12, 1.0]))
    assert result["rank"] == 2


def test_basis_rejects_non_finite_tolerance():
    with pytest.raises(ValueError):
        analyze_basis(np.eye(2), tolerance=np.inf)


def test_transformation_rejects_non_finite_vector():
    with pytest.raises(ValueError):
        apply_transformation(np.eye(2), [1, np.inf])


def test_eigenvalue_validation_rejects_empty_matrix():
    with pytest.raises(ValueError):
        validate_square_matrix(np.empty((0, 0)))


def test_repeated_eigenvalues_can_still_be_diagonalizable():
    result = analyze_diagonalization(np.eye(2))
    assert result["diagonalizable"] is True
    assert result["real_diagonalizable"] is True
