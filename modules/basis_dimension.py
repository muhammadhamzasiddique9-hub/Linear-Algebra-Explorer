"""Rank, pivot-column basis and nullity analysis."""

import numpy as np

from modules.validation import as_finite_array, positive_tolerance


def analyze_basis(matrix, tolerance=1e-10):
    matrix = as_finite_array(matrix, "Matrix", dimensions=(2,))
    if 0 in matrix.shape:
        raise ValueError("Matrix cannot have an empty dimension.")
    tolerance = positive_tolerance(tolerance)

    rows, columns = matrix.shape
    work = matrix.copy()
    pivot_columns = []
    pivot_row = 0

    # Combine the user tolerance with a floating-point precision threshold.
    # This avoids both tiny round-off pivots and overly aggressive scale-based cutoffs.
    scale = float(np.linalg.norm(matrix, ord=np.inf))
    numerical_threshold = (
        np.finfo(float).eps * max(rows, columns) * scale
    )
    threshold = max(tolerance, numerical_threshold)

    for column in range(columns):
        if pivot_row >= rows:
            break

        candidates = np.abs(work[pivot_row:, column])
        best_offset = int(np.argmax(candidates))
        best_row = pivot_row + best_offset

        if abs(work[best_row, column]) <= threshold:
            continue

        if best_row != pivot_row:
            work[[pivot_row, best_row]] = work[[best_row, pivot_row]]

        pivot = work[pivot_row, column]
        for row in range(pivot_row + 1, rows):
            factor = work[row, column] / pivot
            work[row, column:] -= factor * work[pivot_row, column:]
            work[row, column] = 0.0

        pivot_columns.append(column)
        pivot_row += 1

    rank = len(pivot_columns)
    basis_vectors = [matrix[:, column].copy() for column in pivot_columns]

    return {
        "rank": rank,
        "dimension": rank,
        "nullity": columns - rank,
        "pivot_columns": pivot_columns,
        "basis_vectors": basis_vectors,
    }


def describe_basis(rank, rows, columns):
    if rank == 0:
        return "The column space contains only the zero vector. Its dimension is zero."
    if rank == rows:
        return f"The columns span the entire R^{rows} space (full row rank)."
    return f"The column space has dimension {rank}. It is a {rank}-dimensional subspace of R^{rows}."
