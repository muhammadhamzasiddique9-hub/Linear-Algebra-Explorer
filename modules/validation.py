"""Shared input validation helpers for the Linear Algebra Explorer."""

import numpy as np


def as_finite_array(value, name="Input", dimensions=None):
    """Convert input to a float NumPy array and validate shape and values."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain valid real numbers.") from error

    if dimensions is not None and array.ndim not in dimensions:
        expected = " or ".join(str(item) for item in dimensions)
        raise ValueError(f"{name} must have {expected} dimension(s).")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers.")

    return array


def positive_tolerance(tolerance):
    """Validate and return a positive finite numerical tolerance."""
    try:
        value = float(tolerance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Tolerance must be a positive finite number.") from error

    if not np.isfinite(value) or value <= 0:
        raise ValueError("Tolerance must be a positive finite number.")

    return value
