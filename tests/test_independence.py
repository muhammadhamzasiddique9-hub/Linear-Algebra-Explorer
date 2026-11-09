
import numpy as np
import pytest

from modules.independence import (
    analyze_independence,
    describe_independence
)


def test_independent_vectors():

    a = np.array([1, 0])
    b = np.array([0, 1])

    result = analyze_independence(a, b)

    assert result["independent"] is True
    assert result["rank"] == 2


def test_dependent_vectors():

    a = np.array([1, 2])
    b = np.array([2, 4])

    result = analyze_independence(a, b)

    assert result["independent"] is False
    assert result["rank"] == 1


def test_zero_vectors():

    a = np.array([0, 0])
    b = np.array([0, 0])

    result = analyze_independence(a, b)

    assert result["independent"] is False
    assert result["rank"] == 0


def test_determinant():

    a = np.array([1, 0])
    b = np.array([0, 1])

    result = analyze_independence(a, b)

    assert np.isclose(result["determinant"], 1)


def test_different_dimensions():

    a = np.array([1, 2])
    b = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        analyze_independence(a, b)


def test_describe_independence():

    assert "Independent" in describe_independence(2)

    assert "Dependent" in describe_independence(1)

    assert "Dependent" in describe_independence(0)
