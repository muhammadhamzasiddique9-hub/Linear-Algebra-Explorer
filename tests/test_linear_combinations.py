
import numpy as np

from modules.linear_combinations import (
    calculate_linear_combination,
    calculate_span_dimension,
    describe_span
)


def test_linear_combination():

    a = np.array([2, 1])
    b = np.array([1, 3])

    result = calculate_linear_combination(
        a, b, 2, 3
    )

    assert np.array_equal(result, [7, 11])


def test_span_dimension_two():

    a = np.array([1, 0])
    b = np.array([0, 1])

    assert calculate_span_dimension(a, b) == 2


def test_span_dimension_one():

    a = np.array([1, 2])
    b = np.array([2, 4])

    assert calculate_span_dimension(a, b) == 1


def test_span_dimension_zero():

    a = np.array([0, 0])
    b = np.array([0, 0])

    assert calculate_span_dimension(a, b) == 0


def test_describe_span():

    assert "2D plane" in describe_span(2)
    assert "line" in describe_span(1)
    assert "zero vector" in describe_span(0)
