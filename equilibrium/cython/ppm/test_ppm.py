import pytest

from ppm import cost_function


def test_cost_function():
    """
    Tests cost function.
    """
    low_ext = 0.25
    b_lower = 0.35
    b = 0.6
    v = [1.0, 2.0, 3.0]

    actual = cost_function(low_ext, b_lower, v, b)
    expected = 1.0
    assert expected == actual
