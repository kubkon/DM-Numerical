import pytest
import numpy as np

from ppm import cost_function


def test_cost_function():
    """
    Tests cost function.
    """
    # Setup scenario
    low_ext = 0.25
    b_lower = 0.35
    b = 0.6
    v = [1.0, 2.0, 3.0]

    # Calculate expected result
    n = len(v)
    tmp = np.ones(n) * (b - b_lower)
    bs = np.array([t**k for k,t in zip(range(1, n+1), tmp)])
    alphas = np.array(v)
    expected = low_ext + np.dot(alphas, bs)

    # Calculate actual result
    actual = cost_function(low_ext, b_lower, v, b)
    
    # Compare
    assert expected == actual
