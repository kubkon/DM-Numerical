import pytest
import numpy as np

from ppm import cost_function, deriv_cost_function, objective_function


def test_cost_function():
    """
    Tests cost function, c_cost_function.
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


def test_deriv_cost_function():
    """
    Tests derivative of the cost function, c_deriv_cost_function.
    """
    # Setup scenario
    b_lower = 0.35
    b = 0.6
    v = [1.0, 2.0, 3.0]

    # Calculate expected result
    n = len(v)
    tmp = np.ones(n) * (b-b_lower)
    ps = [x**k for k,x in zip(range(n), tmp)]
    bs = np.array([i * x for i,x in zip(range(1, n+1), ps)])
    alphas = np.array(v)
    expected = np.dot(alphas, bs)

    # Calculate actual result
    actual = deriv_cost_function(b_lower, v, b)
    
    # Compare
    assert expected == actual


def test_objective_function():
    """
    Tests objective function, c_objective_function.
    """
    # Setup scenario
    k = 2
    granularity = 10
    b_lower = 0.25
    b_upper = 0.75
    lower_exts = [0.1, 0.2]
    upper_exts = [0.6, 0.7]
    n = len(lower_exts)
    vs = [0.1, 0.2, 0.3, 0.4]

    # Calculate actual result
    actual = objective_function(k, granularity, b_lower, b_upper,
                                lower_exts, upper_exts, vs)

    assert True


if __name__ == '__main__':
    test_objective_function()
