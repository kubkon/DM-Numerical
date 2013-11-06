import pytest
import numpy as np

from ppm_b import cost_function, deriv_cost_function, linspace


def test_cost_function():
    """
    Tests cost function, c_cost_function.
    """
    # Setup scenario
    b_lower = 0.35
    b = 0.6
    v = [1.0, 2.0, 3.0]

    # Calculate expected result
    n = len(v)
    tmp = np.ones(n) * (b - b_lower)
    bs = np.array([t**k for k,t in zip(range(n), tmp)])
    alphas = np.array(v)
    expected = b_lower + np.dot(alphas, bs)

    # Calculate actual result
    actual = cost_function(b_lower, v, b)
    
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
    ps = [x**(k-1) for k,x in zip(range(n), tmp)]
    bs = np.array([i * x for i,x in zip(range(n), ps)])
    alphas = np.array(v)
    expected = np.dot(alphas, bs)

    # Calculate actual result
    actual = deriv_cost_function(b_lower, v, b)
    
    # Compare
    assert expected == actual


def test_linspace():
    """
    Tests linspace function, c_linspace.
    """
    # Setup scenario
    begin = 0.0
    end = 1.0
    granularity = 1000

    # Calculate expected result
    expected = np.linspace(begin, end, granularity)

    # Calculate actual result
    actual = linspace(begin, end, granularity)

    # Compare
    for e, a in zip(expected, actual):
        assert e == a

