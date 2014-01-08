import pytest
import numpy as np

from bajari.ppm.ppm_internal import p_cost_function, p_deriv_cost_function, p_linspace


def test_cost_function():
    """
    Tests cost function.
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
    actual = p_cost_function(b_lower, v, b)
    
    # Compare
    assert expected == actual


def test_deriv_cost_function():
    """
    Tests derivative of the cost function.
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
    actual = p_deriv_cost_function(b_lower, v, b)
    
    # Compare
    assert expected == actual


def test_linspace():
    """
    Tests linspace function.
    """
    # Setup scenario
    begin = 0.0
    end = 1.0
    granularity = 1000

    # Calculate expected result
    expected = np.linspace(begin, end, granularity)

    # Calculate actual result
    actual = p_linspace(begin, end, granularity)

    # Compare
    for e, a in zip(expected, actual):
        assert e == a

