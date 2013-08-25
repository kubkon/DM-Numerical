import numpy as np
from scipy.stats import uniform
from functools import reduce


def upper_bound_bids(lowers, uppers):
    """Returns an estimate on upper bound on bids.

    Arguments (all NumPy arrays):
    lowers -- array of lower extremities
    uppers -- array of upper extremities
    """
    # tabulate the range of permissible values
    vals = np.linspace(uppers[0], uppers[1], 10000)
    tabulated = []
    # solve the optimization problem in Eq. (1.8) in the thesis
    for v in vals:
        probs = [1-uniform(loc=l, scale=(u-l)).cdf(v) for l, u in zip(lowers[1:], uppers[1:])]
        tabulated += [(v - uppers[0]) * reduce(lambda p,r: p*r, probs, 1)]
    tabulated = np.array(tabulated)
    return vals[np.argmax(tabulated)]
