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
    num = 10000
    vals = np.linspace(uppers[0], uppers[1], num)
    tabulated = np.empty(num, dtype=np.float)
    n = lowers.size

    # solve the optimization problem in Eq. (1.8) in the thesis
    for i in np.arange(num):
        v = vals[i]
        probs = 1
        for j in np.arange(1, n):
            l, u = lowers[j], uppers[j]
            probs *= 1 - uniform(loc=l, scale=(u-l)).cdf(v)
        tabulated[i] = (v - uppers[0]) * probs

    return vals[np.argmax(tabulated)]
