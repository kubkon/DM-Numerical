from ppm import solve
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

# set the scenario
w = 0.75
reputations = [0.25, 0.5, 0.75]
# compute an array of lower and upper extremities
lowers = np.array([(1-w)*r for r in reputations])
uppers = np.array([(1-w)*r + w for r in reputations])
# estimate the upper bound on bids
b_upper = upper_bound_bids(lowers, uppers)
# infer number of bidders
n = len(lowers)

# set initial conditions for the PPM algorithm
k = 3
K = 8
poly_coeffs = [[1e-2 for i in range(k)] for j in range(n)]
b_lower = lowers[1] + 1e-3
size_box = [1e-1 for i in range(k*n + 1)]

# run the PPM algorithm until k >= K
while True:
    b_lower, poly_coeffs = solve(b_lower,
                                 b_upper,
                                 lowers,
                                 uppers,
                                 poly_coeffs,
                                 size_box=size_box,
                                 granularity=100)

    if k >= K:
        break

    # extend polynomial coefficients by one element
    # for each bidder
    for i in range(n):
        poly_coeffs[i].append(1e-6)

    # update k
    k += 1

    # update size box
    size_box = [1e-2 for i in range(n*k + 1)]

print("Estimated lower bound on bids: %r" % b_lower)
print("Coefficients: %s" % poly_coeffs)
