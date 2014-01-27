import numpy as np
from numpy.polynomial.polynomial import polyval

from .polyfit import fit
from .interpolate import CubicSpline

def verify_sufficiency(costs, bids, b_upper, cdfs, step=100):
    # Infer number of bidders
    n = costs.shape[0]
    # Sample bidding space
    sample_indices = np.arange(0, costs.shape[1], step)
    m = sample_indices.size
    # Initialize results arrays
    sampled_costs = np.empty((n, m), dtype=np.float)
    best_responses = np.empty((n, m), dtype=np.float)

    for i in np.arange(n):
        z = 0

        for j in sample_indices:
            # Get current cost
            cost = costs[i][j]
            # Populate sampled costs
            sampled_costs[i][z] = cost
            # Tabulate space of feasible bids for each sampled cost
            feasible_bids = np.linspace(cost, b_upper, 100)
            n_bids = feasible_bids.size
            # Initialize utility array
            utility = np.empty(n_bids, dtype=np.float)

            for k in np.arange(n_bids):
                # Get currently considered bid
                bid = feasible_bids[k]
                # The upper bound on utility given current cost-bid pair
                utility[k] = bid - cost

                if bid >= bids[0]:
                    # Compute probability of winning
                    corr_bid = np.argmin(np.absolute(bids - np.ones(bids.size) * bid))
                    probability = 1
                    for jj in np.arange(n):
                        if jj == i:
                            continue

                        probability *= (1 - cdfs[jj].cdf(costs[jj][corr_bid]))

                    utility[k] *= probability

            
            best_responses[i][z] = feasible_bids[np.argmax(utility)]
            z += 1

    return sampled_costs, best_responses

def estimate_winning_probs(costs, cdf, params):
    # initialize
    shape = costs.shape
    probs = np.empty(shape, np.float)
    n, m = shape[0], shape[1]

    # compute probabilities
    for i in np.arange(n):
        # get index of the competing bidder
        j = (i+1) % 2
        # extract cdf function
        try:
            prob_func = cdf(params[j]['shape'], loc=params[j]['location'], scale=params[j]['scale'])

        except KeyError:
            prob_func = cdf(**params[j])

        for k in np.arange(m):
            probs[i][k] = 1 - prob_func.cdf(costs[i][k])

    return probs

def compute_expected_utilities(bids, costs, cdf, params):
    # compute probabilities of winning
    probs = estimate_winning_probs(costs, cdf, params)

    # compute expected utilities
    exp_utilities = np.empty(costs.shape, np.float)

    for i in np.arange(costs.shape[0]):
        exp_utilities[i] = np.multiply((bids - costs[i]), probs[i])
    
    return exp_utilities

def ks_statistic(xs, func1, func2):
    # initialize
    n = xs.size
    differences = np.empty(n, np.float)

    # compute differences between functions for each x
    for i in np.arange(n):
        differences[i] = np.absolute(func1[i] - func2[i])

    # compute argmax of the differences
    max_index = np.argmax(differences)

    return (xs[max_index], differences[max_index])

def polyfit(xs, ys, degree=3, maxiter=500):
    coeffs = fit(xs, ys, num_coeffs=(degree+1), maxiter=maxiter)
    return lambda x: polyval(x, coeffs)

def csplinefit(xs, ys):
    def inner(xss):
        yss = np.empty(xss.size, np.float)

        with CubicSpline(xs, ys) as spline:
            for i in np.arange(xss.size):
                yss[i] = spline.evaluate(xss[i])

        return yss

    return inner
