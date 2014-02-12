import os

from functools import partial
import numpy as np

from dm.common import upper_bound_bids
import dm.efsm.main as efsm
import dm.ppm.main as ppm


def solve(w, reputations):
    # infer number of bidders
    n = reputations.size

    # compute an array of lower and upper extremities
    lowers = np.empty(n, dtype=np.float)
    uppers = np.empty(n, dtype=np.float)

    for i in np.arange(n):
        r = reputations[i]
        lowers[i] = (1-w) * r
        uppers[i] = (1-w) * r + w

    # estimate the upper bound on bids
    b_upper = upper_bound_bids(lowers, uppers)

    # solve the system using EFSM method
    bids, costs = efsm.solve(w, reputations)

    # truncate solution derived by EFSM where k=n
    min_i = np.argmin(np.absolute(np.copy(costs[n-1]) - lowers[n-1]))
    initial = costs.T[min_i,:]
    length = bids.size - min_i
    bids = bids[:min_i]
    costs = costs[:,:min_i]

    # solve the system for k = n using PPM method
    b_lower, b_upper, coeffs = ppm.solve_generic(initial, uppers, b_upper)
    bids_ = np.linspace(b_lower, b_upper, length)

    def cost_func(lower, cs, b):
        sums = sum([c*(b - b_lower)**i for c,i in zip(cs, range(1, len(cs)+1))])
        return lower + sums

    cost_funcs = [partial(cost_func, l, cs) for l,cs in zip(initial, coeffs)]
    costs_ = np.array([[f(b) for b in bids_] for f in cost_funcs])

    # combine results from both methods
    bids = np.append(bids, bids_)
    costs = np.hstack((costs, costs_))

    return bids, costs

if __name__ == "__main__":
    # set the scenario
    w = 0.55
    reputations = np.array([0.25, 0.5, 0.75], dtype=np.float)
    n = reputations.size

    # approximate
    bids, costs = solve(w, reputations)

    print("Estimated lower bound on bids: %r" % bids[0])

    # save the results in a file
    with open('combined.out', 'wt') as f:
        labels = ['w', 'reps', 'bids'] + ['costs_{}'.format(i) for i in range(n)]
        labels = ' '.join(labels)
        values = [w, reputations.tolist(), bids.tolist()] + [c.tolist() for c in costs]
        values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
        f.write(labels)
        f.write('\n')
        f.write(values)
