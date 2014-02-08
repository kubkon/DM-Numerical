import os

import numpy as np

from dm.common import upper_bound_bids
import dm.fsm.fsm_internal as fsm_internal


def solve(w, reputations, granularity=10000):
    # infer number of bidders
    n = reputations.size

    # compute an array of lower and upper extremities
    lowers = np.empty(n, dtype=np.float)
    uppers = np.empty(n, dtype=np.float)
    for i in np.arange(n):
        lowers[i] = (1-w) * reputations[i]
        uppers[i] = (1-w) * reputations[i] + w

    # estimate the upper bound on bids
    b_upper = upper_bound_bids(lowers, uppers)

    # set initial conditions for the FSM algorithm
    low = lowers[1]
    high = b_upper
    epsilon = 1e-6
    num = granularity
    cond1 = np.empty(num, dtype=np.bool)
    cond2 = np.empty(num, dtype=np.bool)
    cond3 = np.empty(num-1, dtype=np.bool)

    # run the FSM algorithm until the estimate of the lower bound
    # on bids is found
    while high - low > epsilon:
        guess = 0.5 * (low + high)
        bids = np.linspace(guess, b_upper, num=num, endpoint=False)

        # solve the system
        try:
            costs = solve_extended(lowers, uppers, bids).T

        except Exception:
            # if an error is raised, set low to guess and continue
            low = guess
            continue

        for i in np.arange(n):
            for j in np.arange(num):
                x = costs[i][j]
                cond1[j] = lowers[i] <= x and x <= b_upper
                cond2[j] = bids[j] > x

        for i in np.arange(1, num):
            cond3[i-1] = bids[i-1] < bids[i]

        if np.all(cond1) and np.all(cond2) and np.all(cond3):
            high = guess
        else:
            low = guess

    try:
        return bids, costs

    except UnboundLocalError:
        raise Exception("Algorithm failed to converge.")

def solve_extended(lowers, uppers, bids):
    # estimate k(b) and c(b)
    k, _ = estimate_kc(bids[0], lowers)
    n = bids.size
    m = lowers.size

    if k == m:
        return fsm_internal.solve(lowers, uppers, bids)

    costs_ = np.empty((n,m), np.float)

    costs = fsm_internal.solve(lowers[:k], uppers[:k], bids)
    costs = add_extension(m, k, costs, bids)
    print(costs)

    # find the index of cut off
    index = np.argmin([np.absolute(c - lowers[k]) for c in costs[:,k]])

    for i in np.arange(index):
        for j in np.arange(m):
            costs_[i][j] = costs[i][j]

    costs = fsm_internal.solve(costs[index], uppers, bids[index:])

    for i in np.arange(bids[index:].size):
        for j in np.arange(m):
            costs_[index+i][j] = costs[i][j]

    return costs_

def estimate_kc(b_lower, lowers):
    n = lowers.size

    for k in np.arange(1, n):

        sums = 0
        for i in np.arange(k+1):
            sums += 1 / (b_lower - lowers[i])

        c = b_lower - k / sums

        if k < n-1:
            if lowers[k] <= c and c < lowers[k+1]:
                break

    return (k+1, c)

def add_extension(n, k, costs, bids):
    m = bids.size
    costs_ = np.empty((m, n-k), np.float)

    for i in np.arange(m):
        b = bids[i]

        sums = 0
        for j in np.arange(k):
            sums += 1 / (b - costs[i][j])

        for j in np.arange(n-k):
            costs_[i][j] = b - (k - 1) / sums

    costs = np.hstack((costs, costs_))

    return costs

if __name__ == "__main__":
    # set the scenario
    w = 0.55
    reputations = np.array([0.25, 0.5, 0.75], dtype=np.float)
    n = reputations.size

    # approximate
    bids, costs = solve(w, reputations)

    print("Estimated lower bound on bids: %r" % bids[0])

    # save the results in a file
    with open('fsm.out', 'wt') as f:
        labels = ['w', 'reps', 'bids'] + ['costs_{}'.format(i) for i in range(n)]
        labels = ' '.join(labels)
        values = [w, reputations.tolist(), bids.tolist()] + [c.tolist() for c in costs]
        values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
        f.write(labels)
        f.write('\n')
        f.write(values)
