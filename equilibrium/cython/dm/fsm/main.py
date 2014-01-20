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

        try:
            costs = fsm_internal.solve(lowers, uppers, bids).T
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


if __name__ == "__main__":
    # set the scenario
    w = 0.85
    reputations = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float)
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
