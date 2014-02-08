import numpy as np

import bajari.fsm.fsm_internal as fsm_internal


def solve(support, params, granularity=10000):
    # set initial conditions for the FSM algorithm
    low = support[0]
    high = support[1]
    epsilon = 1e-6
    num = granularity
    n = len(params)
    cond1 = np.empty(num, dtype=np.bool)
    cond2 = np.empty(num, dtype=np.bool)
    cond3 = np.empty(num-1, dtype=np.bool)

    # run the FSM algorithm until the estimate of the lower bound
    # on bids is found
    while high - low > epsilon:
        guess = 0.5 * (low + high)
        bids = np.linspace(guess, support[1], num=num, endpoint=False)
        try:
            costs = fsm_internal.solve(params, support, bids).T
        except Exception:
            # if an error is raised, set low to guess and continue
            low = guess
            continue

        for i in np.arange(n):
            for j in np.arange(num):
                x = costs[i][j]
                cond1[j] = support[0] <= x and x <= support[1]
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
    support = [2, 8]
    params = [{'location': 4.0, 'scale': 1.5, 'shape': 0},
              {'location': 5.0, 'scale': 1.5, 'shape': 0},
              {'location': 6.0, 'scale': 1.5, 'shape': 0}]
    n = len(params)

    # approximate
    bids, costs = solve(support, params)

    print("Estimated lower bound on bids: %r" % bids[0])

    # save the results in a file
    with open('fsm.out', 'wt') as f:
        labels = ['support', 'params', 'bids'] + ['costs_{}'.format(i) for i in range(n)]
        labels = ' '.join(labels)
        values = [support, params, bids.tolist()] + [c.tolist() for c in costs]
        values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
        f.write(labels)
        f.write('\n')
        f.write(values)
