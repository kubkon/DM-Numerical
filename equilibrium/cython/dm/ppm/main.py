import numpy as np

from dm.common import upper_bound_bids
import dm.ppm.ppm_internal as ppm_internal


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

    # set initial conditions for the PPM algorithm
    k = 3
    K = 8
    poly_coeffs = [[1e-2 for i in range(k)] for j in range(n)]
    b_lower = lowers[1] + 1e-3
    size_box = [1e-1 for i in range(k*n + 1)]

    # run the PPM algorithm until k >= K
    while True:
        b_lower, poly_coeffs = ppm_internal.solve(b_lower,
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

    return b_lower, b_upper, poly_coeffs


if __name__ == "__main__":
    # set the scenario
    w = 0.85
    reputations = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float)
    n = reputations.size

    # approximate
    b_lower, b_upper, poly_coeffs = solve(w, reputations)

    print("Estimated lower bound on bids: %r" % b_lower)
    print("Coefficients: %s" % poly_coeffs)

    # save the results in a file
    with open('ppm.out', 'wt') as f:
        labels = ['w', 'reps', 'b_lower', 'b_upper'] + ['cs_{}'.format(i) for i in range(n)]
        labels = ' '.join(labels)
        values = [w, reputations, b_lower, b_upper] + [c for c in poly_coeffs]
        values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
        f.write(labels)
        f.write('\n')
        f.write(values)
