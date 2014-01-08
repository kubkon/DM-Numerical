import bajari.ppm.ppm_internal as ppm_internal


def solve(support, params):
    # infer number of bidders
    n = len(params)

    # set initial conditions for the PPM algorithm
    k = 3
    K = 4
    poly_coeffs = [[1e-2 for i in range(k)] for j in range(n)]
    b_lower = support[0]
    size_box = [1e-1 for i in range(k*n + 1)]

    # run the PPM algorithm until k >= K
    while True:
        b_lower, poly_coeffs = ppm_internal.solve(b_lower,
                                                  support,
                                                  params,
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
        size_box = [1e-2 for i in range(k*n + 1)]

    return b_lower, support[1], poly_coeffs


if __name__ == '__main__':
    # set the scenario
    support = [2.0, 8.0]
    params = [{'loc': 4.0, 'scale': 1.5, 'shape': 0},
              {'loc': 5.0, 'scale': 1.5, 'shape': 0},
              {'loc': 6.0, 'scale': 1.5, 'shape': 0}]
    n = len(params)

    # approximate
    b_lower, b_upper, poly_coeffs = solve(support, params)

    print("Estimated lower bound on bids: %r" % b_lower)
    print("Coefficients: %s" % poly_coeffs)

    # save the results in a file
    with open('ppm.out', 'wt') as f:
        labels = ['n', 'b_lower', 'b_upper'] + ['cs_{}'.format(i) for i in range(n)]
        labels = ' '.join(labels)
        values = [n, b_lower, b_upper] + [c for c in poly_coeffs]
        values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
        f.write(labels)
        f.write('\n')
        f.write(values)
