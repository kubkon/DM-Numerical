from ppm_b import solve


# set the scenario
support = [2.0, 8.0]
params = [{'mu': 4.0, 'sigma': 1.5},
          {'mu': 5.0, 'sigma': 1.5},
          {'mu': 6.0, 'sigma': 1.5}]

# infer number of bidders
n = len(params)

# set initial conditions for the PPM algorithm
k = 4
K = 4
poly_coeffs = [[1e-1 for i in range(k)] for j in range(n)]
b_lower = support[0] + 2
size_box = [1e-4 for i in range(k*n + 1)]

# run the PPM algorithm until k >= K
while True:
    b_lower, poly_coeffs = solve(b_lower,
                                 support,
                                 params,
                                 poly_coeffs,
                                 size_box=size_box,
                                 granularity=10)

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


print("Estimated lower bound on bids: %r" % b_lower)
print("Coefficients: %s" % poly_coeffs)
