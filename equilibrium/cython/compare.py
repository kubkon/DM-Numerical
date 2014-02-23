import argparse
import itertools as its
import numpy as np
import scipy.stats as ss

import bajari.fsm.main as bajari
import dm.fsm.main as dm
import util.util as util


# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models")
parser.add_argument('--w', type=float, help='price weight')
parser.add_argument('--reps', action='append', type=float, help='reputation array')
parser.add_argument('--locs', action='append', type=float, help='distributions locations')
parser.add_argument('--scales', action='append', type=float, help='distributions scales')
args = parser.parse_args()
w = args.w
reputations = np.array(args.reps)
locations = np.array(args.locs)
scales = np.array(args.scales)

# estimate lower and upper extremities
n = reputations.size
lowers = np.empty(n, dtype=np.float)
uppers = np.empty(n, dtype=np.float)
for i in np.arange(n):
    lowers[i] = (1-w) * reputations[i]
    uppers[i] = (1-w) * reputations[i] + w

# approximate the scenario as common support, differing
# normal distributions
support = [lowers[0], uppers[-1]]
bajari_params = [{'location': loc, 'scale': scale} for loc, scale in zip(locations, scales)]

# compute approximations
dm_bids, dm_costs = dm.solve(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, bajari_params)

# ensure costs are monotonically increasing
dm_costs, dm_bids = util.ensure_monotonicity(dm_costs, dm_bids)
bajari_costs, bajari_bids = util.ensure_monotonicity(bajari_costs, bajari_bids)

# compute expected utilities for both auctions
# 1. DM
dm_params = [{'loc': lowers[i], 'scale': w} for i in np.arange(n)]
cdfs = [ss.uniform(**p) for p in dm_params]
dm_exp_utilities = util.compute_expected_utilities(dm_bids, dm_costs, cdfs)

# 2. Bajari
cdfs = []
for p in bajari_params:
    loc = p['location']
    scale = p['scale']
    a = (support[0] - loc) / scale
    b = (support[1] - loc) / scale
    cdfs.append(ss.truncnorm(a, b, loc=loc, scale=scale))
bajari_exp_utilities = util.compute_expected_utilities(bajari_bids, bajari_costs, cdfs)

# fit polynomial curve to expected utility functions and
# compute KS statistic (distortion between the expected utilities)
dm_exp_funcs = []
bajari_exp_funcs = []
ks_values = []

for i in np.arange(n):
    # fit
    dm_exp_func = util.csplinefit(dm_costs[i], dm_exp_utilities[i])
    bajari_exp_func = util.csplinefit(bajari_costs[i], bajari_exp_utilities[i])

    dm_exp_funcs.append(dm_exp_func)
    bajari_exp_funcs.append(bajari_exp_func)

    # compute KS statistics
    costs = np.linspace(dm_costs[i][0], dm_costs[i][-1], 1000)
    _, ks_value = util.ks_statistic(costs, bajari_exp_func(costs), dm_exp_func(costs))
    ks_values.append(ks_value)

print(ks_values)
