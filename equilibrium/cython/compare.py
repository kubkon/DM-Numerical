import argparse
import itertools as its
import numpy as np
import scipy.integrate as si
import scipy.stats as ss

import bajari.fsm.main as bajari
#import dm.fsm.main as dm
import dm.ppm.main as dm
import util.util as util


# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models")
parser.add_argument('--w', type=float, help='price weight')
parser.add_argument('--reps', action='append', type=float, help='reputation array')
args = parser.parse_args()
w = args.w
reputations = np.array(args.reps)

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
bajari_params = []

for i in np.arange(n):
    location = lowers[i] + w / 2
    scale = w / 4
    bajari_params.append({'location': location, 'scale': scale})

# compute approximations
dm_bids, dm_costs = dm.solve_(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, bajari_params)

# ensure costs are monotonically increasing
dm_costs, dm_bids = util.ensure_monotonicity(dm_costs, dm_bids)
bajari_costs, bajari_bids = util.ensure_monotonicity(bajari_costs, bajari_bids)

# compute expected utilities for both auctions
# 1. DM
dm_params = [{'loc': lowers[i], 'scale': w} for i in np.arange(n)]
dm_cdfs = [ss.uniform(**p) for p in dm_params]
dm_exp_utilities = util.compute_expected_utilities(dm_bids, dm_costs, dm_cdfs)

# 2. Bajari
bajari_cdfs = []
for p in bajari_params:
    loc = p['location']
    scale = p['scale']
    a = (support[0] - loc) / scale
    b = (support[1] - loc) / scale
    bajari_cdfs.append(ss.truncnorm(a, b, loc=loc, scale=scale))
bajari_exp_utilities = util.compute_expected_utilities(bajari_bids, bajari_costs, bajari_cdfs)

# interpolate (using splines) expected utility functions
utils = []
for i in np.arange(n):
    # fit
    dm_exp_func = util.csplinefit(dm_costs[i], dm_exp_utilities[i])
    bajari_exp_func = util.csplinefit(bajari_costs[i], bajari_exp_utilities[i])

    # compute KS statistics
    costs = np.linspace(dm_costs[i][0], min(dm_costs[i][-1], bajari_costs[i][-1]), 1000)

    # compute ex-ante (average) expected utility
    dm_util = si.quad(lambda x: dm_exp_func(x) * dm_cdfs[i].pdf(x), costs[0], costs[-1])
    bajari_util = si.quad(lambda x: bajari_exp_func(x) * bajari_cdfs[i].pdf(x), costs[0], costs[-1])
    utils.append(bajari_util[0] / dm_util[0] * 100)

# interpolate bidding functions
dm_bid_funcs     = []
bajari_bid_funcs = []

for i in np.arange(n):
    # fit
    dm_bid_func     = util.csplinefit(dm_costs[i], dm_bids)
    bajari_bid_func = util.csplinefit(bajari_costs[i], bajari_bids)

    dm_bid_funcs.append(dm_bid_func)
    bajari_bid_funcs.append(bajari_bid_func)

# sample and generate prices for each auction
size = 10000
dm_sampled_costs = []
dm_params = [{'loc': dm_costs[i][0], 'scale': dm_costs[i][-1]-dm_costs[i][0]} for i in np.arange(n)]
for p in dm_params:
    loc   = p['loc']
    scale = p['scale']
    dm_sampled_costs.append(ss.uniform.rvs(loc=loc, scale=scale, size=size))

bajari_sampled_costs = []
for p,i in zip(bajari_params, np.arange(n)):
    loc   = p['location']
    scale = p['scale']
    a     = (bajari_costs[i][0] - loc) / scale
    b     = (bajari_costs[i][-1] - loc) / scale
    bajari_sampled_costs.append(ss.truncnorm.rvs(a, b, loc=loc, scale=scale, size=size))

dm_prices = []
for costs in zip(*dm_sampled_costs):
    bids = [dm_bid_funcs[i](costs[i]) for i in np.arange(n)]
    dm_prices.append(min(bids))

bajari_prices = []
for costs in zip(*bajari_sampled_costs):
    bids = [bajari_bid_funcs[i](costs[i]) for i in np.arange(n)]
    bajari_prices.append(min(bids))

print([utils, (np.mean(bajari_prices) / np.mean(dm_prices)) * 100])

