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
parser.add_argument('--size', type=int, default=100, help='population size')
args = parser.parse_args()
w = args.w
reputations = np.array(args.reps)
size = args.size

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
dm_bids, dm_costs = dm.solve(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, bajari_params)

# ensure costs are monotonically increasing
dm_costs, dm_bids = util.ensure_monotonicity(dm_costs, dm_bids)
bajari_costs, bajari_bids = util.ensure_monotonicity(bajari_costs, bajari_bids)

# interpolate bidding functions
dm_bid_funcs = []
bajari_bid_funcs = []

for i in np.arange(n):
    # fit
    dm_bid_func     = util.csplinefit(dm_costs[i], dm_bids)
    bajari_bid_func = util.csplinefit(bajari_costs[i], bajari_bids)

    dm_bid_funcs.append(dm_bid_func)
    bajari_bid_funcs.append(bajari_bid_func)

# sample and generate prices for each auction
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
prices = []
for costs in zip(*dm_sampled_costs):
    bids = [dm_bid_funcs[i](costs[i]) for i in np.arange(n)]
    dm_prices.append(min(bids))
    j = np.argmin(bids)
    prices.append((bids[j] - (1-w)*reputations[j]) / w)

bajari_prices = []
for costs in zip(*bajari_sampled_costs):
    bids = [bajari_bid_funcs[i](costs[i]) for i in np.arange(n)]
    bajari_prices.append(min(bids))

print(np.mean(dm_prices), np.mean(bajari_prices))
print(np.mean(prices))

