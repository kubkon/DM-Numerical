import argparse
import itertools as its
from functools import reduce
import numpy as np
import scipy.integrate as si
import scipy.stats as ss

import bajari.fsm.main as bajari
import util.util as util


# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models")
parser.add_argument('--w', type=float, help='price weight')
parser.add_argument('--reps', action='append', type=float, help='reputation array')
parser.add_argument('--param', type=float, help='param')
args = parser.parse_args()
w = args.w
reputations = np.array(args.reps)
param = args.param

# load appropriate version of FSM method
n = reputations.size
if n == 2:
    import dm.fsm.main as dm
else:
    import dm.efsm.main as dm

# estimate lower and upper extremities
lowers = np.empty(n, dtype=np.float)
uppers = np.empty(n, dtype=np.float)
for i in np.arange(n):
    lowers[i] = (1-w) * reputations[i]
    uppers[i] = (1-w) * reputations[i] + w

# pick parameters for the common support case
support = [lowers[0], uppers[-1]]
bajari_params = []

for i in np.arange(n):
    location = lowers[i] + w / 2
    scale = w / 4
    bajari_params.append({'location': location, 'scale': scale})

# compute approximations
if n == 2:
    dm_bids, dm_costs = dm.solve(w, reputations)
else:
    dm_bids, dm_costs = dm.solve(w, reputations, param=param)

bajari_bids, bajari_costs = bajari.solve(support, bajari_params)

# ensure costs are monotonically increasing
dm_costs, dm_bids = util.ensure_monotonicity(dm_costs, dm_bids)
bajari_costs, bajari_bids = util.ensure_monotonicity(bajari_costs, bajari_bids)

# interpolate bidding functions and their inverses
dm_bid_funcs     = []
bajari_bid_funcs = []
dm_inverses      = []
bajari_inverses  = []
for i in np.arange(n):
    # fit
    dm_bid_func     = util.csplinefit(dm_costs[i], dm_bids)
    dm_inverse      = util.csplinefit(dm_bids, dm_costs[i])
    dm_bid_funcs.append(dm_bid_func)
    dm_inverses.append(dm_inverse)

    bajari_bid_func = util.csplinefit(bajari_costs[i], bajari_bids)
    bajari_inverse  = util.csplinefit(bajari_bids, bajari_costs[i])
    bajari_bid_funcs.append(bajari_bid_func)
    bajari_inverses.append(bajari_inverse)
    
# compute ex-ante expected utilities
params = [{'loc': lowers[i], 'scale': w} for i in np.arange(n)]
cdfs   = [ss.uniform(**p) for p in params]

dm_utils     = []
bajari_utils = []

dm_exp_funcs     = []
bajari_exp_funcs = []
common_costs     = []
for i in np.arange(n):
    # common costs
    costs = np.linspace(dm_costs[i][0], min(dm_costs[i][-1], bajari_costs[i][-1]), 1000)
    common_costs.append(costs)

    # derive expected utility functions
    js = [j for j in np.arange(n) if j != i]
    def dm_exp_util(x):
        bid   = dm_bid_funcs[i](x)
        probs = [(1 - cdfs[j].cdf(dm_inverses[j](bid))) for j in js]
        return (bid - x) * reduce(lambda x,y: x*y, probs)
    
    dm_exp_funcs.append(dm_exp_util)

    def bajari_exp_util(x):
        bid   = bajari_bid_funcs[i](x) 
        probs = [(1 - cdfs[j].cdf(bajari_inverses[j](bid))) for j in js]
        return (bid - x) * reduce(lambda x,y: x*y, probs)

    bajari_exp_funcs.append(bajari_exp_util)

    # compute ex-ante expected utilities
    dm_util     = si.quad(lambda x: dm_exp_util(x) * cdfs[i].pdf(x), costs[0], costs[-1])
    bajari_util = si.quad(lambda x: bajari_exp_util(x) * cdfs[i].pdf(x), costs[0], costs[-1])
    dm_utils.append(dm_util[0])
    bajari_utils.append(bajari_util[0])

# sample and generate prices for each auction
size = 10000
sampled_costs = []
params = [{'loc': dm_costs[i][0], 'scale': dm_costs[i][-1]-dm_costs[i][0]} for i in np.arange(n)]
for p in params:
    loc   = p['loc']
    scale = p['scale']
    sampled_costs.append(ss.uniform.rvs(loc=loc, scale=scale, size=size))

dm_prices = []
for costs in zip(*sampled_costs):
    bids = [dm_bid_funcs[i](costs[i]) for i in np.arange(n)]
    dm_prices.append(min(bids))

bajari_prices = []
for costs in zip(*sampled_costs):
    bids = [bajari_bid_funcs[i](costs[i]) for i in np.arange(n)]
    bajari_prices.append(min(bids))

print(dm_utils)
print(bajari_utils)
print(np.mean(dm_prices))
print(np.mean(bajari_prices))
