import itertools as its
import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as ss
import scipy.integrate as si

import bajari.fsm.main as bajari
import dm.fsm.main as dm
import util.util as util

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


# set the scenario
w = 0.5
reputations = np.array([0.25, 0.75], dtype=np.float)
n = reputations.size

# estimate lower and upper extremities
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
    j = (i+1) % 2
    def dm_exp_util(x):
        bid  = dm_bid_funcs[i](x) 
        prob = 1 - cdfs[j].cdf(dm_inverses[j](bid))
        return (bid - x) * prob
    
    dm_exp_funcs.append(dm_exp_util)

    def bajari_exp_util(x):
        bid  = bajari_bid_funcs[i](x) 
        prob = 1 - cdfs[j].cdf(bajari_inverses[j](bid))
        return (bid - x) * prob

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

# plots
for i in np.arange(n):
    dm_f     = dm_exp_funcs[i]
    bajari_f = bajari_exp_funcs[i]

    plt.figure()
    plt.plot(common_costs[i], [dm_f(x) for x in common_costs[i]], '-r')
    plt.plot(common_costs[i], [bajari_f(x) for x in common_costs[i]], '--b')
    xlabel = "Cost, $c_{}$".format(i+1)
    ylabel = "Expected utility, $\Pi_{}(c_{})$".format(i+1, i+1)
    plt.xlabel(r"" + xlabel)
    plt.ylabel(r"" + ylabel)
    legend = ["DMP Bidder {}".format(i+1), "CP Bidder {}".format(i+1)]
    plt.legend(legend)
    plt.grid()
    plt.savefig('expected_utilities_{}.pdf'.format(i+1))

