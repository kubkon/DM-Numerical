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
dm_utils         = []
bajari_utils     = []
common_costs     = []
dm_exp_funcs     = []
bajari_exp_funcs = []

for i in np.arange(n):
    # fit
    dm_exp_func = util.csplinefit(dm_costs[i], dm_exp_utilities[i])
    bajari_exp_func = util.csplinefit(bajari_costs[i], bajari_exp_utilities[i])

    costs = np.linspace(dm_costs[i][0], min(dm_costs[i][-1], bajari_costs[i][-1]), 1000)

    # compute ex-ante (average) expected utility
    dm_util = si.quad(lambda x: dm_exp_func(x) * dm_cdfs[i].pdf(x), costs[0], costs[-1])
    bajari_util = si.quad(lambda x: bajari_exp_func(x) * bajari_cdfs[i].pdf(x), costs[0], costs[-1])
    dm_utils.append(dm_util[0])
    bajari_utils.append(bajari_util[0])
    
    dm_exp_funcs.append(dm_exp_func)
    bajari_exp_funcs.append(bajari_exp_func)
    common_costs.append(costs)

common_costs = np.array(common_costs)

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

print("Expected prices: DMP={}, CP={}".format(np.mean(dm_prices), np.mean(bajari_prices)))
print("Ex-ante expected utilities: DMP={}, CP={}".format(dm_utils, bajari_utils))

# plots
# 1. equilibrium bids
plt.figure()
linestyles = ['-r', '--b', '-.g']
markerstyles = ['.r', 'xb', '+g']
linecycle = its.cycle(linestyles)
markercycle = its.cycle(markerstyles)
legend = ['Bidder ' + str(i) for i in range(1, n+1)]
for i in range(n):
    plt.plot(dm_costs[i], dm_bids, next(linecycle))
plt.grid()
plt.xlabel(r'Cost-hat, $\hat{c}_i$')
plt.ylabel(r'Bid-hat, $\hat{b}_i$')
plt.xlim(support)
plt.ylim([dm_bids[0], dm_bids[-1]])
plt.legend(legend, loc='upper left')
plt.savefig('dm.pdf')

plt.figure()
for i in range(n):
    plt.plot(bajari_costs[i], bajari_bids, next(linecycle))
plt.grid()
plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Bid, $b_i$')
plt.xlim(support)
plt.ylim([bajari_bids[0], bajari_bids[-1]])
plt.legend(legend, loc='upper left')
plt.savefig('bajari.pdf')

# 2. expected utilities
plt.figure()
for i in range(n):
    plt.plot(dm_costs[i][::200], dm_exp_utilities[i][::200], next(markercycle))
    plt.plot(dm_costs[i], dm_exp_funcs[i](dm_costs[i]), next(linecycle))
plt.grid()
plt.xlabel(r'Cost-hat, $\hat{c}_i$')
plt.ylabel(r'Expected utility')
plt.xlim(support)
labels = [["Bidder {}".format(i+1), "Bidder {}: Interpolated".format(i+1)] for i in range(n)] 
plt.legend(list(its.chain(*labels)))
plt.savefig('dm_exp_utilities.pdf')

plt.figure()
for i in range(n):
    plt.plot(bajari_costs[i][::200], bajari_exp_utilities[i][::200], next(markercycle))
    plt.plot(bajari_costs[i], bajari_exp_funcs[i](bajari_costs[i]), next(linecycle))

plt.grid()
plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Expected utility')
plt.xlim(support)
plt.legend(list(its.chain(*labels)))
plt.savefig('bajari_exp_utilities.pdf')

# 3. expected utilities for corresponding bidders across two auctions
for i in range(n):
    plt.figure()
    plt.plot(common_costs[i], dm_exp_funcs[i](common_costs[i]), '-r')
    plt.plot(common_costs[i], bajari_exp_funcs[i](common_costs[i]), '--b')
    plt.grid()
    plt.xlabel(r'Cost, $c_%d$' % (i+1))
    plt.ylabel(r'Expected utility, $\Pi_%d(c_%d)$' % (i+1, i+1))
    plt.legend(list(map(lambda x: x + str(i+1), ['DMP Bidder ', 'CP Bidder '])), loc='upper right')
    plt.savefig('compare_' + str(i+1) + '.pdf')

# 4. pdfs of the scenario
plt.figure()
xs = np.linspace(support[0], support[1], 1000)

for f in bajari_cdfs:
    ys = [f.pdf(x) for x in xs]
    plt.plot(xs, ys, next(linecycle))

plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Probability density function, $f_i(c_i)$')
plt.grid()
plt.xlim(support)
plt.legend(legend)
plt.savefig('pdfs_scenario.pdf')

# 5. cdfs of the scenario
plt.figure()

for f in bajari_cdfs:
    ys = [f.cdf(x) for x in xs]
    plt.plot(xs, ys, next(linecycle))

plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Cumulative distribution function, $F_i(c_i)$')
plt.grid()
plt.xlim(support)
plt.legend(legend)
plt.savefig('cdfs_scenario.pdf')
