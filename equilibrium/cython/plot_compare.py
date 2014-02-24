import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as ss

import bajari.fsm.main as bajari
import dm.fsm.main as dm
import util.util as util

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


# set the scenario
w = 0.6 + 1e-6
reputations = np.array([0.5, 0.75], dtype=np.float)
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
params = []

for i in np.arange(n):
    location = lowers[i] + w / 2
    scale = w / 4
    params.append({'location': location, 'scale': scale})

# compute approximations
dm_bids, dm_costs = dm.solve(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, params)

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
for p in params:
    loc = p['location']
    scale = p['scale']
    a = (support[0] - loc) / scale
    b = (support[1] - loc) / scale
    cdfs.append(ss.truncnorm(a, b, loc=loc, scale=scale))
bajari_exp_utilities = util.compute_expected_utilities(bajari_bids, bajari_costs, cdfs)

# interpolate using cubic splines and
# compute KS statistic (distortion between the expected utilities)
dm_exp_funcs = []
bajari_exp_funcs = []
ks_coords = []
ks_values = []
common_costs = []

for i in np.arange(n):
    # fit
    dm_exp_func = util.csplinefit(dm_costs[i], dm_exp_utilities[i])
    bajari_exp_func = util.csplinefit(bajari_costs[i], bajari_exp_utilities[i])

    dm_exp_funcs.append(dm_exp_func)
    bajari_exp_funcs.append(bajari_exp_func)

    # compute KS statistics
    # we pick min upper cost since, due to the nature of FSM, the generated numerical solution
    # might be a strict subset of the support, and hence, we might exceed the interpolation
    # range; the interpolation and computation of K-S statistic is as such unaffected
    costs = np.linspace(dm_costs[i][0], min(dm_costs[i][-1], bajari_costs[i][-1]), 1000)
    common_costs.append(costs)
    ks_x, ks_value = util.ks_statistic(costs, bajari_exp_func(costs), dm_exp_func(costs))
    y1 = bajari_exp_func(np.array([ks_x]))
    y2 = dm_exp_func(np.array([ks_x]))
    coords = (ks_x, y1, y2) if y1 < y2 else (ks_x, y2, y1)
    ks_coords.append(coords)
    ks_values.append(ks_value)

common_costs = np.array(common_costs)

# plots
# 1. equilibrium bids
plt.figure()
linestyles = ['-r', '--b']
markerstyles = ['.r', 'xb']
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
plt.legend(["Bidder 1", "Bidder 1: Interpolated", "Bidder 2", "Bidder 2: Interpolated"])
plt.savefig('dm_exp_utilities.pdf')

plt.figure()
#for i in range(n):
#    plt.plot(bajari_costs[i][::10], bajari_exp_utilities[i][::10], next(markercycle))
#    plt.plot(bajari_costs[i], bajari_exp_funcs[i](bajari_costs[i]), next(linecycle))
plt.plot(bajari_costs[0][::200], bajari_exp_utilities[0][::200], next(markercycle))
plt.plot(bajari_costs[0], bajari_exp_funcs[0](bajari_costs[0]), next(linecycle))
plt.plot(np.concatenate((bajari_costs[1][:3], bajari_costs[1][5:20:20], bajari_costs[1][20::200])), np.concatenate((bajari_exp_utilities[1][:3], bajari_exp_utilities[1][5:20:20], bajari_exp_utilities[1][20::200])), next(markercycle))
plt.plot(bajari_costs[1], bajari_exp_funcs[1](bajari_costs[1]), next(linecycle))

plt.grid()
plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Expected utility')
plt.xlim(support)
plt.legend(["Bidder 1", "Bidder 1: Interpolated", "Bidder 2", "Bidder 2: Interpolated"])
plt.savefig('bajari_exp_utilities.pdf')

# 3. expected utilities for corresponding bidders across two auctions
for i in range(n):
    plt.figure()
    plt.plot(common_costs[i], dm_exp_funcs[i](common_costs[i]), '-r')
    plt.plot(common_costs[i], bajari_exp_funcs[i](common_costs[i]), '--b')
    plt.annotate("",
                xy=(ks_coords[i][0], ks_coords[i][1]),
                xycoords="data",
                xytext=(ks_coords[i][0], ks_coords[i][2]),
                textcoords="data",
                arrowprops=dict(arrowstyle="<->"),
                fontsize=8)
    plt.annotate(r'$D_%d = %f$' % (i+1, ks_values[i]),
                 xy=(ks_coords[i][0], ks_coords[i][2] + 0.001),
                 xycoords="data",
                 fontsize=14)
    plt.grid()
    plt.xlabel(r'Cost, $c_%d$' % (i+1))
    plt.ylabel(r'Expected utility, $\Pi_%d(c_%d)$' % (i+1, i+1))
    plt.legend(list(map(lambda x: x + str(i+1), ['DMP Bidder ', 'CP Bidder '])), loc='upper right')
    plt.savefig('compare_' + str(i+1) + '.pdf')

# 4. pdfs of the scenario
plt.figure()
xs = np.linspace(support[0], support[1], 1000)

for f in cdfs:
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

for f in cdfs:
    ys = [f.cdf(x) for x in xs]
    plt.plot(xs, ys, next(linecycle))

plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Cumulative distribution function, $F_i(c_i)$')
plt.grid()
plt.xlim(support)
plt.legend(legend)
plt.savefig('cdfs_scenario.pdf')
