import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as ss

import bajari.fsm.main as bajari
import dm.fsm.main as dm

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


def estimate_winning_probs(costs, func, params):
    # initialize
    shape = costs.shape
    probs = np.empty(shape, np.float)
    n, m = shape[0], shape[1]

    # compute probabilities
    for i in np.arange(n):
        # get index of the competing bidder
        j = (i+1) % 2
        # extract cdf function
        prob_func = func(**params[j])

        for k in np.arange(m):
            probs[i][k] = 1 - prob_func.cdf(costs[i][k])

    return probs


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
support = [lowers[0], uppers[1]]
params = []

for i in np.arange(n):
    mu = lowers[i] + w / 4
    sigma = w / 4
    params.append({'mu': mu, 'sigma': sigma})

# compute approximations
dm_bids, dm_costs = dm.solve(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, params)

# compute probabilities of winning for both auctions
# 1. DM
dm_params = [{'loc': lowers[i], 'shape': w} for i in np.arange(n)]
dm_probs = estimate_winning_probs(dm_costs, ss.uniform, dm_params)

# 2. Bajari
bajari_params = [{'loc': p['mu'], 'shape': p['sigma']} for p in params]
bajari_probs = estimate_winning_probs(bajari_costs, ss.norm, bajari_params)

# compute expected utilities
dm_exp_utilities = np.empty(dm_costs.shape, np.float)
bajari_exp_utilities = np.empty(bajari_costs.shape, np.float)

for i in np.arange(n):
    dm_exp_utilities[i] = np.multiply((dm_bids - dm_costs[i]), dm_probs[i])
    bajari_exp_utilities[i] = np.multiply((bajari_bids - bajari_costs[i]), bajari_probs[i])

# chuck away costs & utilities that are irrelevant
t_bajari_costs, t_bajari_exps = [], []

for i in np.arange(n):
    exps = bajari_exp_utilities[i]
    costs = bajari_costs[i]
    low, high = dm_costs[i][0], dm_costs[i][-1]
    t_exps = []
    t_costs = []

    for j in np.arange(costs.size):
        if costs[j] < low or costs[j] > high:
            continue

        t_exps.append(exps[j])
        t_costs.append(costs[j])

    t_bajari_exps.append(np.array(t_exps))
    t_bajari_costs.append(np.array(t_costs))

# plots
# 1. equilibrium bids
plt.figure()
styles = ['-r', '--b']
style = its.cycle(styles)
legend = ['Bidder ' + str(i) for i in range(n)]
for i in range(n):
    plt.plot(dm_costs[i], dm_bids, next(style))
plt.grid()
plt.xlabel(r'Cost-hat, $\hat{c}_i$')
plt.ylabel(r'Bid-hat, $\hat{b}_i$')
plt.xlim(support)
plt.ylim([dm_bids[0], dm_bids[-1]])
plt.legend(legend, loc='upper left')
plt.savefig('dm.pdf')

plt.figure()
for i in range(n):
    plt.plot(bajari_costs[i], bajari_bids, next(style))
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
    plt.plot(dm_costs[i], dm_exp_utilities[i], next(style))
plt.grid()
plt.xlabel(r'Cost-hat, $\hat{c}_i$')
plt.ylabel(r'Expected utility')
plt.xlim(support)
plt.legend(legend)
plt.savefig('dm_exp_utilities.pdf')

plt.figure()
for i in range(n):
    plt.plot(bajari_costs[i], bajari_exp_utilities[i], next(style))
plt.grid()
plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Expected utility')
plt.xlim(support)
plt.legend(legend)
plt.savefig('bajari_exp_utilities.pdf')

# 3. expected utilities for corresponding bidders across two auctions
for i in range(n):
    plt.figure()
    plt.plot(dm_costs[i], dm_exp_utilities[i], '-r')
    plt.plot(t_bajari_costs[i], t_bajari_exps[i], '--b')
    plt.grid()
    plt.xlabel(r'Cost')
    plt.ylabel(r'Expected utility')
    plt.legend(list(map(lambda x: x + str(i), ['DM Bidder ', 'Bajari Bidder '])), loc='upper right')
    plt.savefig('compare_' + str(i) + '.pdf')

# 4. pdfs of the scenario
plt.figure()
xs = np.linspace(support[0], support[1], 1000)

funcs = []
for p in params:
  funcs.append(ss.truncnorm((support[0] - p['mu']) / p['sigma'], (support[1] - p['mu']) / p['sigma'], loc=p['mu'], scale=p['sigma']))

for f in funcs:
    ys = [f.pdf(x) for x in xs]
    plt.plot(xs, ys, next(style))

plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Probability density function, $f_i(c_i)$')
plt.grid()
plt.xlim(support)
plt.legend(legend)
plt.savefig('pdfs_scenario.pdf')

# 5. cdfs of the scenario
plt.figure()

for f in funcs:
    ys = [f.cdf(x) for x in xs]
    plt.plot(xs, ys, next(style))

plt.xlabel(r'Cost, $c_i$')
plt.ylabel(r'Cumulative distribution function, $F_i(c_i)$')
plt.grid()
plt.xlim(support)
plt.legend(legend)
plt.savefig('cdfs_scenario.pdf')
