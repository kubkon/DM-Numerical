import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as ss

import bajari.fsm.main as bajari
import dm.fsm.main as dm
from bajari.dists.main import skewnormal
from util import compute_expected_utilities, fit_curve, ks_statistic

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


def compare(w, reputations, locations, scales, shapes):
    # estimate lower and upper extremities
    n = reputations.size
    lowers = np.empty(n, dtype=np.float)
    uppers = np.empty(n, dtype=np.float)
    for i in np.arange(n):
        lowers[i] = (1-w) * reputations[i]
        uppers[i] = (1-w) * reputations[i] + w

    # approximate the scenario as common support, differing
    # normal distributions
    support = [lowers[0], uppers[1]]
    bajari_params = [{'location': loc, 'scale': scale, 'shape': shape} for loc, scale, shape in zip(locations, scales, shapes)]

    # compute approximations
    dm_bids, dm_costs = dm.solve(w, reputations)
    bajari_bids, bajari_costs = bajari.solve(support, bajari_params)

    # compute expected utilities for both auctions
    # 1. DM
    dm_params = [{'loc': lowers[i], 'scale': w} for i in np.arange(n)]
    dm_exp_utilities = compute_expected_utilities(dm_bids, dm_costs, ss.uniform, dm_params)

    # 2. Bajari
    bajari_exp_utilities = compute_expected_utilities(bajari_bids, bajari_costs, skewnormal, bajari_params)

    # fit polynomial curve to expected utility functions and
    # compute KS statistic (distortion between the expected utilities)
    dm_exp_funcs = []
    bajari_exp_funcs = []
    ks_values = []

    for i in np.arange(n):
        # fit
        dm_exp_func = fit_curve(dm_costs[i], dm_exp_utilities[i], degree=5)
        bajari_exp_func = fit_curve(bajari_costs[i], bajari_exp_utilities[i], degree=5)

        dm_exp_funcs.append(dm_exp_func)
        bajari_exp_funcs.append(bajari_exp_func)

        # compute KS statistics
        costs = np.linspace(dm_costs[i][0], dm_costs[i][-1], 1000)
        _, ks_value = ks_statistic(costs, bajari_exp_func(costs), dm_exp_func(costs))
        ks_values.append(ks_value)

    return ks_values


# initialize the scenario
w = 0.5
reputations = [0.25, 0.75]

lowers = [(1-w) * r for r in reputations]
uppers = [l + w for l in lowers]

locations = [l + w / 2 for l in lowers]
scales = np.linspace(w/8, w, 100)
shapes = [0, 0]

ks_values = []

for s in scales:
    ks = compare(w, np.array(reputations), locations, [s, s], shapes)
    ks_values.append(ks)
    
# plot
plt.figure()
ks = list(zip(*ks_values))
plt.plot(scales, ks[0], '.r')
plt.plot(scales, ks[1], '.b')
plt.xlabel(r"Variance, $\rho^2$")
plt.ylabel(r"KS Statistic")
plt.grid()
plt.savefig('compare.pdf')
