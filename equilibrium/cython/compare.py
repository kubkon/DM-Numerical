import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import bajari.fsm.main as bajari
import dm.fsm.main as dm

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
support = [lowers[0], uppers[1]]
params = []

for i in np.arange(n):
    mu = lowers[i] + w / 2
    sigma = w / 4
    params.append({'mu': mu, 'sigma': sigma})
print(params)
print(uppers)

# compute approximations
dm_bids, dm_costs = dm.solve(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, params)

# chuck away bids & costs that are irrelevant
t_bajari_costs, t_bajari_bids = [], []

for i in np.arange(n):
    costs = bajari_costs[i]
    low, high = dm_costs[i][0], dm_costs[i][-1]
    t_bids = []
    t_costs = []

    for j in np.arange(costs.size):
        if costs[j] < low or costs[j] > high:
            continue

        t_bids.append(bajari_bids[j])
        t_costs.append(costs[j])

    t_bajari_bids.append(np.array(t_bids))
    t_bajari_costs.append(np.array(t_costs))

# plots
plt.figure()
styles = ['-r', '--b']
style = its.cycle(styles)
for i in range(n):
    plt.plot(dm_costs[i], dm_bids, next(style))
plt.grid()
plt.legend(['Bidder ' + str(i) for i in range(n)], loc='upper left')
plt.savefig('dm.pdf')

plt.figure()
for i in range(n):
    plt.plot(bajari_costs[i], bajari_bids, next(style))
plt.grid()
plt.legend(['Bidder ' + str(i) for i in range(n)], loc='upper left')
plt.savefig('bajari.pdf')

for i in range(n):
    plt.figure()
    plt.plot(dm_costs[i], dm_bids, '-r')
    plt.plot(t_bajari_costs[i], t_bajari_bids[i], '--b')
    plt.grid()
    plt.legend(list(map(lambda x: x + str(i), ['DM Bidder ', 'Bajari Bidder '])), loc='upper left')
    plt.savefig('compare_' + str(i) + '.pdf')
