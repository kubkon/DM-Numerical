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
    sigma = w / 2
    params.append({'mu': mu, 'sigma': sigma})

# compute approximations
dm_bids, dm_costs = dm.solve(w, reputations)
bajari_bids, bajari_costs = bajari.solve(support, params)

# plot
for i in range(n):
    plt.figure()
    plt.plot(dm_costs[i], dm_bids, '-r')
    plt.plot(bajari_costs[i], bajari_bids, '.-b')
    plt.grid()
    plt.savefig('compare_' + str(i) + '.pdf')
