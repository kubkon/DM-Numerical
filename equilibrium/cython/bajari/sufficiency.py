import argparse
import csv
import functools as fts
import itertools as its
from math import erf, sqrt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib import rc

csv.field_size_limit(1000000000)

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


def verify_sufficiency(costs, bids, support, cdfs, step=100):
    n = len(cdfs)
    best_responses = []
    sampled_costs = [c[::step] for c in costs]
    for i in range(n):
        best_response = []
        for c in sampled_costs[i]:
            feasible_bids = np.linspace(c, support[1], 100)
            utility = []
            for b in feasible_bids:
                if b < bids[0]:
                    utility += [(b, (b-c))]
                else:
                    diffs = list(map(lambda x: abs(x-b), bids))
                    index = diffs.index(min(diffs))
                    indexes = [j for j in range(n) if j != i]
                    probability = fts.reduce(lambda x,y: x*y, [(1-cdfs[i].cdf(costs[i][index])) for i in indexes], 1)
                    utility += [(b, (b-c)*probability)]
            best_response += [max(utility, key=lambda x: x[1])[0]]
        best_responses += [best_response]
    return sampled_costs, best_responses


### Parse command line arguments
parser = argparse.ArgumentParser(description="Bajari common priors -- sufficiency analyzer")
parser.add_argument('file_name', help='file with approximation results')
args = parser.parse_args()
file_name = args.file_name

# Read data from file
data_in = {}
with open(file_name, 'rt') as f:
    f_reader = csv.DictReader(f, delimiter=',')
    for row in f_reader:
        for key in row:
            data_in.setdefault(key, []).append(float(row[key]))

# Parse data
n = len(data_in) - 1
costs = [data_in['costs_{}'.format(i)] for i in range(n)]
bids = data_in['bids']
support = [2.0, 8.0]
params = [{'mu': 4.0, 'sigma': 1.5},
          {'mu': 5.0, 'sigma': 1.5},
          {'mu': 6.0, 'sigma': 1.5}]

# Verify sufficiency
cdfs = [ss.truncnorm((support[0] - p['mu']) / p['sigma'], (support[1] - p['mu']) / p['sigma'], loc=p['mu'], scale=p['sigma']) for p in params]
step = len(bids) // 35
s_costs, s_bids = verify_sufficiency(costs, bids, support, cdfs, step=step)

# Plot
styles = ['b', 'r--', 'g:', 'm-.']
colors = ['b.', 'r.', 'g.', 'm.']

plt.figure()
sts = its.cycle(styles)
for c in costs:
  plt.plot(c, bids, next(sts))
plt.grid()
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}_i$")
labels = ['Network operator {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')

plt.figure()
sts = its.cycle(styles)
clss = its.cycle(colors)
for c, sc, sb in zip(costs, s_costs, s_bids):
  plt.plot(c, bids, next(sts))
  plt.plot(sc, sb, next(clss))
plt.grid()
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}_i$")
labels_1 = ['NO {}'.format(i) for i in range(1, n+1)]
labels_2 = ['NO {}: Best response'.format(i) for i in range(1, n+1)]
labels = fts.reduce(lambda acc,x: acc + [x[0], x[1]], zip(labels_1, labels_2), [])
plt.legend(labels, loc='upper left')
plt.savefig('sufficiency.pdf')

for i, c, sc, sb in zip(range(1, n+1), costs, s_costs, s_bids):
  plt.figure()
  plt.plot(c, bids, 'b')
  plt.plot(sc, sb, 'r.')
  plt.grid()
  plt.xlabel(r"Cost-hat, $\hat{c}_i$")
  plt.ylabel(r"Bid-hat, $\hat{b}_i$")
  plt.savefig('sufficiency_{}.pdf'.format(i))
