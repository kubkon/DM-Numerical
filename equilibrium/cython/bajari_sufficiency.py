import argparse
import csv
from itertools import cycle, chain

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib import rc

from util import verify_sufficiency

csv.field_size_limit(1000000000)

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


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
costs = np.array([data_in['costs_{}'.format(i)] for i in range(n)])
bids = np.array(data_in['bids'])
support = [2.0, 8.0]
params = [{'mu': 4.0, 'sigma': 1.5},
          {'mu': 5.0, 'sigma': 1.5},
          {'mu': 6.0, 'sigma': 1.5}]

# Verify sufficiency
cdfs = []
for p in params:
  cdfs.append(ss.truncnorm((support[0] - p['mu']) / p['sigma'], (support[1] - p['mu']) / p['sigma'], loc=p['mu'], scale=p['sigma']))

step = len(bids) // 35
s_costs, s_bids = verify_sufficiency(costs, bids, support[1], cdfs, step=step)

# Plot
styles = ['b', 'r--', 'g:', 'm-.']
colors = ['b.', 'r.', 'g.', 'm.']

plt.figure()
sts = cycle(styles)
for c in costs:
  plt.plot(c, bids, next(sts))
plt.grid()
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}_i$")
labels = ['Network operator {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')

plt.figure()
sts = cycle(styles)
clss = cycle(colors)
for c, sc, sb in zip(costs, s_costs, s_bids):
  plt.plot(c, bids, next(sts))
  plt.plot(sc, sb, next(clss))
plt.grid()
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}_i$")
labels_1 = ['NO {}'.format(i) for i in range(1, n+1)]
labels_2 = ['NO {}: Best response'.format(i) for i in range(1, n+1)]
labels = list(chain.from_iterable(zip(labels_1, labels_2)))
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
