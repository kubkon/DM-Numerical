import argparse
import ast
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
    f_reader = csv.DictReader(f, delimiter=' ')
    for row in f_reader:
        for key in row:
            data_in[key] = row[key]

# Parse data
support = ast.literal_eval(data_in['support'])
params = ast.literal_eval(data_in['params'])
n = len(params)
costs = np.array([ast.literal_eval(data_in['costs_{}'.format(i)]) for i in range(n)])
bids = np.array(ast.literal_eval(data_in['bids']))

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
plt.xlim(support)
plt.ylim([int(bids[0]), bids[-1]])
plt.xlabel(r"Cost, $c_i$")
plt.ylabel(r"Bid, $b_i$")
labels = ['Bidder {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')

plt.figure()
sts = cycle(styles)
clss = cycle(colors)
for c, sc, sb in zip(costs, s_costs, s_bids):
  plt.plot(c, bids, next(sts))
  plt.plot(sc, sb, next(clss))
plt.grid()
plt.xlim(support)
plt.ylim([int(bids[0]), bids[-1]])
plt.xlabel(r"Cost, $c_i$")
plt.ylabel(r"Bid, $b_i$")
labels_1 = ['Bidder {}'.format(i) for i in range(1, n+1)]
labels_2 = ['Bidder {}: Best response'.format(i) for i in range(1, n+1)]
labels = list(chain.from_iterable(zip(labels_1, labels_2)))
plt.legend(labels, loc='upper left')
plt.savefig('sufficiency.pdf')

for i, c, sc, sb in zip(range(1, n+1), costs, s_costs, s_bids):
  plt.figure()
  plt.plot(c, bids, 'b')
  plt.plot(sc, sb, 'r.')
  plt.grid()
  plt.xlim(support)
  plt.ylim([int(bids[0]), bids[-1]])
  plt.xlabel(r"Cost, $c_i$")
  plt.ylabel(r"Bid, $b_i$")
  plt.savefig('sufficiency_{}.pdf'.format(i))
