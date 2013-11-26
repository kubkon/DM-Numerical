import argparse
import csv
from itertools import cycle, chain
from functools import partial

import numpy as np
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from dm.common import upper_bound_bids
from util import verify_sufficiency

csv.field_size_limit(1000000000)

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})


### Parse command line arguments
parser = argparse.ArgumentParser(description="Numerical approximation -- sufficiency analyzer")
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

# Parse data common to FSM and PPM methods
w = float(data_in['w'])
reps = [float(r.replace("[", "").replace("]", "")) for r in data_in['reps'].split(',')]
n = len(reps)

# Estimate cost support bounds
lower_extremities = np.array([(1-w)*r for r in reps])
upper_extremities = np.array([(1-w)*r + w for r in reps])

# Estimate upper bound on bids
b_upper = upper_bound_bids(lower_extremities, upper_extremities)

# Parse the rest of the data
try:
  bids = [float(b.replace("[","").replace("]","")) for b in data_in['bids'].split(',')]
  costs = []
  
  for i in range(n):
    label = 'costs_{}'.format(i)
    costs.append([float(c.replace("[","").replace("]","")) for c in data_in[label].split(',')])

  # Convert to numpy arrays
  costs = np.array(costs)
  bids = np.array(bids)

except KeyError:
  bs = [float(data_in['b_lower']), float(data_in['b_upper'])]
  css = []

  for i in range(n):
    label = 'cs_{}'.format(i)
    cs = [float(c.replace("[","").replace("]","")) for c in data_in[label].split(',')]
    css += [cs]

# Verify sufficiency
cdfs = []
for bounds in zip(lower_extremities, upper_extremities):
  cdfs.append(ss.uniform(loc=bounds[0], scale=bounds[1]-bounds[0]))

try:
  step = len(bids) // 35
  s_costs, s_bids = verify_sufficiency(costs, bids, b_upper, cdfs, step=step)

except NameError:
  # Define inverse bid function
  def cost_func(l, cs, x):
    return l + sum([c*(x-bs[0])**i for c,i in zip(cs, range(1,len(cs)+1))])
  
  # Compute bids and costs
  bids = np.linspace(bs[0], bs[1], 10000)
  cost_funcs = [partial(cost_func, l, cs) for l,cs in zip(lower_extremities, css)]
  costs = np.array([[f(b) for b in bids] for f in cost_funcs])

  step = len(bids) // 50
  s_costs, s_bids = verify_sufficiency(costs, bids, b_upper, cdfs, step=step)

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
