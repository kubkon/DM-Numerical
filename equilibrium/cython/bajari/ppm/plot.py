import argparse
import csv
import itertools as its
import functools as fts
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

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

# Parse
n = int(data_in['n'])
b_lower = float(data_in['b_lower'])
b_upper = float(data_in['b_upper'])
css = []
for i in range(n):
  label = 'cs_{}'.format(i)
  cs = [float(c.replace("[","").replace("]","")) for c in data_in[label].split(',')]
  css += [cs]

# Define inverse bid function
def cost_func(b_lower, cs, b):
  return b_lower - sum([c*(b-b_lower)**i for c,i in zip(cs, range(len(cs)))])

# Compute bids and costs
bids = np.linspace(b_lower, b_upper, 10000)
cost_funcs = [fts.partial(cost_func, b_lower, cs) for cs in css]
costs = [[f(b) for b in bids] for f in cost_funcs]

# Plot
styles = ['b', 'r--', 'g:', 'm-.']

plt.figure()
sts = its.cycle(styles)
for c in costs:
  plt.plot(c, bids, next(sts))
plt.grid()
plt.xlabel(r"Cost, $c_i$")
plt.ylabel(r"Bid, $b_i$")
labels = ['Bidder {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')
