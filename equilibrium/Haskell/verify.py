import argparse
import csv
import functools as fts
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import sys

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']))
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

### Parse command line arguments
parser = argparse.ArgumentParser(description="Numerical approximation -- verifier for 2 bidders")
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

# Parse data common to ODE and polynomial methods
w = float(data_in['w'])
reps = [float(r.replace("[", "").replace("]", "")) for r in data_in['reps'].split(',')]
n = len(reps)
# Estimated cost support bounds
lower_extremities = [(1-w)*r for r in reps]
upper_extremities = [(1-w)*r + w for r in reps]

# Parse the rest of the data
try:
  bids = [float(b.replace("[","").replace("]","")) for b in data_in['bids'].split(',')]
  costs = []
  for i in range(n):
    label = 'costs_{}'.format(i)
    costs += [[float(c.replace("[","").replace("]","")) for c in data_in[label].split(',')]]
except KeyError:
  bs = [float(data_in['b_lower']), float(data_in['b_upper'])]
  css = []
  for i in range(n):
    label = 'cs_{}'.format(i)
    cs = [float(c.replace("[","").replace("]","")) for c in data_in[label].split(',')]
    css += [cs]

# Quantize costs and bids
try:
  bids = bids[::20]
  costs = [c[::20] for c in costs]
except NameError:
  pass

# Compute theoretical results
c1 = [lower_extremities[0], upper_extremities[0]]
c2 = [lower_extremities[1], upper_extremities[1]]
b = [(c1[0]*c2[0] - ((c1[1] + c2[1]) / 2)**2) / (c1[0] - c1[1] + c2[0] - c2[1]),
     (c1[1] + c2[1]) / 2]
# Constants of integration
d1 = ((c2[1]-c1[1])**2 + 4*(b[0]-c2[1])*(c1[0]-c1[1])) / (-2*(b[0]-b[1])*(c1[0]-c1[1])) * np.exp((c2[1]-c1[1]) / (2*(b[0]-b[1])))
d2 = ((c1[1]-c2[1])**2 + 4*(b[0]-c1[1])*(c2[0]-c2[1])) / (-2*(b[0]-b[1])*(c2[0]-c2[1])) * np.exp((c1[1]-c2[1]) / (2*(b[0]-b[1])))
# Inverse bid functions
inv1 = lambda x: c1[1] + (c2[1]-c1[1])**2 / (d1*(c2[1]+c1[1]-2*x)*np.exp((c2[1]-c1[1])/(c2[1]+c1[1]-2*x)) + 4*(c2[1]-x))
inv2 = lambda x: c2[1] + (c1[1]-c2[1])**2 / (d2*(c1[1]+c2[1]-2*x)*np.exp((c1[1]-c2[1])/(c1[1]+c2[1]-2*x)) + 4*(c1[1]-x))
t_bids = np.linspace(b[0], b[1], 10000)

# Plot
plt.figure()
plt.plot([inv1(b) for b in t_bids], t_bids, 'b')
plt.plot(costs[0], bids, 'b.')
plt.plot([inv2(b) for b in t_bids], t_bids, 'r--')
plt.plot(costs[1], bids, 'rx')
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}$")
plt.grid()
labels = ['NO 1: Theory', 'NO 1: Numerical', 'NO 2: Theory', 'NO 2: Numerical']
plt.legend(labels, loc='upper left')
plt.savefig('verify.pdf')
