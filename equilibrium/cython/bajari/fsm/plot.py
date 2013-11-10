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
    f_reader = csv.DictReader(f, delimiter=',')
    for row in f_reader:
        for key in row:
            data_in.setdefault(key, []).append(float(row[key]))

# Parse
n = len(data_in) - 1

# Plot
styles = ['b', 'r--', 'g:', 'm-.']

plt.figure()
sts = its.cycle(styles)
for key in data_in:
    if 'costs' in key:
      plt.plot(data_in[key], data_in['bids'], next(sts))
plt.grid()
plt.xlabel(r"Cost, $c_i$")
plt.ylabel(r"Bid, $b_i$")
labels = ['Bidder {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')
