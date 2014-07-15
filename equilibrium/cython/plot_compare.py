import argparse
import csv
import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.offsetbox as moffsetbox
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os
from os.path import join, isfile, split


rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# parse command line arguments
parser = argparse.ArgumentParser(description="Plot")
parser.add_argument('filename', help='filename')
args = parser.parse_args()
filename = args.filename

# parse files
parsed = {}
with open(filename, 'rt') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key in row:
            parsed.setdefault(key, []).append(float(row[key]))

# plot
ws = parsed['w']
fig = plt.figure()
ax  = fig.add_subplot(111)
styles = ['o', 'x', '+', 'v', '^']
style = its.cycle(styles)
ax.errorbar(ws, parsed['price mean'], yerr=parsed['price ci'], fmt=next(style))
keys = sorted([key for key in parsed if 'bidder' in key])
for i in range(0,len(keys),2):
    ks = keys[i:i+2]
    ax.errorbar(ws, parsed[ks[1]], yerr=parsed[ks[0]], fmt=next(style))
plt.grid()
plt.xlim([0.5, 1.0])
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error")
plt.legend(['Price'] + ['Bidder %d' % i for i in range(1,len(keys)//2 + 1)], loc='upper right')
plt.savefig('.'.join([filename.split('.')[0], 'pdf']))

