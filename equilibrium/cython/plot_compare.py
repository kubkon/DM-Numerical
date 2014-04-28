import argparse
import csv
import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import os

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# parse command line arguments
parser = argparse.ArgumentParser(description="Plot comparison results")
parser.add_argument('folder', help='folder with comparison results')
args   = parser.parse_args()
folder = args.folder

# parse files
filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]
parsed_dm = {}
parsed_cp = {}
for fn in filenames:
    with open(folder + '/' + fn, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if 'dm' in fn:
                    parsed_dm.setdefault(key, []).append(float(row[key]))
                else:
                    parsed_cp.setdefault(key, []).append(float(row[key]))

# prepare for plotting
keys         = [k for k in parsed_dm if k != 'w']
util_errors  = {}
price_errors = []
for key in keys:
    for x_y in zip(parsed_dm[key], parsed_cp[key]):
        err = (x_y[0] - x_y[1]) / x_y[0] * 100
        if 'price' == key:
            price_errors.append(err)
        else:
            util_errors.setdefault(key, []).append(err)

plt.figure()
styles = ['b+', 'rx', 'og']
style  = its.cycle(styles)
keys = sorted([k for k in util_errors])
for key in keys:
    plt.plot(parsed_dm['w'], util_errors[key], next(style))
plt.grid()
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error, $\eta_{\Pi_i}\cdot 100\%$")
legend = ["Bidder %d" % i for i in range(1, len(keys) + 1)]
plt.legend(legend, loc="lower right")
plt.savefig(folder + '/error_utilities.pdf')

plt.figure()
plt.plot(parsed_dm['w'], price_errors, '+b')
plt.grid()
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error, $\eta_p\cdot 100\%$")
plt.savefig(folder + '/error_prices.pdf')

