import argparse
import csv
import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import linregress
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

# regress
def nonlinear_model(xs, *params):
    n = len(params)
    r_params = params[::-1]
    ys = np.ones(len(xs), np.float) * params[0]

    for i in np.arange(1, n):
        ys += np.power(xs, i) * params[i]

    return ys

util_errors_params = {}
for key in util_errors:
    init = np.ones(5, np.float) * 0.1
    params, _ = curve_fit(nonlinear_model, parsed_dm['w'], util_errors[key], init)
    util_errors_params[key] = params

def linear_model(xs, *params):
    return params[0] * xs + params[1]

price_params = linregress(parsed_dm['w'], price_errors)

# plot
fig = plt.figure()
ax  = fig.add_subplot(111)
styles = ['b+', '-b', 'rx', '-r', 'og', '-g']
style  = its.cycle(styles)
keys = sorted([k for k in util_errors])
xs = np.linspace(parsed_dm['w'][0], parsed_dm['w'][-1], 100)
for key in keys:
    ax.plot(parsed_dm['w'], util_errors[key], next(style), label=key)
    ax.plot(xs, nonlinear_model(xs, *util_errors_params[key]), next(style), label=key)
plt.grid()
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error, $\eta_{\Pi_i}\cdot 100\%$")
labels     = ["Bidder %d" % i for i in range(1, len(keys) + 1)]
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles[::2], labels, loc="lower right")
plt.savefig(folder + '/error_utilities.pdf')

plt.figure()
plt.plot(parsed_dm['w'], price_errors, '+b')
plt.plot(xs, linear_model(xs, *price_params), '-b')
plt.grid()
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error, $\eta_p\cdot 100\%$")
plt.savefig(folder + '/error_prices.pdf')

