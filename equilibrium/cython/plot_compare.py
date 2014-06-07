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
parser.add_argument('directory', help='directory')
args = parser.parse_args()
directory = args.directory

# get all filenames in directory
fns = sorted([join(directory, f) for f in os.listdir(directory) if isfile(join(directory, f)) and 'csv' in f.lower()])

# parse filenames for reputation values
reps = sorted(list(set([tuple(map(lambda x: float(x) / 100, split(fn)[-1].split('_')[:2])) for fn in fns])))

# parse files
def parse_files(suffix):
    parsed = []
    for fn in filter(lambda x: suffix in x, fns):
        with open(fn, 'rt') as f:
            reader = csv.DictReader(f)
            dct = {}
            for row in reader:
                for key in row:
                    dct.setdefault(key, []).append(float(row[key]))
            parsed.append(dct)
    return parsed

parsed_dm = parse_files('dm')
parsed_cp = parse_files('cp')

# prepare for plotting
ws = parsed_dm[0]['w']
utils_errors = []
price_errors = []
for dm, cp in zip(parsed_dm, parsed_cp):
    keys         = [k for k in dm if k != 'w']
    utils  = {}
    prices = []
    for key in keys:
        for x_y in zip(dm[key], cp[key]):
            err = abs((x_y[0] - x_y[1]) / x_y[0]) * 100
            if 'price' == key:
                prices.append(err)
            else:
                utils.setdefault(key, []).append(err)
    utils_errors.append(utils)
    price_errors.append(prices)

# plot
xs = np.linspace(ws[0], ws[-1], 100)
keys = [k for k in utils_errors[0] if k != 'w' and k != 'price']
styles = ['bv', 'r^', 'og', 'sk']
labels = [r"$(%.2f, %.2f)$" % (r[0], r[1]) for r in reps]
n_reps = ["$r_%d$" % i for i in range(1, len(reps[0])+1)]
txt = "Marker\hspace{4mm} (" + ",\hspace{2.5mm}".join(n_reps) + ")"
txt = moffsetbox.TextArea(txt)
for key,i in zip(sorted(keys), range(1, len(keys)+1)):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    style = its.cycle(styles)
    for u in utils_errors:
        ax.plot(ws, u[key], next(style), label=key)
    plt.grid()
    plt.xlabel(r"Price weight, $w$")
    plt.ylabel(r"" + "Percentage relative error, $\eta_{\Pi_" + str(i) + "}\cdot 100\%$")
    handles, _ = ax.get_legend_handles_labels()
    loc = "lower right" if i % 3 != 0 else "upper right"
    l = ax.legend(handles, labels, loc=loc)
    box = l._legend_box
    box.get_children().insert(0, txt)
    box.set_figure(box.figure)
    plt.xlim([0.5, 1.0])
    plt.savefig('error_utilities_%s.pdf' % key)
    plt.close()

fig = plt.figure()
ax  = fig.add_subplot(111)
style = its.cycle(styles)
for u in price_errors:
    ax.plot(ws, u, next(style), label='label')
plt.grid()
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error, $\eta_p\cdot 100\%$")
handles, _ = ax.get_legend_handles_labels()
l = ax.legend(handles, labels, loc="lower right")
box = l._legend_box
box.get_children().insert(0, txt)
box.set_figure(box.figure)
plt.xlim([0.5, 1.0])
plt.savefig('error_prices.pdf')
plt.close()

