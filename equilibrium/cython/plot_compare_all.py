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

filenames = ['results/averages3/%d_compare.csv' % i for i in range(2,6)]

# parse files
data = []
for filename in filenames:
    parsed = {}
    with open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                parsed.setdefault(key, []).append(float(row[key]))
    data.append(parsed)

# plot
ws = data[0]['w']
legend = ['%d bidders' % i for i in range(2,6)]
styles = ['o', 'x', '+', 'v', '^']

fig = plt.figure()
ax  = fig.add_subplot(111)
style = its.cycle(styles)
for d in data:
    ax.errorbar(ws, d['price mean'], yerr=d['price ci'], fmt=next(style))
plt.grid()
plt.xlim([0.5, 1.0])
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error")
plt.legend(legend, loc='upper left')
plt.savefig('price.pdf')

fig = plt.figure()
ax  = fig.add_subplot(111)
style = its.cycle(styles)
for d in data:
    ax.errorbar(ws, d['bidder_1 mean'], yerr=d['bidder_1 ci'], fmt=next(style))
plt.grid()
plt.xlim([0.5, 1.0])
plt.ylim([0, 21])
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error")
plt.legend(legend, loc='upper left')
plt.savefig('bidder_1.pdf')

fig = plt.figure()
ax  = fig.add_subplot(111)
style = its.cycle(styles)
for d in data:
    ax.errorbar(ws, d['bidder_2 mean'], yerr=d['bidder_2 ci'], fmt=next(style))
plt.grid()
plt.xlim([0.5, 1.0])
plt.ylim([0, 21])
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error")
plt.legend(legend, loc='upper left')
plt.savefig('bidder_2.pdf')

fig = plt.figure()
legend = ['%d bidders' % i for i in range(3,6)]
ax  = fig.add_subplot(111)
style = its.cycle(styles)
for d in data:
    try:
        ax.errorbar(ws, d['bidder_3 mean'], yerr=d['bidder_3 ci'], fmt=next(style))
    except KeyError:
        continue
plt.grid()
plt.xlim([0.5, 1.0])
plt.ylim([0, 21])
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error")
plt.legend(legend, loc='upper left')
plt.savefig('bidder_3.pdf')

fig = plt.figure()
legend = ['%d bidders' % i for i in range(4,6)]
ax  = fig.add_subplot(111)
style = its.cycle(styles)
for d in data:
    try:
        ax.errorbar(ws, d['bidder_4 mean'], yerr=d['bidder_4 ci'], fmt=next(style))
    except KeyError:
        continue
plt.grid()
plt.xlim([0.5, 1.0])
plt.ylim([0, 21])
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Percentage relative error")
plt.legend(legend, loc='upper right')
plt.savefig('bidder_4.pdf')

