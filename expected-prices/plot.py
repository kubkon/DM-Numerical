#!/usr/bin/env python
# encoding: utf-8
"""
plot.py

Created by Jakub Konka on 2012-10-27.
Copyright (c) 2012 University of Strathclyde. All rights reserved.
"""
import argparse
from csv import DictReader
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.offsetbox as moffsetbox
from matplotlib import rc
import os

## for Palatino and other serif fonts use:
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 18})

### Parse command line arguments
parser = argparse.ArgumentParser(description="Expected prices in one-shot DM auction")
parser.add_argument('save_dir', help='output directory')
args = parser.parse_args()
save_dir = args.save_dir

### Load data from files
print("Loading data from files...")
extension = '.out'
file_names = [f[:f.find(extension)] for root, _, files in os.walk(save_dir) for f in files if f.endswith(extension)]
data_dct = {}
for fn in file_names:
  data_dct[fn] = {}
  with open(save_dir + '/' + fn + extension, mode='rt') as f:
    reader = DictReader(f)
    for row in reader:
      for key in row:
        data_dct[fn].setdefault(key,[]).append(float(row[key]))

### Plot
print("Plotting the results...")
# Plot the results
plt.figure()
styles = cycle(['.', 'x', 'o', 's'])
for key in sorted(data_dct.keys()):
  label = '(' + ', '.join(key.split('_')) + ')'
  plt.plot(data_dct[key]['w'], data_dct[key]['mean'], next(styles), label=label)
plt.annotate('', xy=(0.4, 2.0), xycoords='data',
             xytext=(0.18, 0.9), textcoords='data',
             arrowprops=dict(arrowstyle="->"), fontsize=14)
plt.annotate(r"as $(r_2-r_1)$ increases", xy=(0.4, 2.0),
             xycoords='data', fontsize=14)
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Average price")
plt.ylim([0.5, 5])
plt.grid()
l = plt.legend(prop={'size':14})
matplotlib.rcParams.update({'font.size': 14})
txt = moffsetbox.TextArea(r"Line\hspace{7mm} ($r_1$,\hspace{2.5mm} $r_2$)")
box = l._legend_box
box.get_children().insert(0, txt)
box.set_figure(box.figure)
plt.savefig(save_dir + "/expected_prices.pdf")
