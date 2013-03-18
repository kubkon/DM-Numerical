#!/usr/bin/env python
# encoding: utf-8
"""
error.py

Created by Jakub Konka on 2012-10-30.
Copyright (c) 2012 University of Strathclyde. All rights reserved.
"""
import argparse
from csv import DictReader
from itertools import combinations
import os

### Parse command line arguments
parser = argparse.ArgumentParser(description="Expected prices in one-shot DM auction")
parser.add_argument('save_dir', help='output directory')
args = parser.parse_args()
save_dir = args.save_dir

### Load data from files
print("Loading data from files...")
extension = '.out'
file_names = [f[:f.find(extension)] for root, _, files in os.walk(save_dir) for f in files if f.endswith(extension)]
fn_split_set = set([item for sublist in [fn.split('_') for fn in file_names] for item in sublist])
fn_comb = list(combinations(fn_split_set, 2))
for comb in fn_comb:
  try:
    data = []
    for i in range(2):
      path = save_dir + '/' + '_'.join([comb[i%2], comb[(i+1)%2]]) + extension
      with open(path, 'rt') as f:
        reader = DictReader(f)
        dct = {}
        for row in reader:
          for key in row:
            dct.setdefault(key,[]).append(float(row[key]))
        data.append(dct)

    ### Calculate absolute errors
    error = [abs(x-y) for (x,y) in zip(data[0]['mean'], data[1]['mean'])]
    rel_error = map(lambda x: x / data[0]['mean'][error.index(max(error))], error)
    print("Maximum relative error for reps {}: {}".format(comb, max(rel_error)))
  except IOError:
    continue
