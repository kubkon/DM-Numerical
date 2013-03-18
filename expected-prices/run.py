#!/usr/bin/env python
# encoding: utf-8
"""
run.py

Created by Jakub Konka on 2012-10-13.
Copyright (c) 2012 University of Strathclyde. All rights reserved.
"""
import argparse
import os
import os.path
import subprocess as sub


### Parse command line arguments
parser = argparse.ArgumentParser(description="Expected prices in one-shot DM auction")
parser.add_argument('N', metavar='N',
                    type=int, help='number of replications')
parser.add_argument('--batch_size', dest='batch_size', default=4,
                    type=int, help='batch size for multiprocessing')
parser.add_argument('--save_dir', dest='save_dir', default='out',
                    help='output directory')
args = parser.parse_args()
N = args.N
batch_size = args.batch_size
save_dir = args.save_dir

### Init
reps = [[.25,1.0], [.25,.75], [.25,.5], [.25,.25]]

### Run simulations
try:
  # One process at a time
  if batch_size == 1:
    for r in reps:
      sub.call("python expected_prices.py {} {} --save_dir={}".format(N, "{},{}".format(*r), save_dir), shell=True)
  # In batches
  else:
    # Split num of repetitions into batches
    repetitions = len(reps)
    quotient = repetitions // batch_size
    remainder = repetitions % batch_size
    # Run the simulations in parallel as subprocesses
    num_proc = batch_size if batch_size <= repetitions else remainder
    procs = [sub.Popen("python expected_prices.py {} {} --save_dir={}".format(N, "{},{}".format(*reps[n]), save_dir), shell=True) for n in range(num_proc)]
    while True:
      procs_poll = list(map(lambda x: x.poll() != None, procs))
      if not all(procs_poll):
        procs[procs_poll.index(False)].wait()
      elif num_proc < repetitions:
        temp_num = batch_size if num_proc + batch_size <= repetitions else remainder
        for n in range(num_proc, num_proc + temp_num):
          procs += [sub.Popen("python expected_prices.py {} {} --save_dir={}".format(N, "{},{}".format(*reps[n]), save_dir), shell=True)]
        num_proc += temp_num
      else:
        break
except OSError as e:
  print("Execution failed: ", e)
