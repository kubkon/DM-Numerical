import argparse
import ast
import csv
from itertools import cycle, repeat
import numpy as np
from numpy.random import choice
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess as sub

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models -- helper script")
parser.add_argument('n_bidders', type=int, help='number of bidders')
parser.add_argument('n_ws', type=int, help='number of price weight values')
parser.add_argument('n_reps', type=int, help='number of reputation values to draw per price weight')
parser.add_argument('--batch_size', dest='batch_size', default=8,
                    type=int, help='batch size for multiprocessing')
args = parser.parse_args()
n_bidders = args.n_bidders
n_ws = args.n_ws
n_reps = args.n_reps
batch_size = args.batch_size

# prepare the scenario
ws = np.linspace(0.55, 0.99, n_ws)
rands = choice(np.linspace(0.01, 0.99, 10000), n_bidders*n_reps, replace=False)
reputations = [sorted(rands[i:i+n_bidders].tolist()) for i in range(0, len(rands), n_bidders)]

results = {}
for w in ws:
    # prepare the subprocess commands
    cmds = []

    for reps in reputations:
        cmd =  "python compare.py --w=%r " % w
        cmd += " ".join(["--reps=%r" % r for r in reps])
        cmds.append(cmd)

    # run comparisons
    try:
        # one process at a time
        if batch_size == 1:
            for cmd in cmds:
                output = ast.literal_eval(sub.check_output(cmd, shell=True)
                            .decode('utf-8').rstrip())
                results.setdefault(w, []).append(output)
        # in batches
        else:
            # split into batches
            repetitions = len(cmds)
            quotient = repetitions // batch_size
            remainder = repetitions % batch_size

            # run the simulations in parallel as subprocesses
            num_proc = batch_size if batch_size <= repetitions else remainder
            
            procs = []
            for i in range(num_proc):
                procs.append(sub.Popen(cmds[i], shell=True, stdout=sub.PIPE))

            while True:
                for p in procs:
                    output = ast.literal_eval(p.communicate()[0]
                                .decode('utf-8').rstrip())
                    results.setdefault(w, []).append(output)

                if len(results[w]) == repetitions:
                    break

                procs = []
                temp_num = batch_size if num_proc + batch_size <= repetitions else remainder

                for i in range(num_proc, num_proc + temp_num):
                    procs.append(sub.Popen(cmds[i], shell=True, stdout=sub.PIPE))
                num_proc += temp_num

    except OSError as e:
        print("Execution failed: ", e)

print(results)

# # save results to files
# prefix   = '_'.join(list(map(lambda x: str(int(x*100)), reps)))
# for t in ['dm', 'cp']:
#     with open(prefix + "_compare_%s.csv" % t, "wt") as f:
#         writer = csv.writer(f)
#         writer.writerow(["w", "price"] + ["bidder_{}".format(i+1) for i in range(n)])
    
#         for w,r in zip(ws, results):
#             values = [w, r['prices'][t]] + r['utilities'][t]
#             writer.writerow(list(map(lambda x: '%r' % x, values)))

