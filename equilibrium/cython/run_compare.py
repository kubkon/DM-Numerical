import argparse
import ast
import csv
from itertools import cycle, repeat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess as sub

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models -- helper script")
parser.add_argument('--batch_size', dest='batch_size', default=8,
                    type=int, help='batch size for multiprocessing')
args = parser.parse_args()
batch_size = args.batch_size

# prepare the scenario
ws = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.99]
reps = [0.25, 0.75]
n = len(reps)

# prepare the subprocess commands
cmds = []

for w in ws:
    cmd = "python compare.py --w=%r " % w
    cmd += " ".join(["--reps=%r" % r for r in reps])
    cmds.append(cmd)

# run comparisons
try:
    results = []

    # one process at a time
    if batch_size == 1:
        for cmd in cmds:
            output = ast.literal_eval(sub.check_output(cmd, shell=True)
                        .decode('utf-8').rstrip())
            results.append(output)
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
                results.append(output)

            if len(results) == repetitions:
                break

            procs = []
            temp_num = batch_size if num_proc + batch_size <= repetitions else remainder

            for i in range(num_proc, num_proc + temp_num):
                procs.append(sub.Popen(cmds[i], shell=True, stdout=sub.PIPE))
            num_proc += temp_num

except OSError as e:
    print("Execution failed: ", e)

# save results to files
with open("compare.csv", "wt") as f:
    writer = csv.writer(f)
    writer.writerow(["w", "price"] + ["bidder_{}".format(i) for i in range(n)])
    
    for w,r in zip(ws, results):
        writer.writerow([w, r[1]] + r[0])

# plot errors in utilities
plt.figure()
styles = ['+b', 'xr', 'og']
styles_cycle = cycle(styles)

for i in range(n):
    xs, ys = [], []

    for w,r in zip(ws, results):
        xs.append(w)
        ys.append(r[0][i])

    plt.plot(xs, ys, next(styles_cycle))

plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Difference in ex-ante expected utilities")
plt.grid()
plt.legend(["Bidder %d" % (i+1) for i in range(n)], loc='upper right')
plt.savefig("compare_utilities.pdf")

# plot difference in prices
plt.figure()

xs, ys = [], []
for w,r in zip(ws, results):
    xs.append(w)
    ys.append(r[1])

plt.plot(xs, ys, '+b')
plt.xlabel(r"Price weight, $w$")
plt.ylabel(r"Difference in expected prices")
plt.ylim([75,125])
plt.grid()
plt.savefig("compare_prices.pdf")
