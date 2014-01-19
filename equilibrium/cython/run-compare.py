import argparse
import ast
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub

# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models -- Helper script")
parser.add_argument('num', type=int, help='number of scale values to consider')
parser.add_argument('--batch_size', dest='batch_size', default=8,
                    type=int, help='batch size for multiprocessing')
args = parser.parse_args()
num = args.num
batch_size = args.batch_size

# prepare the scenario
w = 0.5
reps = [0.25, 0.75]
locs = [(1-w)*r + w/2 for r in reps]
shapes = [-1, 1]
scales = np.linspace(w/5, w, num)
n = len(reps)

# prepare the subprocess command
cmd = "python compare.py --w=%f" % w

for r,l,sh in zip(reps, locs, shapes):
    cmd += " --reps=%f --locs=%f --shapes=%f" % (r,l,sh)

# run comparisons
try:
    results = []

    # one process at a time
    if batch_size == 1:
        for i,s in zip(range(scales.size), scales):
            execmd = cmd + ''.join([" --scales=%f" % s for j in range(n)])
            output = ast.literal_eval(sub.check_output(execmd, shell=True).decode('utf-8').rstrip())
            results.append((i, output))
    # in batches
    else:
        # split into batches
        repetitions = scales.size
        quotient = repetitions // batch_size
        remainder = repetitions % batch_size

        # run the simulations in parallel as subprocesses
        num_proc = batch_size if batch_size <= repetitions else remainder
        
        procs = []
        for i in range(num_proc):
            execmd = cmd + ''.join([" --scales=%f" % scales[i] for j in range(n)])
            procs.append((i, sub.Popen(execmd, shell=True, stdout=sub.PIPE)))

        while True:
            for p in procs:
                output = ast.literal_eval(p[1].communicate()[0].decode('utf-8').rstrip())
                results.append((p[0], output))

            if len(results) == repetitions:
                break

            procs = []
            temp_num = batch_size if num_proc + batch_size <= repetitions else remainder

            for i in range(num_proc, num_proc + temp_num):
                execmd = cmd + ''.join([" --scales=%f" % scales[i] for j in range(n)])
                procs.append((i, sub.Popen(execmd, shell=True, stdout=sub.PIPE)))
            num_proc += temp_num

except OSError as e:
    print("Execution failed: ", e)

# plot
plt.figure()
styles = ['+b', 'xr']
styles_cycle = cycle(styles)

for i in range(n):
    xs, ys = [], []

    for r in results:
        xs.append(scales[r[0]])
        ys.append(r[1][i])

    plt.plot(xs, ys, next(styles_cycle))

plt.xlabel("Variance")
plt.ylabel("Kolmogorov-Smirnov statistic")
plt.grid()
plt.legend(["Bidder %d" % i for i in range(n)])
plt.savefig("compare.pdf")
