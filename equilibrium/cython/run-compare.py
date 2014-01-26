import argparse
import ast
from itertools import cycle, repeat
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
# locs = [(1-w)*r + w/2 for r in reps]
# scales = np.linspace(w/5, w, num)
locs = [np.linspace((1-w)*r, (1-w)*r + w, num) for r in reps]
scales = [0.128, 0.128]
shapes = [0, 0]

n = len(reps)

# unpack
# locs = list(repeat(locs, num))
# scales = list(zip(scales, scales))
locs = list(zip(*locs))
scales = list(repeat(scales, num))
shapes = list(repeat(shapes, num))

# prepare the subprocess commands
cmds = []

for zipped in zip(locs, scales, shapes):
    cmd = "python compare.py --w=%f" % w
    cmd += " --reps=%f --reps=%f" % tuple(reps)
    cmd += " --locs=%f --locs=%f" % tuple(zipped[0])
    cmd += " --scales=%f --scales=%f" % tuple(zipped[1])
    cmd += " --shapes=%f --shapes=%f" % tuple(zipped[2])
    cmds.append(cmd)

# run comparisons
try:
    results = []

    # one process at a time
    if batch_size == 1:
        for cmd in cmds:
            output = ast.literal_eval(sub.check_output(cmd, shell=True).decode('utf-8').rstrip())
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
                output = ast.literal_eval(p.communicate()[0].decode('utf-8').rstrip())
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

# plot
plt.figure()
styles = ['+b', 'xr']
styles_cycle = cycle(styles)

for i in range(n):
    xs, ys = [], []

    for s,r in zip(locs, results):
        xs.append(s[i])
        ys.append(r[i])

    plt.plot(xs, ys, next(styles_cycle))

plt.xlabel("Mean")
plt.ylabel("Kolmogorov-Smirnov statistic")
plt.grid()
plt.legend(["Bidder %d" % i for i in range(n)], loc='upper left')
plt.savefig("compare.pdf")
