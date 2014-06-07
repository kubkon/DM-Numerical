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
parser.add_argument('reps', nargs='+', type=float, help='reputation array')
parser.add_argument('--batch_size', dest='batch_size', default=8,
                    type=int, help='batch size for multiprocessing')
args = parser.parse_args()
reps = args.reps
batch_size = args.batch_size

# prepare the scenario
ws = np.linspace(0.55, 0.99, 40)
n = len(reps)

# prepare the subprocess commands
cmds = []

for w in ws:
    cmd =  "python compare.py --w=%.12f " % w
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
prefix   = '_'.join(list(map(lambda x: str(int(x*100)), reps)))
for t in ['dm', 'cp']:
    with open(prefix + "_compare_%s.csv" % t, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["w", "price"] + ["bidder_{}".format(i+1) for i in range(n)])
    
        for w,r in zip(ws, results):
            writer.writerow([w, r['prices'][t]] + r['utilities'][t])

