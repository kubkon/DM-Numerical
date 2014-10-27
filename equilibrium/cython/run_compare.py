import argparse
import ast
import csv
from itertools import cycle, repeat
import numpy as np
from numpy.random import choice
from scipy.stats import t
from multiprocessing import Process, Queue
from os.path import isfile
import sys

from compare import compare


# parse command line arguments
parser = argparse.ArgumentParser(description="Compare auction models -- helper script")
parser.add_argument('n_bidders', type=int, help='number of bidders')
parser.add_argument('n_reps', type=int, help='number of reputation values to draw per price weight')
parser.add_argument('--batch_size', dest='batch_size', default=8,
                    type=int, help='batch size for multiprocessing')
parser.add_argument('--n_ws', dest='n_ws', default=10,
                    type=int, help='number of price weight values')
parser.add_argument('--w', dest='w', default=None, type=float,
                    help='price weight value')
args = parser.parse_args()
n_bidders = args.n_bidders
n_reps = args.n_reps
batch_size = args.batch_size
n_ws = args.n_ws
w = args.w

# prepare the scenario
if w:
    ws = [w]
else:
    ws = np.linspace(0.55, 0.99, n_ws)

rs = np.linspace(0.01, 0.99, 10000)

reps = []
for i in range(n_reps):
    while True:
        rand = sorted(choice(rs, n_bidders, replace=False).tolist())
        if rand not in reps:
            break
    reps.append(rand)

reputations = {w: reps for w in ws}

# prepare function params
def worker(input, output):
    for args in iter(input.get, 'STOP'):
        output.put(compare(*args))

task_queue = Queue()
counter = 0
for w in reputations:
    for r in reputations[w]:
        task_queue.put((w, r))
        counter += 1

result_queue = Queue()
for i in range(batch_size):
    Process(target=worker, args=(task_queue, result_queue)).start()

sys.stdout.write("Completed  0%")
sys.stdout.flush()

results = {}
i = 0
while i < counter:
    dct = result_queue.get()
    w = list(dct.keys())[0]
    if dct[w]['utilities'] is None or dct[w]['prices'] is None:
        while True:
            reps = sorted(choice(rs, n_bidders, replace=False).tolist())
            if reps not in reputations[w]:
                break
        reputations[w].append(reps)
        task_queue.put((w, reps))
        continue
    else:
        results.setdefault(w, []).append({
            'utilities': dct[w]['utilities'],
            'prices': dct[w]['prices']
        })
        i += 1

    percent = int(i / counter * 100)
    if percent / 10 > 1.0:
        sys.stdout.write("\b\b\b%d%%" % percent)
    else:
        sys.stdout.write("\b\b%d%%" % percent)
    sys.stdout.flush()

for i in range(batch_size):
    task_queue.put('STOP')

# calculate averages and confidence intervals
filename = str(n_bidders) + '_compare.csv'
has_header = False
if isfile(filename):
    has_header = True

headers = ['w', 'price mean', 'price ci']
for b in ['bidder_{}'.format(i+1) for i in range(n_bidders)]:
    headers.extend([b + ' mean', b + ' ci'])

with open(filename, 'wt') as f:
    writer = csv.DictWriter(f, headers)
    # write header if does not exist
    if not has_header:
        writer.writeheader()

    for w in sorted(results.keys()):
        prices_errors = []
        utilities_errors = []
        for r in results[w]:
            prices = r['prices']
            prices_errors.append(abs((prices['dm'] - prices['cp']) / prices['dm']) * 100)

            utilities = r['utilities']
            error = list(map(lambda x,y: abs((x - y) / x) * 100, utilities['dm'], utilities['cp']))
            utilities_errors.append(error)

        prices_mean = np.mean(prices_errors)
        prices_std  = np.sqrt(np.var(prices_errors))
        alpha       = 0.05
        n           = len(prices_errors)
        prices_ci   = t.ppf(1 - alpha/2, n-1) * prices_std / np.sqrt(n)

        utilities = []
        for x_y in zip(*utilities_errors):
            mean = np.mean(x_y)
            std  = np.sqrt(np.var(x_y))
            ci   = t.ppf(1 - alpha/2, n-1) * std / np.sqrt(n)
            utilities.extend([mean, ci])

        # save results to files
        values = [w, prices_mean, prices_ci] + utilities
        values = list(map(lambda x: '%r' % x, values))
        writer.writerow(dict(zip(headers, values)))

