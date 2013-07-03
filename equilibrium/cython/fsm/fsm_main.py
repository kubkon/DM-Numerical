from fsm import solve
import numpy as np
from scipy.stats import uniform
from functools import reduce

def upper_bound_bids(lowers, uppers):
  """Returns an estimate on upper bound on bids.

  Arguments (all NumPy arrays):
  lowers -- array of lower extremities
  uppers -- array of upper extremities
  """
  # tabulate the range of permissible values
  vals = np.linspace(uppers[0], uppers[1], 10000)
  tabulated = []
  # solve the optimization problem in Eq. (1.8) in the thesis
  for v in vals:
    probs = [1-uniform(loc=l, scale=(u-l)).cdf(v) for l, u in zip(lowers[1:], uppers[1:])]
    tabulated += [(v - uppers[0]) * reduce(lambda p,r: p*r, probs, 1)]
  tabulated = np.array(tabulated)
  return vals[np.argmax(tabulated)]

# set the scenario
w = 0.85
reputations = [0.2, 0.4, 0.6, 0.8]
# compute an array of lower and upper extremities
lowers = np.array([(1-w)*r for r in reputations])
uppers = np.array([(1-w)*r + w for r in reputations])
# estimate the upper bound on bids
b_upper = upper_bound_bids(lowers, uppers)

# set initial conditions for the FSM algorithm
low = lowers[1]
high = b_upper
epsilon = 1e-6

# run the FSM algorithm until the estimate of the lower bound
# on bids is found
while high - low > epsilon:
  guess = 0.5 * (low + high)
  bids = np.linspace(guess, b_upper, num=10000, endpoint=False)
  try:
    costs = solve(lowers, uppers, bids).T
  except Exception:
    # if an error is raised, set low to guess and continue
    low = guess
    continue
  cond1, cond2 = [], []
  for l,c in zip(lowers, costs):
    for x,b in zip(c, bids):
        cond1 += [l <= x and x <= b_upper]
        cond2 += [b > x]
  cond3 = [b1 < b2 for b1, b2 in zip(bids, bids[1:])]
  if all(cond1 + cond2 + cond3):
    high = guess
  else:
    low = guess

print("Estimated lower bound on bids: %r" % guess)

# save the results in a file
with open('ode.out', 'wt') as f:
  labels = ['w', 'reps', 'bids'] + ['costs_{}'.format(i) for i in range(len(reputations))]
  labels = ' '.join(labels)
  values = [w, reputations, bids.tolist()] + [c.tolist() for c in costs]
  values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
  f.write(labels)
  f.write('\n')
  f.write(values)
