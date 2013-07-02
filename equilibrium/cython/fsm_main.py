from fsm import solve
import numpy as np
from scipy.stats import uniform
from functools import reduce
import matplotlib.pyplot as plt

def upper_bound_bids(lowers, uppers):
  vals = np.linspace(uppers[0], uppers[1], 10000)
  tabulated = []
  for v in vals:
    probs = [1-uniform(loc=l, scale=(u-l)).cdf(v) for l, u in zip(lowers[1:], uppers[1:])]
    tabulated += [(v - uppers[0]) * reduce(lambda p,r: p*r, probs, 1)]
  tabulated = np.array(tabulated)
  return vals[np.argmax(tabulated)]

w = 0.85
reputations = [0.2, 0.4, 0.6, 0.8]
lowers = np.array([(1-w)*r for r in reputations])
uppers = np.array([(1-w)*r + w for r in reputations])
b_upper = upper_bound_bids(lowers, uppers)

low = lowers[1]
high = b_upper
epsilon = 1e-6

while high - low > epsilon:
  guess = 0.5 * (low + high)
  print(guess)
  bids = np.linspace(guess, b_upper, num=1000, endpoint=False)
  try:
    costs = solve(lowers, uppers, bids).T
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
  except Exception:
    low = guess

plt.figure()
for c in costs:
  plt.plot(c, bids)
plt.savefig('test.pdf')
