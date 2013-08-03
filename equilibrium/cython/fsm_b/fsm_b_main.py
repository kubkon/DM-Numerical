from fsm_b import solve
import numpy as np
import matplotlib.pyplot as plt

# set the scenario
support = [2.0, 8.0]
params = [{'mu': 4.0, 'sigma': 1.5},
          {'mu': 5.0, 'sigma': 1.5},
          {'mu': 6.0, 'sigma': 1.5}]

support = [0.0, 1.0]
params = [{'mu': 0.0, 'sigma': 1.0},
          {'mu': 0.0, 'sigma': 1.0}]

# set initial conditions for the FSM algorithm
low = support[0]
high = support[1]
epsilon = 1e-6

# run the FSM algorithm until the estimate of the lower bound
# on bids is found
costs = []
while high - low > epsilon:
  guess = 0.5 * (low + high)
  print(guess)
  bids = np.linspace(guess, support[1], num=10000, endpoint=False)
  try:
    costs = solve(params, support, bids).T
  except Exception:
    # if an error is raised, set low to guess and continue
    low = guess
    continue
  cond1, cond2 = [], []
  for c in costs:
    for x,b in zip(c, bids):
        cond1 += [support[0] <= x and x <= support[1]]
        cond2 += [b > x]
  # cond3 = [b1 < b2 for b1, b2 in zip(bids, bids[1:])]
  # if all(cond1 + cond2 + cond3):
  if all(cond1 + cond2):
    high = guess
  else:
    low = guess

print("Estimated lower bound on bids: %r" % guess)

plt.figure()
for c in costs:
  plt.plot(c, bids)
plt.grid()
plt.savefig("test.pdf")
