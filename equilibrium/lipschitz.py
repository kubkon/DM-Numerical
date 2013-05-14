from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc('text', usetex=True)

def backward_runge_kutta(ode, terminal, step, begin):
  table = [terminal]
  t = terminal[0]
  y1 = terminal[1]
  y2 = terminal[2]
  while t > begin:
    o_1 = ode(t, y1, y2)
    v_1, w_1 = step * o_1[0], step * o_1[1]
    o_2 = ode(t - 0.5*step, y1 - 0.5*v_1, y2 - 0.5*w_1)
    v_2, w_2 = step * o_2[0], step * o_2[1]
    o_3 = ode(t - 0.5*step, y1 - 0.5*v_2, y2 - 0.5*w_2)
    v_3, w_3 = step * o_3[0], step * o_3[1]
    o_4 = ode(t - step, y1 - v_3, y2 - w_3)
    v_4, w_4 = step * o_4[0], step * o_4[1]
    y1 -= 1/6 * (v_1 + 2*v_2 + 2*v_3 + v_4)
    y2 -= 1/6 * (w_1 + 2*w_2 + 2*w_3 + w_4)
    t -= step
    table += [(t, y1, y2)]
  return table

# Scenario
w = 0.5
reps = [0.25, 0.75]
lowers = list(map(lambda r: (1-w)*r, reps))
uppers = list(map(lambda l: w + l, lowers))
b_low = (lowers[0]*lowers[1] - sum(uppers)**2 / 4) / (sum(lowers) - sum(uppers))
b_upper = sum(uppers) / 2
# System of ODEs for 2 nos
def ode(t,y1,y2):
  dy1 = (uppers[0] - y1) / (t - y2)
  dy2 = (uppers[1] - y2) / (t - y1)
  return (dy1, dy2)
sol = backward_runge_kutta(ode, (b_upper, uppers[0], b_upper-0.005), 0.01, b_low + 0.15)

# Compute theoretical results
c1 = [lowers[0], uppers[0]]
c2 = [lowers[1], uppers[1]]
b = [b_low, b_upper]
# Constants of integration
d1 = ((c2[1]-c1[1])**2 + 4*(b[0]-c2[1])*(c1[0]-c1[1])) / (-2*(b[0]-b[1])*(c1[0]-c1[1])) * np.exp((c2[1]-c1[1]) / (2*(b[0]-b[1])))
d2 = ((c1[1]-c2[1])**2 + 4*(b[0]-c1[1])*(c2[0]-c2[1])) / (-2*(b[0]-b[1])*(c2[0]-c2[1])) * np.exp((c1[1]-c2[1]) / (2*(b[0]-b[1])))
# Inverse bid functions
inv1 = lambda x: c1[1] + (c2[1]-c1[1])**2 / (d1*(c2[1]+c1[1]-2*x)*np.exp((c2[1]-c1[1])/(c2[1]+c1[1]-2*x)) + 4*(c2[1]-x))
inv2 = lambda x: c2[1] + (c1[1]-c2[1])**2 / (d2*(c1[1]+c2[1]-2*x)*np.exp((c1[1]-c2[1])/(c1[1]+c2[1]-2*x)) + 4*(c1[1]-x))
t_bids = np.linspace(b[0], b[1], 10000)

# Plot
plt.figure()
ts = list(map(lambda x: x[0], sol))
plt.plot(list(map(lambda x: x[1], sol)), ts, 'p')
plt.plot(list(map(lambda x: x[2], sol)), ts, 'p')
plt.plot([inv1(b) for b in t_bids], t_bids)
plt.plot([inv2(b) for b in t_bids], t_bids)
plt.grid()
plt.xlabel('Cost-hat, $\hat{c}_i$')
plt.ylabel('Bid-hat, $\hat{b}$')
plt.savefig('lipschitz.pdf')
