import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.offsetbox as moffsetbox
from matplotlib import rc
from scipy.optimize import curve_fit

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# empirical simulation time of run_compare.py script
xs = [2,3,4,5]
ys = [1.5577768472168818, 25.55814958737956, 76.21417879720529, 271.95091264373724]

# fit polynomial
cs = np.polyfit(xs, ys, 2)
f = np.poly1d(cs)

# fit exponential
def func(x, a, b, c):
    return a * np.exp(b*x) + c
popt, pcov = curve_fit(func, xs, ys)

print("Fitting errors")
for x,y in zip(xs, ys):
    f1 = abs(f(x) - y)
    f2 = abs(func(x, *popt) - y)
    print("At ({}, {}): poly={}, exp={}".format(x,y,f1,f2))

print("Exp prediction for x=10: {}".format(func(10, *popt)))

# plot
xss = np.linspace(2, 5, 1000)
plt.plot(xs, ys, 'ro')
# plt.plot(xss, f(xss), 'b-')
plt.plot(xss, func(xss, *popt), 'b-')
plt.grid()
plt.xlabel(r"Number of bidders, $n$")
plt.ylabel(r"Simulation time in hours")
plt.legend(['Empirical results', 'Exponential fit'], loc="upper left")
plt.savefig('simulation_time.pdf')

