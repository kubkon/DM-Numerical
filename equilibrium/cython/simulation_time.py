import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.offsetbox as moffsetbox
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# empirical simulation time of run_compare.py script
xs = [2,3,4]
ys = [1.5577768472168818, 25.55814958737956, 76.21417879720529]

# fit polynomial
cs = np.polyfit(xs, ys, 2)
f = np.poly1d(cs)

# plot
xss = np.linspace(2, 10, 1000)
plt.plot(xs, ys, 'ro')
plt.plot(xss, f(xss), 'b-')
plt.grid()
plt.xlabel(r"Number of bidders, $n$")
plt.ylabel(r"Simulation time in hours")
plt.legend(['Empirical results', 'Polynomial fit'], loc="upper left")
plt.savefig('simulation_time.pdf')

