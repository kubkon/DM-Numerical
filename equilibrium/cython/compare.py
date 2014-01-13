import itertools as its
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as ss

import bajari.fsm.main as bajari
import dm.fsm.main as dm
from bajari.dists.main import skewnormal
from util import compute_expected_utilities

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})



