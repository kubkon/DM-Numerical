from cython_gsl cimport *
import numpy as np
from scipy.stats import uniform
import functools as fts

def upper_bound_bids(lowers, uppers):
  vals = np.linspace(uppers[0], uppers[1], 10000)
  tabulated = []
  for v in vals:
    probs = [1-uniform(loc=l, scale=(u-l)).cdf(v) for l, u in zip(lowers[1:], uppers[1:])]
    tabulated += [(v - uppers[0]) * fts.reduce(lambda p,r: p*r, probs, 1)]
  tabulated = np.array(tabulated)
  return vals[np.argmax(tabulated)]

cdef int ode(double t, double y[], double f[], void *params) nogil:

  return GSL_SUCCESS

def solve(w, reputations):
  lowers = [(1-w)*r for r in reputations]
  uppers = list(map(lambda x: x + w, lowers))
  b_upper = upper_bound_bids(lowers, uppers)

  cdef gsl_odeiv2_system sys
  sys.function = ode
  sys.jacobian = NULL
  sys.dimension = len(reputations)
  sys.params = NULL

  cdef gsl_odeiv2_driver * d
  d = gsl_odeiv2_driver_alloc_y_new(
      &sys, gsl_odeiv2_step_rkf45,
      1e-6, 1e-6, 0.0)

  cdef int i
  cdef double t, t1, y[1]
  t = 0.0
  t1 = 1.0
  y[0] = 1.0

  cdef int status
  cdef double ti
  sol = []
  for i from 1 <= i <= 100:
    ti = i * ti / 100.0
    status = gsl_odeiv2_driver_apply(d, &t, ti, y)

    if status != GSL_SUCCESS:
      print("Error, return value=%d\n" % status)
      break

    sol += [(t, y[0])]

  gsl_odeiv2_driver_free(d)
  return sol
