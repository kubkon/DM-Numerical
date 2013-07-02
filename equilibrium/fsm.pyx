from cython_gsl cimport *
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np
from scipy.stats import uniform
from functools import reduce

cnp.import_array()

def upper_bound_bids(lowers, uppers):
  vals = np.linspace(uppers[0], uppers[1], 10000)
  tabulated = []
  for v in vals:
    probs = [1-uniform(loc=l, scale=(u-l)).cdf(v) for l, u in zip(lowers[1:], uppers[1:])]
    tabulated += [(v - uppers[0]) * reduce(lambda p,r: p*r, probs, 1)]
  tabulated = np.array(tabulated)
  return vals[np.argmax(tabulated)]

cdef int ode(double t, double y[], double f[], void *params) nogil:
  cdef int n = <int>(<double *>params)[0]
  cdef cnp.npy_intp shape[1]
  
  cdef double * params_ = <double *> malloc(n*sizeof(double))

  cdef int i
  for i from 0 <= i < n:
    params_[i] = (<double *>params)[i+1]

  with gil:
    shape[0] = <cnp.npy_intp> n
    ys = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, y)
    uppers = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, params_)
    rs = np.empty(n, dtype=np.float64)
    for i in np.arange(n):
      rs[i] = 1.0 / (t - ys[i])
    const = np.ones(n) * (np.sum(rs) / (n - 1))
    fs = (uppers - ys) * (const - rs)
    for i from 0 <= i < n:
      f[i] = fs[i]

  free(params_)

  return GSL_SUCCESS

def solve(initial, uppers, bids):
  cdef int n = len(initial)

  cdef gsl_odeiv2_system sys
  sys.function = ode
  sys.jacobian = NULL
  sys.dimension = n
  
  cdef double* params = <double *> malloc((n+1)*sizeof(double))
  if params is NULL:
    raise MemoryError()

  cdef int i
  params[0] = <double> n
  for i from 0 <= i < n:
    params[i+1] = uppers[i]
  sys.params = params

  cdef gsl_odeiv2_driver * d
  d = gsl_odeiv2_driver_alloc_y_new(
      &sys, gsl_odeiv2_step_rkf45,
      1e-6, 1e-6, 0.0)

  cdef double *y = <double *> malloc(n*sizeof(double))
  if y is NULL:
    raise MemoryError()
  
  for i from 0 <= i < n:
    y[i] = initial[i]

  cdef int status
  cdef double b, t
  t = bids[0]
  sol = [initial]
  for b in bids[1:]:
    status = gsl_odeiv2_driver_apply(d, &t, b, y)

    if status != GSL_SUCCESS:
      print("Error, return value=%d\n" % status)
      break

    costs = np.empty(n, dtype=np.float64)
    for i from 0 <= i < n:
      costs[i] = y[i]
    sol += [costs]

  free(params)
  free(y)
  gsl_odeiv2_driver_free(d)

  return np.array(sol)

def fsm(w, reputations):
  lowers = [(1-w)*r for r in reputations]
  uppers = list(map(lambda x: x + w, lowers))
  b_upper = upper_bound_bids(lowers, uppers)

  low = lowers[1]
  high = b_upper
  epsilon = 1e-6

  # while high - low > epsilon:
  # guess = 0.5 * (low + high)
  guess = 0.52
  bids = np.linspace(guess, b_upper - 1e-3, 10000)
  costs = solve(lowers, uppers, bids).T
  
  return (bids, costs)
