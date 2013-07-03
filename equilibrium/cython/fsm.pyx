from cython_gsl cimport *
from libc.stdlib cimport calloc, free
import numpy as np

ctypedef struct Tode:
  int n
  double * params
  int f(int, double *, double, double *, double *) nogil

cdef int f(int n, double * params, double t, double * y, double * f) nogil:
  cdef double * rs = <double *> calloc(n, sizeof(double))
  cdef int i
  cdef double r, rs_sum
  rs_sum = 0

  for i from 0 <= i < n:
    r = 1 / (t - y[i])
    rs[i] = r
    rs_sum += r

  for i from 0 <= i < n:
    f[i] = (params[i] - y[i]) * (rs_sum / (n-1) - rs[i])

  free(rs)
  return GSL_SUCCESS

cdef int ode(double t, double y[], double f[], void *params) nogil:
  cdef Tode * P = <Tode *> params
  P.f(P.n, P.params, t, y, f)
  return GSL_SUCCESS

def solve(initial, uppers, bids):
  cdef int n = initial.size
  cdef Tode P
  P.n = n
  cdef double * params = <double *> calloc(n, sizeof(double))
  cdef int i
  for i from 0 <= i < n:
    params[i] = uppers[i]
  P.params = params
  P.f = f

  cdef gsl_odeiv2_system sys
  sys.function = ode
  sys.jacobian = NULL
  sys.dimension = n
  sys.params = &P

  cdef double hstart, epsAbs
  hstart = (bids[1] - bids[0]) / 100.0
  epsAbs = epsRel = 1.49012e-8
  cdef gsl_odeiv2_driver * d
  d = gsl_odeiv2_driver_alloc_y_new(
      &sys, gsl_odeiv2_step_rkf45,
      hstart, epsAbs, epsRel)

  cdef double *y = <double *> calloc(n, sizeof(double))
  if y is NULL:
    raise MemoryError()
  
  for i from 0 <= i < n:
    y[i] = initial[i]

  cdef int status, j
  cdef double t, ti
  t = bids[0]
  sol = [initial]

  try:
    for j from 1 <= j < bids.size:
      ti = bids[j]
      status = gsl_odeiv2_driver_apply(d, &t, ti, y)

      if status != GSL_SUCCESS:
        print("Error, return value=%d\n" % status)
        raise Exception(t, ti)

      costs = np.empty(n, dtype=np.float64)
      for i from 0 <= i < n:
        costs[i] = y[i]
      sol += [costs]
  finally:
    free(params)
    free(y)
    gsl_odeiv2_driver_free(d)

  return np.array(sol)
