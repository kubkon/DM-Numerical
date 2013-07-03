from cython_gsl cimport *
from libc.stdlib cimport calloc, free
cimport numpy as cnp
import numpy as np

cnp.import_array()

ctypedef struct Tode:
  int n
  double * params
  int f(int, double *, double, double *, double *)

cdef int f(int n, double * params, double t, double * y, double * f) with gil:
  cdef cnp.npy_intp shape[1]
  shape[0] = <cnp.npy_intp> n
  ys = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, y)
  uppers = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, params)
  rs = np.empty(n, dtype=np.float64)
  for i from 0 <= i < n:
    rs[i] = 1.0 / (t - ys[i])
  const = np.ones(n) * (np.sum(rs) / (n - 1))
  fs = (uppers - ys) * (const - rs)
  for i from 0 <= i < n:
    f[i] = fs[i]
  return GSL_SUCCESS

cdef int ode(double t, double y[], double f[], void *params) nogil:
  cdef Tode * P = <Tode *> params
  with gil:
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
  epsAbs = 1.49012e-8
  cdef gsl_odeiv2_driver * d
  d = gsl_odeiv2_driver_alloc_y_new(
      &sys, gsl_odeiv2_step_rkf45,
      hstart, epsAbs, 0.0)

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
