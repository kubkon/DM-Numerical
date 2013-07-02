from cython_gsl cimport *
from libc.stdlib cimport calloc, free
cimport numpy as cnp
import numpy as np

cnp.import_array()

cdef int ode(double t, double y[], double f[], void *params) nogil:
  cdef int n = <int>(<double *>params)[0]
  cdef cnp.npy_intp shape[1]
  
  cdef double * params_ = <double *> calloc(n, sizeof(double))

  cdef int i
  for i from 0 <= i < n:
    params_[i] = (<double *>params)[i+1]

  with gil:
    shape[0] = <cnp.npy_intp> n
    ys = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, y)
    uppers = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, params_)
    rs = np.empty(n, dtype=np.float64)
    for i from 0 <= i < n:
      rs[i] = 1.0 / (t - ys[i])
    const = np.ones(n) * (np.sum(rs) / (n - 1))
    fs = (uppers - ys) * (const - rs)
    for i from 0 <= i < n:
      f[i] = fs[i]

  free(params_)

  return GSL_SUCCESS

def solve(initial, uppers, bids):
  cdef int n = initial.size

  cdef gsl_odeiv2_system sys
  sys.function = ode
  sys.jacobian = NULL
  sys.dimension = n
  
  cdef double* params = <double *> calloc((n+1), sizeof(double))
  if params is NULL:
    raise MemoryError()

  cdef int i
  params[0] = <double> n
  for i from 0 <= i < n:
    params[i+1] = uppers[i]
  sys.params = params

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

  cdef int status
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
