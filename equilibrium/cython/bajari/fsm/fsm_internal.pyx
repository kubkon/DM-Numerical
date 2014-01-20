from cython_gsl cimport *

from libc.stdlib cimport calloc, free
from libc.math cimport exp, sqrt, pow, erf

from bajari.dists.dists cimport c_skew_normal_pdf, c_skew_normal_cdf

import numpy as np


# struct describing distribution params
ctypedef struct DistParams:
    double location
    double scale
    double shape

# struct specifying the system of ODE
ctypedef struct Tode:
  int n # number of bidders
  DistParams * params # array of params describing bidders
  # pointer to function describing system of ODEs
  int f(int, DistParams *, double, double *, double *) nogil

cdef int f(int n, DistParams * params, double t, double * y, double * f) nogil:
  """Evolves system of ODEs at a particular independent variable t,
  and for a vector of particular dependent variables y_i(t). Mathematically,
  dy_i(t)/dt = f_i(t, y_1(t), ..., y_n(t)).

  Arguments:
  n -- number of bidders
  params -- array of params describing bidders
  t -- independent variable
  y -- array of dependent variables
  f -- array of solved vector elements f_i(..)
  """
  cdef double * rs = <double *> calloc(n, sizeof(double))
  cdef int i
  cdef double r, rs_sum, num, den
  cdef DistParams param
  rs_sum = 0

  for i from 0 <= i < n:
    r = t - y[i]

    if r == 0:
      return GSL_EZERODIV

    rs[i] = 1 / r
    rs_sum += 1 / r

  # this loop corresponds to the system of equations (1.26) in the thesis
  for i from 0 <= i < n:
    param = params[i]
    num = 1 - c_skew_normal_cdf(y[i], param.location, param.scale, param.shape)
    den = c_skew_normal_pdf(y[i], param.location, param.scale, param.shape)

    if den == 0:
      return GSL_EZERODIV

    f[i] = num / den * (rs_sum / (n-1) - rs[i])

  free(rs)
  return GSL_SUCCESS

cdef int ode(double t, double y[], double f[], void *params) nogil:
  """This function matches signature required by gsl_odeiv2_system.f.
  See http://www.gnu.org/software/gsl/manual/html_node/Defining-the-ODE-System.html.
  """
  # unpack Tode struct from params
  cdef Tode * P = <Tode *> params
  # solve ODE at instant t
  P.f(P.n, P.params, t, y, f)
  return GSL_SUCCESS

def solve(params, support, bids):
  """Returns matrix of costs that establish the solution (and equilibrium)
  to the system of ODEs (1.26) in the thesis.

  Arguments (all NumPy arrays):
  params -- list of dicts of parameters (normal distributions)
  initial -- initial condition (common to all bidders)
  bids -- array of bids (t's to solve for)
  """
  cdef int n = len(params) # number of bidders

  # convert list of dicts params into C array of structs
  cdef DistParams * c_params = <DistParams *> calloc(n, sizeof(DistParams))
  if c_params is NULL:
    raise MemoryError()
  cdef int i
  cdef DistParams param
  for i from 0 <= i < n:
    param.location = params[i]['location']
    param.scale = params[i]['scale']
    param.shape = params[i]['shape']
    c_params[i] = param

  # initialize the struct describing system of ODEs
  cdef Tode P
  P.n = n
  P.params = c_params
  P.f = f

  # initialize GSL ODE system
  cdef gsl_odeiv2_system sys
  sys.function = ode
  sys.jacobian = NULL
  sys.dimension = n
  sys.params = &P

  # initialize initial step size (hstart), absolute error (epsAbs),
  # and relative error (epsRel)
  cdef double hstart, epsAbs, epsRel
  hstart = (bids[1] - bids[0]) / 100.0
  epsAbs = epsRel = 1.49012e-8

  # intialize GSL driver
  cdef gsl_odeiv2_driver * d
  d = gsl_odeiv2_driver_alloc_y_new(
      &sys, gsl_odeiv2_step_rkf45,
      hstart, epsAbs, epsRel)

  # populate y_i with initial conditions
  cdef double *y = <double *> calloc(n, sizeof(double))
  if y is NULL:
    raise MemoryError()
  for i from 0 <= i < n:
    y[i] = support[0]

  cdef int status, j
  cdef double t, ti
  # set independent variable to initial point
  t = bids[0]
  # add initial values to the solution set
  sol = [[support[0] for k in range(n)]]

  try:
    # try solving the system at instants in bids array
    for j from 1 <= j < bids.size:
      # advance independent variable
      ti = bids[j]
      # solve the system of ODEs from t to ti
      status = gsl_odeiv2_driver_apply(d, &t, ti, y)

      if status != GSL_SUCCESS:
        # if unsuccessful, raise an error
        # FIX:ME define custom errors
        msg = "Error, return value=%d\n" % status
        raise Exception(msg)

      # create empty array of costs (next row in the solution matrix)
      costs = np.empty(n, dtype=np.float64)
      # populate the array with the contents of y C array
      for i from 0 <= i < n:
        costs[i] = y[i]
      # append to the solution matrix
      sol += [costs]
  finally:
    free(c_params)
    free(y)
    gsl_odeiv2_driver_free(d)

  return np.array(sol)
