from cython_gsl cimport *
from libc.stdlib cimport calloc, free
import numpy as np

# C struct
# specifies the system of ODE
ctypedef struct Tode:
  int n # number of bidders
  double * uppers # array of upper extremities
  int f(int, double *, double, double *, double *) nogil # pointer to function describing system of ODEs

cdef int f(int n, double * uppers, double t, double * y, double * f) nogil:
  """Evolves system of ODEs at a particular independent variable t,
  and for a vector of particular dependent variables y_i(t). Mathematically,
  dy_i(t)/dt = f_i(t, y_1(t), ..., y_n(t)).

  Arguments:
  n -- number of bidders
  uppers -- array of upper extremities
  t -- independent variable
  y -- array of dependent variables
  f -- array of solved vector elements f_i(..)
  """
  cdef double * rs = <double *> calloc(n, sizeof(double))
  cdef int i
  cdef double r, rs_sum
  rs_sum = 0

  for i from 0 <= i < n:
    r = t - y[i]

    if r == 0:
      return GSL_EZERODIV

    rs[i] = 1 / r
    rs_sum += 1 / r

  # this loop corresponds to the system of equations (1.26) in the thesis
  for i from 0 <= i < n:
    f[i] = (uppers[i] - y[i]) * (rs_sum / (n-1) - rs[i])

  free(rs)
  return GSL_SUCCESS

cdef int ode(double t, double y[], double f[], void *params) nogil:
  """This function matches signature required by gsl_odeiv2_system.f.
  See http://www.gnu.org/software/gsl/manual/html_node/Defining-the-ODE-System.html.
  """
  # unpack Tode struct from params
  cdef Tode * P = <Tode *> params
  # solve ODE at instant t
  P.f(P.n, P.uppers, t, y, f)
  return GSL_SUCCESS

def solve(initial, uppers, bids):
  """Returns matrix of costs that establish the solution (and equilibrium)
  to the system of ODEs (1.26) in the thesis.

  Arguments (all NumPy arrays):
  initial -- array of initial values y_i
  uppers -- array of upper extremities
  bids -- array of bids (t's to solve for)
  """
  cdef int n = initial.size # number of bidders

  # convert NumPy array of uppers into C array
  cdef double * c_uppers = <double *> calloc(n, sizeof(double))
  if c_uppers is NULL:
    raise MemoryError()
  cdef int i
  for i from 0 <= i < n:
    c_uppers[i] = uppers[i]

  # initialize the struct describing system of ODEs
  cdef Tode P
  P.n = n
  P.uppers = c_uppers
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
  
  # set maximum number of steps per bid evolution
  gsl_odeiv2_driver_set_nmax(d, 10000)

  # populate y_i with initial conditions
  cdef double *y = <double *> calloc(n, sizeof(double))
  if y is NULL:
    raise MemoryError()
  for i from 0 <= i < n:
    y[i] = initial[i]

  cdef int status, j
  cdef double t, ti
  # set independent variable to initial point
  t = bids[0]
  # add initial values to the solution set
  sol = [initial]

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
    free(c_uppers)
    free(y)
    gsl_odeiv2_driver_free(d)

  return np.array(sol)
