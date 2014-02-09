from cython_gsl cimport *
from libc.stdlib cimport calloc, free
import numpy as np

# C struct
# specifies the system of ODE
ctypedef struct Tode:
    # number of bidders
    int n
    # gsl vector of upper extremities
    const gsl_vector * uppers
    # pointer to function describing system of ODEs
    int f(int, const gsl_vector *, double, double *, double *) nogil

cdef int f(int n, const gsl_vector * uppers, double t, double * y, double * f) nogil:
    """Evolves system of ODEs at a particular independent variable t,
    and for a vector of particular dependent variables y_i(t). Mathematically,
    dy_i(t)/dt = f_i(t, y_1(t), ..., y_n(t)).

    Arguments:
    n -- number of bidders
    uppers -- gsl vector of upper extremities
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
        f[i] = (gsl_vector_get(uppers, i) - y[i]) * (rs_sum / (n-1) - rs[i])

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

cdef int estimate_k(double b, const gsl_vector *lowers) nogil:
    """This function estimates the value of k(b). See Algorithm 2.4 in Chapter 2
    of the thesis for more information.

    Arguments:
    b -- estimate of the lower bound on bids
    lowers -- gsl vector of lower extremities
    n -- number of bidders
    """
    cdef int n = lowers.size
    cdef int i, j, k = n
    cdef double sums, c = 0

    for i from 1 <= i < n:
        sums = 0
        for j from 0 <= j <= i:
            sums += 1 / (b - gsl_vector_get(lowers, j))

        c = b - i / sums

        if i < n-1:
            if gsl_vector_get(lowers, i) <= c and c < gsl_vector_get(lowers, i+1):
                k = i+1
                break

    return k

cdef int solve_ode(gsl_vector_const_view v_initial,
                   gsl_vector_const_view v_uppers,
                   const gsl_vector * bids,
                   gsl_matrix * costs) nogil:
    cdef const gsl_vector * initial = &v_initial.vector
    cdef const gsl_vector * uppers = &v_uppers.vector

    cdef int i, j
    cdef int m = bids.size
    cdef int n = initial.size

    # initialize the struct describing system of ODEs
    cdef Tode P
    P.n = n
    P.uppers = uppers
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
    hstart = (gsl_vector_get(bids, 1) - gsl_vector_get(bids, 0)) / 100.0
    epsAbs = epsRel = 1.49012e-8

    # intialize GSL driver
    cdef gsl_odeiv2_driver * d
    d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rkf45,
        hstart, epsAbs, epsRel)

    # populate y_i with initial conditions
    cdef double *y = <double *> calloc(n, sizeof(double))
    if y is NULL:
        return GSL_ENOMEM

    for i from 0 <= i < n:
        y[i] = gsl_vector_get(initial, i)

    cdef int status
    cdef double t, ti
    # set independent variable to initial point
    t = gsl_vector_get(bids, 0)
    # add initial values to the solution set
    for j from 0 <= j < n:
        gsl_matrix_set(costs, 0, j, gsl_vector_get(initial, j))

    # solve the system at instants in bids array
    for i from 1 <= i < m:
        # advance independent variable
        ti = gsl_vector_get(bids, i)
        # solve the system of ODEs from t to ti
        status = gsl_odeiv2_driver_apply(d, &t, ti, y)

        if status != GSL_SUCCESS:
            return status

        # add result to the solution matrix
        for j from 0 <= j < n:
            gsl_matrix_set(costs, i, j, y[j])

    free(y)
    gsl_odeiv2_driver_free(d)

    return GSL_SUCCESS

def solve(lowers, uppers, bids):
    """Returns matrix of costs that establish the solution (and equilibrium)
    to the system of ODEs (1.26) in the thesis.

    Arguments (all NumPy arrays):
    lowers -- array of lower extremities
    uppers -- array of upper extremities
    bids -- array of bids (t's to solve for)
    """
    cdef int i, j
    cdef int m = bids.size
    cdef int n = lowers.size

    cdef gsl_vector * c_lowers = gsl_vector_calloc(n)
    cdef gsl_vector * c_uppers = gsl_vector_calloc(n)
    for i from 0 <= i < n:
        gsl_vector_set(c_lowers, i, lowers[i])
        gsl_vector_set(c_uppers, i, uppers[i])

    cdef gsl_vector * c_bids = gsl_vector_calloc(m)
    for i from 0 <= i < m:
        gsl_vector_set(c_bids, i, bids[i])

    cdef gsl_matrix * c_costs = gsl_matrix_calloc(m, n)

    try:
        # try solving the system at instants in bids array
        status = solve_ode(gsl_vector_const_subvector(c_lowers, 0, n),
                           gsl_vector_const_subvector(c_uppers, 0, n),
                           c_bids,
                           c_costs)

        if status != GSL_SUCCESS:
            # if unsuccessful, raise an error
            # FIX:ME define custom errors
            msg = "Error, return value=%d\n" % status
            raise Exception(msg)

        costs = np.empty((m, n), np.float)

        for i from 0 <= i < m:
            for j from 0 <= j < n:
                costs[i][j] = gsl_matrix_get(c_costs, i, j)

    finally:
        gsl_vector_free(c_lowers)
        gsl_vector_free(c_uppers)
        gsl_vector_free(c_bids)
        gsl_matrix_free(c_costs)

    return costs
