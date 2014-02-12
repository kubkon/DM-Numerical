from cython_gsl cimport *
from libc.stdlib cimport calloc, free
from libc.math cimport fabs

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

cdef int solve_ode(gsl_vector_const_view v_uppers,
                   gsl_vector_const_view v_initial,
                   gsl_vector_const_view v_bids,
                   gsl_matrix_view v_costs) nogil:
    cdef const gsl_vector * uppers = &v_uppers.vector
    cdef const gsl_vector * initial = &v_initial.vector
    cdef const gsl_vector * bids = &v_bids.vector
    cdef gsl_matrix * costs = &v_costs.matrix

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

cdef int add_extension(gsl_vector_const_view v_bids,
                       gsl_matrix_view v_costs,
                       gsl_matrix_view v_costs_ext) nogil:
    cdef const gsl_vector * bids = &v_bids.vector
    cdef const gsl_matrix * costs = &v_costs.matrix
    cdef gsl_matrix * costs_ext = &v_costs_ext.matrix

    cdef int i, j
    cdef int k = costs.size2
    cdef double b, c, sums, ext

    for i from 0 <= i < bids.size:
        b = gsl_vector_get(bids, i)

        sums = 0
        for j from 0 <= j < k:
            c = gsl_matrix_get(costs, i, j)
            sums += 1 / (b - c)

        ext = b - (k-1) / sums

        for j from 0 <= j < costs_ext.size2:
            gsl_matrix_set(costs_ext, i, j, ext)

    return GSL_SUCCESS

cdef int min_index(const gsl_vector * v, const double x) nogil:
    """Finds the index of an element in a GSL vector matching
    the value specified as x.

    Arguments:
    v -- input gsl vector of elements to search through
    x -- searched element
    """
    cdef gsl_vector * copy = gsl_vector_calloc(v.size)
    cdef int i, index
    cdef double y

    gsl_vector_memcpy(copy, v)

    gsl_vector_add_constant(copy, (-1) * x)

    for i from 0 <= i < v.size:
        y = gsl_vector_get(copy, i)
        gsl_vector_set(copy, i, fabs(y))

    index = gsl_vector_min_index(copy)

    gsl_vector_free(copy)

    return index

cdef int estimate_k(double b, const gsl_vector * c_lowers) nogil:
    """Estimates the value of k(b). See Algorithm 2.4 in Chapter 2
    of the thesis for more information.

    Arguments:
    b -- estimate of the lower bound on bids
    c_lowers -- array of lower extremities
    """
    cdef int n = c_lowers.size
    cdef int i, j, k = n
    cdef double c = 0
    cdef double sums

    for i from 1 <= i < n:
        sums = 0
        for j from 0 <= j <= i:
            sums += 1 / (b - gsl_vector_get(c_lowers, j))

        c = b - i / sums

        if i < n-1:
            if gsl_vector_get(c_lowers, i) <= c and c < gsl_vector_get(c_lowers, i+1):
                k = i+1
                break

    return k

def p_estimate_k(b, lowers):
    """Python wrapper for estimate_k function.
    """
    cdef int i, n = lowers.size
    cdef gsl_vector * c_lowers = gsl_vector_calloc(n)

    for i from 0 <= i < n:
        gsl_vector_set(c_lowers, i, lowers[i])

    cdef int k = estimate_k(b, c_lowers)

    gsl_vector_free(c_lowers)

    return k

def assert_success(status):
    if status != GSL_SUCCESS:
        # if unsuccessful, raise an error
        # FIX:ME define custom errors
        msg = "Error, return value=%d\n" % status
        raise Exception(msg)

def solve(lowers, uppers, bids):
    """Returns matrix of costs that establish the solution (and equilibrium)
    to the system of ODEs (1.26) in the thesis.

    Arguments (all NumPy arrays):
    lowers -- array of lower extremities
    uppers -- array of upper extremities
    bids -- array of bids (t's to solve for)
    """
    cdef int i, j, k
    cdef int m = bids.size
    cdef int n = lowers.size
    cdef int index
    cdef double prev, curr

    cdef gsl_vector * c_uppers = gsl_vector_calloc(n)
    cdef gsl_vector * initial = gsl_vector_calloc(n)
    for i from 0 <= i < n:
        gsl_vector_set(c_uppers, i, uppers[i])
        gsl_vector_set(initial, i, lowers[i])

    cdef gsl_vector * c_bids = gsl_vector_calloc(m)
    for i from 0 <= i < m:
        gsl_vector_set(c_bids, i, bids[i])

    cdef gsl_matrix * c_costs = gsl_matrix_calloc(m, n)

    cdef gsl_vector_view v

    # estimate k
    k = estimate_k(bids[0], initial)

    # try solving the system at instants in bids array
    status = solve_ode(gsl_vector_const_subvector(c_uppers, 0, k),
                       gsl_vector_const_subvector(initial, 0, k),
                       gsl_vector_const_subvector(c_bids, 0, m),
                       gsl_matrix_submatrix(c_costs, 0, 0, m, n))

    assert_success(status)

    while k < n:
        # compute bidding extension
        status = add_extension(gsl_vector_const_subvector(c_bids, 0, m),
                               gsl_matrix_submatrix(c_costs, 0, 0, m, k),
                               gsl_matrix_submatrix(c_costs, 0, k, m, n-k))

        # find index of truncation
        v = gsl_matrix_column(c_costs, k)
        index = min_index(&v.vector, lowers[k])

        # set new initial conditions
        for j from 0 <= j < k+1:
            gsl_vector_set(initial, j, gsl_matrix_get(c_costs, index, j))

        # solve system at new initial conditions
        status = solve_ode(gsl_vector_const_subvector(c_uppers, 0, k+1),
                           gsl_vector_const_subvector(initial, 0, k+1),
                           gsl_vector_const_subvector(c_bids, index, m-index),
                           gsl_matrix_submatrix(c_costs, index, 0, m-index, n))

        assert_success(status)

        # increment k
        k += 1

    costs = np.empty((m, n), np.float)

    for i from 0 <= i < m:
        for j from 0 <= j < n:
            costs[i][j] = gsl_matrix_get(c_costs, i, j)

    gsl_vector_free(c_uppers)
    gsl_vector_free(initial)
    gsl_vector_free(c_bids)
    gsl_matrix_free(c_costs)

    return costs
