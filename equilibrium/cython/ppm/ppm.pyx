from cython_gsl cimport *

from libc.stdlib cimport calloc, free
from libc.stdio cimport printf
cimport libc.math as m

import numpy as np

cdef double c_cost_function(double low_ext,
                            double b_lower,
                            const gsl_vector * v,
                            double b) nogil:
    """
    Defines polynomial function that approximates equilibrium cost
    function for each bidder.

    Arguments:
    low_ext -- lower extremity
    b_lower -- lower bound on bids
    v -- gsl_vector of polynomial coefficients
    b -- bid value
    """
    cdef size_t k = v.size
    cdef int i
    cdef double sums = 0

    for i from 0 <= i < k:
        sums += gsl_vector_get(v, i) * m.pow(b - b_lower, i+1)
    
    return low_ext + sums

def cost_function(low_ext, b_lower, v, b):
    """
    Python wrapper for c_cost_function. This function exists
    mainly for internal (testing) purposes. It should not be used
    as a standalone function.
    """
    cdef double c_low_ext, c_b_lower, c_b
    c_low_ext = low_ext
    c_b_lower = b_lower
    c_b = b

    cdef gsl_vector * c_v
    n = len(v)
    c_v = gsl_vector_alloc(n)
    for i in range(n):
        gsl_vector_set(c_v, i, v[i])

    return c_cost_function(c_low_ext, c_b_lower, c_v, c_b)


cdef double c_deriv_cost_function(double b_lower,
                                  const gsl_vector * v,
                                  double b) nogil:
    """
    Defines derivative of the polynomial cost function, c_cost_function.

    Arguments:
    b_lower -- lower bound on bids
    v -- gsl_vector of polynomial coefficients
    b -- bid value
    """
    cdef size_t k = v.size
    cdef int i
    cdef double sums = 0

    for i from 0 <= i < k:
        sums += (i+1) * gsl_vector_get(v, i) * m.pow(b - b_lower, i)

    return sums

def deriv_cost_function(b_lower, v, b):
    """
    Python wrapper for c_deriv_cost_function. This function exists
    mainly for internal (testing) purposes. It should not be used
    as a standalone function.
    """
    cdef double c_b_lower, c_b
    c_b_lower = b_lower
    c_b = b

    cdef gsl_vector * c_v
    n = len(v)
    c_v = gsl_vector_alloc(n)
    for i in range(n):
        gsl_vector_set(c_v, i, v[i])

    return c_deriv_cost_function(c_b_lower, c_v, c_b)


cdef gsl_vector * c_linspace(double begin, double end, int granularity) nogil:
    """
    Returns gsl_vector of linearly spaced points such that for each x
    in the gsl_vector, begin <= x <= end, and for any two points x1, x2
    in the gsl_vector such that x1 < x2, x2 - x1 =

    Attributes:
    begin -- starting point of the linearly spaced grid
    end -- end point
    granularity -- number of points in the grid (begin and end inclusive)
    """
    cdef gsl_vector * v
    v = gsl_vector_alloc(granularity)

    cdef double step, value
    cdef int i

    step = (end - begin) / (granularity - 1)

    for i from 0 <= i < granularity:
        value = begin + step * i
        gsl_vector_set(v, i, value)

    return v

def linspace(begin, end,  granularity):
    """
    Python wrapper for c_linspace. This function exists mainly for internal
    (testing) purposes. It should not be used as a standalone function.
    """
    cdef double c_begin, c_end
    cdef int c_granularity
    c_begin = begin
    c_end = end
    c_granularity = granularity

    cdef gsl_vector * v
    v = c_linspace(c_begin, c_end, c_granularity)

    cdef int n = v.size
    cdef int i
    output = []

    for i from 0 <= i < n:
        output += [gsl_vector_get(v, i)]

    return output


cdef double c_objective_function(int k,
                                 int granularity,
                                 double b_lower,
                                 double b_upper,
                                 const gsl_vector * lower_exts,
                                 const gsl_vector * upper_exts,
                                 const gsl_vector * vs) nogil:
    """
    Defines objective function for the nonlinear minimization problem.

    Arguments:
    k -- number of polynomial coefficients per bidder
    granularity -- grid granularity
    b_lower -- lower bound on bids
    b_upper -- upper bound on bids
    lower_exts -- gsl_vector of lower extremities
    upper_exts -- gsl_vector of upper extremities
    vs -- gsl_vector of polynomial coefficients for each bidder
    """
    cdef double sums = 0
    cdef size_t n = lower_exts.size
    cdef int i
    cdef gsl_vector_view v_view
    cdef gsl_vector * v

    for i from 0 <= i < n:
        v_view = gsl_vector_subvector(vs, i*k, k)
        v = &v_view.vector
        for j from 0 <= j < k:
            printf("%f", gsl_vector_get(v, j))

    return 1.0

def objective_function(k, granularity, b_lower, b_upper,
                       lower_exts, upper_exts, vs):
    """
    Python wrapper for c_objective_function.
    """
    cdef int c_k, c_granularity
    cdef double c_b_lower, c_b_upper
    c_k = k
    c_granularity = granularity
    c_b_lower = b_lower
    c_b_upper = b_upper

    cdef gsl_vector * c_lower_exts, * c_upper_exts, * c_vs
    n = len(lower_exts)
    c_lower_exts = gsl_vector_alloc(n)
    c_upper_exts = gsl_vector_alloc(n)
    for i in range(n):
        gsl_vector_set(c_lower_exts, i, lower_exts[i])
        gsl_vector_set(c_upper_exts, i, upper_exts[i])

    n = len(vs)
    c_vs = gsl_vector_alloc(n)
    for i in range(n):
        gsl_vector_set(c_vs, i, vs[i])

    return c_objective_function(c_k, c_granularity, c_b_lower,
                                c_b_upper, c_lower_exts, c_upper_exts,
                                c_vs)


cdef double min_f(const gsl_vector * v, void * params) nogil:
    cdef double x, y

    x = gsl_vector_get(v, 0)
    y = gsl_vector_get(v, 1)

    return x*x + y*y

def solve (bids, b_lower, poly_coeffs):

    cdef size_t iter = 0
    cdef int max_iter = 100
    cdef int status

    cdef const gsl_multimin_fminimizer_type * T
    cdef gsl_multimin_fminimizer * s
    cdef gsl_vector * ss
    cdef gsl_vector * x

    cdef gsl_multimin_function my_func
    my_func.n = 2
    my_func.f = &min_f
    my_func.params = NULL

    # Starting point (10,10)
    x = gsl_vector_alloc(2)
    gsl_vector_set(x, 0, 10.0)
    gsl_vector_set(x, 1, 10.0)

    # Set initial step size to 0.1
    ss = gsl_vector_alloc(2)
    gsl_vector_set_all(ss, 0.1)

    T = gsl_multimin_fminimizer_nmsimplex2
    s = gsl_multimin_fminimizer_alloc(T, 2)

    gsl_multimin_fminimizer_set(s, &my_func, x, ss)

    status = GSL_CONTINUE

    while (status == GSL_CONTINUE and iter <= max_iter):
        iter += 1
        status = gsl_multimin_fminimizer_iterate(s)

        if status:
            break

        size = gsl_multimin_fminimizer_size(s)
        status = gsl_multimin_test_size(size, 1e-2)

        if status == GSL_SUCCESS:
            print("Minimum found at:\n")

        print("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n" %\
              (iter, gsl_vector_get (s.x, 0), gsl_vector_get (s.x, 1), s.fval, size))

    gsl_multimin_fminimizer_free(s)
    gsl_vector_free(x)
    gsl_vector_free(ss)

    return (b_lower, poly_coeffs)
