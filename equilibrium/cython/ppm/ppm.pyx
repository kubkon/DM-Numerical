from cython_gsl cimport *

cimport libc.math as m

import numpy as np

from libc.stdio cimport printf

# C struct
# specifies the minimization system
ctypedef struct Tmin:
    int n, # number of bidders
    int k, # number of polynomial coefficients per bidder
    int granularity, # grid granularity
    double b_upper, # upper bound on bids
    gsl_vector * lower_exts, # gsl_vector of lower extremities
    gsl_vector * upper_exts, # gsl_vector of upper extremities
    double f(int, int, double, double, const gsl_vector *,
             const gsl_vector *, const gsl_vector *) nogil # pointer to objective function


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
    # Get number of bidders
    cdef int n = lower_exts.size

    # Generate grid of linearly spaced points
    cdef gsl_vector * grid
    grid = c_linspace(b_lower, b_upper, granularity)

    # Compute first-order condition function over the grid
    # for each bidder
    cdef double sums = 0
    cdef double b, deriv, cost, lower_ext, upper_ext, g
    cdef int i, j, l
    cdef gsl_vector_view v_view
    cdef gsl_vector * v
    cdef double summed, t_lower_ext
    cdef gsl_vector * t_v

    cdef int z

    for i from 0 <= i < n:
        # Extract vector of polynomial coefficients for
        # bidder i
        v_view = gsl_vector_subvector(vs, i*k, k)
        v = &v_view.vector
        for z from 0 <= z < k:
            printf("%f", gsl_vector_get(v, z))
        printf('\n')
        # Get lower and upper extremities for the current
        # bidder
        lower_ext = gsl_vector_get(lower_exts, i)
        upper_ext = gsl_vector_get(upper_exts, i)

        for j from 0 <= j < granularity:
            # Get bid value
            b = gsl_vector_get(grid, j)
            # Get cost derivative value
            deriv = c_deriv_cost_function(b_lower, v, b)
            # Get cost value
            cost = c_cost_function(lower_ext, b_lower, v, b)
            # Calculate first-order condition at b
            summed = 0
            for l from 0 <= l < n:
                v_view = gsl_vector_subvector(vs, l*k, k)
                t_v = &v_view.vector
                t_lower_ext = gsl_vector_get(lower_exts, l)
                summed += 1 / (b - c_cost_function(t_lower_ext, b_lower, t_v, b))

            g = deriv - (upper_ext - cost) * (1/(n-1) * summed - 1/(b - cost))
            sums += m.pow(g, 2)

        # Add upper boundary condition
        cost = c_cost_function(lower_ext, b_lower, v, b_upper)
        sums += granularity * m.pow(b_upper - cost, 2)

    return sums

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
    # Extract minimization system
    cdef Tmin * P = <Tmin *> params

    # Extract lower bound on bids
    cdef double b_lower = gsl_vector_get(v, 0)

    # Extract polynomial coefficients
    cdef int i
    cdef int m = P.k * P.n
    cdef double value

    cdef gsl_vector * vs
    vs = gsl_vector_alloc(m)

    for i from 0 <= i < m:
        value = gsl_vector_get(v, i+1)
        gsl_vector_set(vs, i, value)

    return P.f(P.k, P.granularity, b_lower, P.b_upper, P.lower_exts, P.upper_exts, vs)


def solve (b_lower, b_upper, lowers, uppers, poly_coeffs, granularity=1000):
    # Get number of bidders
    cdef int n = len(lowers)
    # Get number of polynomial coefficients per bidder
    cdef int k = len(poly_coeffs)
    # Flatten polynomial coefficients
    poly_coeffs_flat = []
    for p in poly_coeffs:
        poly_coeffs_flat += p

    # Initialize struct describing minimization system
    cdef Tmin P
    P.k = k
    P.n = n
    P.granularity = granularity
    P.b_upper = b_upper

    cdef gsl_vector * lower_exts
    lower_exts = gsl_vector_alloc(n)
    cdef gsl_vector * upper_exts
    upper_exts = gsl_vector_alloc(n)

    cdef int i
    for i from 0 <= i < n:
        gsl_vector_set(lower_exts, i, lowers[i])
        gsl_vector_set(upper_exts, i, uppers[i])
    P.lower_exts = lower_exts
    P.upper_exts = upper_exts

    P.f = c_objective_function

    cdef gsl_multimin_function my_func
    my_func.n = n*k + 1
    my_func.f = &min_f
    my_func.params = &P

    cdef size_t iter = 0
    cdef int max_iter = 100
    cdef int status

    cdef const gsl_multimin_fminimizer_type * T
    cdef gsl_multimin_fminimizer * s
    cdef gsl_vector * ss
    cdef gsl_vector * x

    # Starting point
    x = gsl_vector_alloc(my_func.n)
    gsl_vector_set(x, 0, b_lower)
    for i from 0 <= i < (my_func.n - 1):
        gsl_vector_set(x, i+1, poly_coeffs_flat[i])
    
    # Set initial step size to 0.1
    ss = gsl_vector_alloc(my_func.n)
    gsl_vector_set_all(ss, 0.1)
    
    T = gsl_multimin_fminimizer_nmsimplex2
    s = gsl_multimin_fminimizer_alloc(T, my_func.n)

    gsl_multimin_fminimizer_set(s, &my_func, x, ss)

    status = GSL_CONTINUE

    while (status == GSL_CONTINUE and iter <= max_iter):
        iter += 1
        status = gsl_multimin_fminimizer_iterate(s)

        if status:
            break

        size = gsl_multimin_fminimizer_size(s)
        status = gsl_multimin_test_size(size, 1e-6)

    b_lower = gsl_vector_get(s.x, 0)

    for i from 0 <= i < (my_func.n - 1):
        poly_coeffs_flat[i] = gsl_vector_get(s.x, i+1)
    poly_coeffs = [poly_coeffs_flat[j:j+k] for j in range(0, my_func.n - 1, k)]

    gsl_multimin_fminimizer_free(s)
    gsl_vector_free(x)
    gsl_vector_free(ss)

    return (b_lower, poly_coeffs)
