from cython_gsl cimport *

from libc.stdlib cimport calloc, free
from libc.math cimport exp, sqrt, pow, erf

from bajari.dists.dists cimport trunc_normal_pdf, trunc_normal_cdf

import numpy as np

# C structs
# describes distribution params
ctypedef struct DistParams:
    double location
    double scale
    double a
    double b

# specifies the minimization system
ctypedef struct Tmin:
    int n, # number of bidders
    int k, # number of polynomial coefficients per bidder
    int granularity, # grid granularity
    const DistParams * dist_params, # distribution params for each bidder
    const double * support, # support range
    double f(int, int, int, double, const DistParams *,
             const double *, const gsl_vector *) nogil # pointer to objective function

cdef double c_cost_function(double b_lower,
                          const gsl_vector * v,
                          double b) nogil:
    """
    Defines polynomial function that approximates equilibrium cost
    function for each bidder.

    Arguments:
    b_lower -- lower bound on bids
    v -- gsl_vector of polynomial coefficients
    b -- bid value
    """
    cdef size_t k = v.size
    cdef int i
    cdef double diff = b - b_lower
    cdef double sums = 0

    # Case i in {0}
    sums += gsl_vector_get(v, 0)
    # Remaining cases
    for i from 1 <= i < k:
        sums += gsl_vector_get(v, i) * pow(diff, i)

    return b_lower + sums

def cost_function(b_lower, v, b):
    """
    Python wrapper for c_cost_function. This function exists
    mainly for internal (testing) purposes. It should not be used
    as a standalone function.
    """
    cdef double c_b_lower, c_b, c_cost
    c_b_lower = b_lower
    c_b = b

    cdef gsl_vector * c_v
    n = len(v)
    c_v = gsl_vector_alloc(n)
    for i in range(n):
        gsl_vector_set(c_v, i, v[i])

    c_cost = c_cost_function(c_b_lower, c_v, c_b)

    gsl_vector_free(c_v)

    return c_cost


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
    cdef double diff = b - b_lower
    cdef double sums = 0

    # Cases i in {0, 1}
    sums += gsl_vector_get(v, 1)
    # Remaining cases
    for i from 2 <= i < k:
        sums += i * gsl_vector_get(v, i) * pow(diff, i-1)

    return sums

def deriv_cost_function(b_lower, v, b):
    """
    Python wrapper for c_deriv_cost_function. This function exists
    mainly for internal (testing) purposes. It should not be used
    as a standalone function.
    """
    cdef double c_b_lower, c_b, c_deriv
    c_b_lower = b_lower
    c_b = b

    cdef gsl_vector * c_v
    n = len(v)
    c_v = gsl_vector_alloc(n)
    for i in range(n):
        gsl_vector_set(c_v, i, v[i])

    c_deriv = c_deriv_cost_function(c_b_lower, c_v, c_b)

    gsl_vector_free(c_v)

    return c_deriv


cdef gsl_vector * c_linspace(double begin, double end, int granularity) nogil:
    """
    Returns gsl_vector of linearly spaced points such that for each x
    in the gsl_vector, begin <= x < end, and for any two points x1, x2
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

    gsl_vector_free(v)

    return output


cdef double c_objective_function(int n,
                               int k,
                               int granularity,
                               double b_lower,
                               const DistParams * dist_params,
                               const double * support,
                               const gsl_vector * vs) nogil:
    """
    Defines objective function for the nonlinear minimization problem.

    Arguments:
    n -- number of bidders
    k -- number of polynomial coefficients per bidder
    granularity -- grid granularity
    b_lower -- lower bound on bids
    normal_params -- array of params describing distributions of each bidder
    support -- array of min and max values of the support set for each bidder
    vs -- gsl_vector of polynomial coefficients for each bidder
    """
    # Generate grid of linearly spaced points
    cdef gsl_vector * grid
    cdef double b_upper = support[1]
    grid = c_linspace(b_lower+1e-6, b_upper, granularity)

    # Compute first-order condition function over the grid
    # for each bidder
    cdef double sums = 0
    cdef double b, deriv, cost, cdf, pdf, prob, g
    cdef int i, j, l
    cdef DistParams dist_param
    cdef gsl_vector_view v_view
    cdef gsl_vector * v
    cdef double summed, t_cost
    cdef gsl_vector * t_v
    cdef gsl_vector_view t_v_view

    for i from 0 <= i < n:
        # Extract vector of polynomial coefficients for
        # bidder i
        v_view = gsl_vector_subvector(vs, i*k, k)
        v = &v_view.vector
        # Get dist params for the current bidder
        dist_param = dist_params[i]

        for j from 0 <= j < granularity:
            # Get bid value
            b = gsl_vector_get(grid, j)
            # Get cost derivative value
            deriv = c_deriv_cost_function(b_lower, v, b)
            # Get cost value
            cost = c_cost_function(b_lower, v, b)
            # Calculate probabilities for bidder i
            cdf = trunc_normal_cdf(cost, dist_param.location, dist_param.scale, dist_param.a, dist_param.b)
            pdf = trunc_normal_pdf(cost, dist_param.location, dist_param.scale, dist_param.a, dist_param.b)
            prob = (1 - cdf) / pdf
            # Calculate first-order condition at b
            summed = 0
            for l from 0 <= l < n:
                if l == i:
                    continue
                t_v_view = gsl_vector_subvector(vs, l*k, k)
                t_v = &t_v_view.vector
                t_cost = c_cost_function(b_lower, t_v, b)
                summed += (1 / (b - t_cost))

            g = deriv - (prob / (n-1)) * (summed + (2-n) / (b - cost))
            sums += pow(g, 2)

        # Add lower boundary condition
        cost = c_cost_function(b_lower, v, b_lower)
        sums += granularity * pow(support[0] - cost, 2)

        # Add upper boundary condition
        cost = c_cost_function(b_lower, v, b_upper)
        sums += granularity * pow(b_upper - cost, 2)

    gsl_vector_free(grid)

    return sums


cdef double min_f(const gsl_vector * v, void * params) nogil:
    # Extract minimization system
    cdef Tmin * P = <Tmin *> params

    # Extract lower bound on bids
    cdef double b_lower = gsl_vector_get(v, 0)

    # Extract polynomial coefficients
    cdef int i
    cdef int m = P.k * P.n
    cdef double value, min_f_output

    cdef gsl_vector * vs
    vs = gsl_vector_alloc(m)

    for i from 0 <= i < m:
        value = gsl_vector_get(v, i+1)
        gsl_vector_set(vs, i, value)

    min_f_output = P.f(P.n, P.k, P.granularity, b_lower, P.dist_params, P.support, vs)

    gsl_vector_free(vs)

    return min_f_output


def solve (b_lower, support, params, poly_coeffs, size_box=None, granularity=100):
    # Get number of bidders
    cdef int n = len(params)
    # Get number of polynomial coefficients per bidder
    cdef int k = len(poly_coeffs[0])
    # Flatten polynomial coefficients
    poly_coeffs_flat = []
    for p in poly_coeffs:
        poly_coeffs_flat += p

    # Initialize struct describing minimization system
    cdef Tmin P
    P.k = k
    P.n = n
    P.granularity = granularity
    P.f = c_objective_function
    
    cdef DistParams * dist_params
    dist_params = <DistParams *> calloc(n, sizeof(DistParams))
    cdef int i
    for i from 0 <= i < n:
        dist_params[i].location = params[i]['loc']
        dist_params[i].scale = params[i]['scale']
        dist_params[i].a = support[0]
        dist_params[i].b = support[1]
    P.dist_params = dist_params

    cdef int supp_size = len(support)
    cdef double * supp = <double *> calloc(supp_size, sizeof(double))
    for i from 0 <= i < supp_size:
        supp[i] = support[i]
    P.support = supp

    cdef gsl_multimin_function my_func
    my_func.n = n*k + 1
    my_func.f = &min_f
    my_func.params = &P

    cdef size_t iterator = 0
    cdef int max_iter = 10000
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
    
    # Set initial step size
    ss = gsl_vector_alloc(my_func.n)
    if size_box:
        for i from 0 <= i < my_func.n:
            gsl_vector_set(ss, i, size_box[i])
    else:
        # If undefined, set to 0.1
        gsl_vector_set_all(ss, 0.1)
    
    T = gsl_multimin_fminimizer_nmsimplex2
    s = gsl_multimin_fminimizer_alloc(T, my_func.n)

    gsl_multimin_fminimizer_set(s, &my_func, x, ss)

    status = GSL_CONTINUE

    while (status == GSL_CONTINUE and iterator < max_iter):
        iterator += 1
        status = gsl_multimin_fminimizer_iterate(s)

        if status:
            break

        size = gsl_multimin_fminimizer_size(s)
        status = gsl_multimin_test_size(size, 1e-8)

        printf("%5d %.5f %5.10f %.10f\n",\
                iterator,\
                gsl_vector_get(s.x, 0),\
                s.fval,\
                size)

    b_lower = gsl_vector_get(s.x, 0)
    
    for i from 0 <= i < (my_func.n - 1):
        poly_coeffs_flat[i] = gsl_vector_get(s.x, i+1)

    poly_coeffs = [poly_coeffs_flat[j:j+k] for j in range(0, my_func.n - 1, k)]

    gsl_multimin_fminimizer_free(s)
    gsl_vector_free(x)
    gsl_vector_free(ss)

    free(dist_params)
    free(supp)

    return (b_lower, poly_coeffs)
