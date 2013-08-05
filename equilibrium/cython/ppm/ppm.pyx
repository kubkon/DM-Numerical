from cython_gsl cimport *

from libc.stdlib cimport calloc, free
import numpy as np

cdef double f (const gsl_vector * v, void * params) nogil:
    cdef double x, y

    x = gsl_vector_get(v, 0)
    y = gsl_vector_get(v, 1)

    return x*x + y*y

def solve(bids, b_lower, poly_coeffs):

    cdef size_t iter = 0
    cdef int max_iter = 100
    cdef int status

    cdef const gsl_multimin_fminimizer_type * T
    cdef gsl_multimin_fminimizer * s
    cdef gsl_vector * ss
    cdef gsl_vector * x

    cdef gsl_multimin_function my_func
    my_func.n = 2
    my_func.f = &f
    my_func.params = NULL

    # Starting point (10,10)
    x = gsl_vector_alloc (2)
    gsl_vector_set (x, 0, 10.0)
    gsl_vector_set (x, 1, 10.0)

    # Set initial step size to 0.1
    ss = gsl_vector_alloc (2)
    gsl_vector_set_all (ss, 0.1)

    T = gsl_multimin_fminimizer_nmsimplex2
    s = gsl_multimin_fminimizer_alloc (T, 2)

    gsl_multimin_fminimizer_set (s, &my_func, x, ss)

    status = GSL_CONTINUE

    while (status == GSL_CONTINUE and iter <= max_iter):
        iter += 1
        status = gsl_multimin_fminimizer_iterate (s)

        if status:
            break

        size = gsl_multimin_fminimizer_size (s)
        status = gsl_multimin_test_size (size, 1e-2)

        if status == GSL_SUCCESS:
            print("Minimum found at:\n")

        print("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n" %\
              (iter, gsl_vector_get (s.x, 0), gsl_vector_get (s.x, 1), s.fval, size))

    gsl_multimin_fminimizer_free (s)
    gsl_vector_free (x)
    gsl_vector_free (ss)

    return (b_lower, poly_coeffs)
