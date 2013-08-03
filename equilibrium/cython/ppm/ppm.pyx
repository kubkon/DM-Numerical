from cython_gsl cimport *
from gsl_multimin cimport *
from libc.stdlib cimport calloc, free
import numpy as np

cdef double f(const gsl_vector * v, void * params) nogil:
    cdef double x, y

    x = gsl_vector_get(v, 0)
    y = gsl_vector_get(v, 1)

    return x*x + y*y

def solve(bids, b_lower, poly_coeffs):

    cdef gsl_multimin_function my_func
    my_func.n = 2
    my_func.f = &f
    my_func.params = NULL

    cdef size_t iter = 0
    cdef int status
    cdef double size
    cdef const gsl_multimin_fminimizer_type * T

    return (b_lower, poly_coeffs)