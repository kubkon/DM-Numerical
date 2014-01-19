from cython_gsl cimport *

from libc.stdlib cimport calloc, free
from libc.math cimport pow

import numpy as np


# struct describing data points
ctypedef struct Data:
    int n
    int p
    double * xs
    double * ys

cdef int poly_f(const gsl_vector * x, void * data, gsl_vector * f) nogil:
    cdef Data * d = <Data *> data
    cdef int n = d.n
    cdef int p = d.p
    cdef double * xs = d.xs
    cdef double * ys = d.ys

    cdef double y
    cdef int i

    for i from 0 <= i < n:
        y = gsl_poly_eval(x.data, p, xs[i])
        gsl_vector_set(f, i, (y - ys[i]))

    return GSL_SUCCESS

cdef int poly_df(const gsl_vector * x, void * data, gsl_matrix * J) nogil:
    cdef Data * d = <Data *> data
    cdef int n = d.n
    cdef int p = d.p
    cdef double * xs = d.xs

    cdef int i, j

    for i from 0 <= i < n:
        for j from 0 <= j < p:
            gsl_matrix_set(J, i, j, pow(xs[i], j))

    return GSL_SUCCESS

cdef int poly_fdf(const gsl_vector * x,
                  void * data,
                  gsl_vector * f,
                  gsl_matrix * J) nogil:
    poly_f(x, data, f)
    poly_df(x, data, J)

    return GSL_SUCCESS


def fit(xs, ys, num_coeffs=3, maxiter=500):
    cdef const gsl_multifit_fdfsolver_type * T
    cdef gsl_multifit_fdfsolver * s
    cdef int status = GSL_CONTINUE
    cdef int i
    cdef int iter = 0
    cdef int c_maxiter = maxiter

    # convert input data to C
    cdef int n = xs.size
    cdef int p = num_coeffs
    cdef double * c_xs = <double *> calloc(n, sizeof(double))
    cdef double * c_ys = <double *> calloc(n, sizeof(double))

    for i from 0 <= i < n:
        c_xs[i] = xs[i]
        c_ys[i] = ys[i]

    cdef Data params
    params.n = n
    params.p = p
    params.xs = c_xs
    params.ys = c_ys

    cdef gsl_multifit_function_fdf f
    f.f = &poly_f
    f.df = &poly_df
    f.fdf = &poly_fdf
    f.n = n
    f.p = p
    f.params = &params

    # specify initial guess
    cdef double * x_init = <double *> calloc(p, sizeof(double))
    
    for i from 0 <= i < p:
        x_init[i] = 1e-1

    cdef gsl_vector_view x = gsl_vector_view_array(x_init, p)

    T = gsl_multifit_fdfsolver_lmsder
    s = gsl_multifit_fdfsolver_alloc(T, n, p)
    gsl_multifit_fdfsolver_set(s, &f, &x.vector)

    while (status == GSL_CONTINUE and iter < c_maxiter):
        iter += 1
        status = gsl_multifit_fdfsolver_iterate(s)

        if status:
            break

        status = gsl_multifit_test_delta(s.dx, s.x, 1e-4, 1e-4)

    ps = np.empty(p, np.float)
    for i from 0 <= i < p:
        ps[i] = gsl_vector_get(s.x, i)

    gsl_multifit_fdfsolver_free(s)
    free(x_init)
    free(c_ys)
    free(c_xs)

    return ps
