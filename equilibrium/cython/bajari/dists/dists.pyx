from cython_gsl cimport *
from libc.math cimport exp, sqrt, pow, erf

import numpy as np


# C functions
cdef double c_normal_cdf(double x, double loc, double scale) nogil:
    return 0.5 * (1 + erf((x - loc) / (scale * sqrt(2))))

cdef double c_normal_pdf(double x, double loc, double scale) nogil:
    cdef double pi = 3.14159265
    return exp(- pow(x - loc, 2) / (2 * pow(scale, 2))) / (sqrt(2 * pi) * scale)

cdef double c_standard_normal_cdf(double x) nogil:
    return c_normal_cdf(x, 0, 1)

cdef double c_standard_normal_pdf(double x) nogil:
    return c_normal_pdf(x, 0, 1)

cdef double c_skew_normal_pdf(double x, double loc, double scale, double shape) nogil:
    cdef double x_value = (x - loc) / scale
    return 2 / scale * c_standard_normal_pdf(x_value) * c_standard_normal_cdf(shape * x_value)

cdef double c_gsl_skew_normal_pdf(double x, void * params) nogil:
    cdef double * t_params = <double *> params
    return c_skew_normal_pdf(x, t_params[0], t_params[1], t_params[2])

cdef double c_skew_normal_cdf(double x, double loc, double scale, double shape) nogil:
    # Initialize
    cdef gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000)
    cdef double result, error

    # Define integrand
    cdef double * params = [loc, scale, shape]
    cdef gsl_function f
    f.function = &c_gsl_skew_normal_pdf
    f.params = params

    # Integrate
    gsl_integration_qagil(&f, x, 1e-8, 1e-8, 1000, w, &result, &error)

    # Clean up
    gsl_integration_workspace_free(w)

    return result


# Python bindings
def normal_pdf(xs, loc, scale):
    try:
        length = xs.shape[0]
        ys = np.empty(length, np.float)

        for i in np.arange(length):
            ys[i] = c_normal_pdf(xs[i], loc, scale)

        return ys

    except IndexError:
        return c_normal_pdf(xs, loc, scale)

def normal_cdf(xs, loc, scale):
    try:
        length = xs.shape[0]
        ys = np.empty(length, np.float)

        for i in np.arange(length):
            ys[i] = c_normal_cdf(xs[i], loc, scale)

        return ys

    except IndexError:
        return c_normal_cdf(xs, loc, scale)

def skew_normal_pdf(xs, loc, scale, shape):
    try:
        length = xs.shape[0]
        ys = np.empty(length, np.float)

        for i in np.arange(length):
            ys[i] = c_skew_normal_pdf(xs[i], loc, scale, shape)

        return ys

    except IndexError:
        return c_skew_normal_pdf(xs, loc, scale, shape)

def skew_normal_cdf(xs, loc, scale, shape):
    try:
        length = xs.shape[0]
        ys = np.empty(length, np.float)

        for i in np.arange(length):
            ys[i] = c_skew_normal_cdf(xs[i], loc, scale, shape)

        return ys

    except IndexError:
        return c_skew_normal_cdf(xs, loc, scale, shape)
