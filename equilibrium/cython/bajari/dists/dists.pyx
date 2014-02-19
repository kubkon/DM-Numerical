from cython_gsl cimport *
from libc.math cimport exp, sqrt, pow, erf


cdef double standard_normal_cdf(double x) nogil:
    return 0.5 * (1 + erf(x / sqrt(2)))

cdef double standard_normal_pdf(double x) nogil:
    cdef double pi = 3.14159265
    return exp(-0.5 * pow(x, 2)) / sqrt(2 * pi)

cdef double normal_cdf(double x, double loc, double scale) nogil:
    return standard_normal_cdf((x - loc) / scale)

cdef double normal_pdf(double x, double loc, double scale) nogil:
    return standard_normal_pdf((x - loc) / scale)

cdef double trunc_normal_cdf(double x,
                             double loc,
                             double scale,
                             double a,
                             double b) nogil:
    cdef double epsilon = (x - loc) / scale
    cdef double alpha = (a - loc) / scale
    cdef double beta = (b - loc) / scale
    
    return ((standard_normal_cdf(epsilon) - standard_normal_cdf(alpha))
           /(standard_normal_cdf(beta) - standard_normal_cdf(alpha)))

cdef double trunc_normal_pdf(double x,
                             double loc,
                             double scale,
                             double a,
                             double b) nogil:
    cdef double epsilon = (x - loc) / scale
    cdef double alpha = (a - loc) / scale
    cdef double beta = (b - loc) / scale

    return (standard_normal_pdf(epsilon)
           /(scale * (standard_normal_cdf(beta) - standard_normal_cdf(alpha))))

