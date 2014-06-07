from cython_gsl cimport *

from libc.stdlib cimport calloc, free


cdef class CubicSpline:
    cdef gsl_interp_accel * acc
    cdef gsl_spline * spline
    cdef double * c_xs
    cdef double * c_ys
    cdef double min_x, max_x

    def __init__(self, xs, ys):
        cdef int n = xs.size
        cdef int i

        self.c_xs = <double *> calloc(n, sizeof(double))
        self.c_ys = <double *> calloc(n, sizeof(double))

        for i from 0 <= i < n:
            self.c_xs[i] = xs[i]
            self.c_ys[i] = ys[i]

        self.min_x = self.c_xs[0]
        self.max_x = self.c_xs[n-1]

        self.acc = gsl_interp_accel_alloc()
        self.spline = gsl_spline_alloc(gsl_interp_cspline, n)

        gsl_spline_init(self.spline, self.c_xs, self.c_ys, n)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        gsl_spline_free(self.spline)
        gsl_interp_accel_free(self.acc)
        free(self.c_ys)
        free(self.c_xs)

    def evaluate(self, x):
        if x < self.min_x:
            x = self.min_x

        if x > self.max_x:
            x = self.max_x

        return gsl_spline_eval(self.spline, x, self.acc)

