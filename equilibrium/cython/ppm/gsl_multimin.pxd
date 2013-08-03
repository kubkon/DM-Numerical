from cython_gsl cimport *

cdef extern from "gsl/gsl_multimin.h":

    ctypedef struct gsl_multimin_function:
        double (* f) (gsl_vector * x, void * params)
        size_t n
        void * params

    ctypedef struct gsl_multimin_fminimizer_type:
        const char * name
        int (* alloc) (void * state, size_t n)
        int (* set) (void * state, gsl_multimin_function * f,
                     const gsl_vector * x,
                     double * size,
                     const gsl_vector * step_size)
        int (* iterate) (void * state, gsl_multimin_function * f,
                         gsl_vector * x,
                         double * size,
                         double * fval)
        void (* free) (void * state)
