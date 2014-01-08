# distribution functions (pdfs and cdfs)
cdef double c_normal_cdf(double x, double loc, double scale) nogil
cdef double c_normal_pdf(double x, double loc, double scale) nogil

cdef double c_standard_normal_cdf(double x) nogil
cdef double c_standard_normal_pdf(double x) nogil

cdef double c_skew_normal_pdf(double x, double loc, double scale, double shape) nogil
cdef double c_gsl_skew_normal_pdf(double x, void * params) nogil
cdef double c_skew_normal_cdf(double x, double loc, double scale, double shape) nogil
