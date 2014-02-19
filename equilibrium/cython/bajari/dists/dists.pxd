# distribution functions (pdfs and cdfs)
cdef double standard_normal_cdf(double x) nogil
cdef double standard_normal_pdf(double x) nogil

cdef double normal_cdf(double x, double loc, double scale) nogil
cdef double normal_pdf(double x, double loc, double scale) nogil

cdef double trunc_normal_cdf(double x,
                             double loc,
                             double scale,
                             double a,
                             double b) nogil

cdef double trunc_normal_pdf(double x,
                             double loc,
                             double scale,
                             double a,
                             double b) nogil

