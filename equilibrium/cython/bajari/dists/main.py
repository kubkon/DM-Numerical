from bajari.dists.dists import skew_normal_pdf, skew_normal_cdf

import numpy as np
import scipy.stats as ss


class skewnormal_gen(ss.rv_continuous):
    """Skew normal distribution."""
    def _argcheck(self, a):
        return True

    def _pdf(self, x, a):
        return skew_normal_pdf(x, 0, 1, a[0])

    def _cdf(self, x, a):
        return skew_normal_cdf(x, 0, 1, a[0])

skewnormal = skewnormal_gen(name='skewnormal')
