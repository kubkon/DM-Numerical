from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy as np

ext_modules = [Extension("bajarifsm", ["fsm.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()])]

setup(
  name = 'Bajari FSM solver',
  include_dirs = [np.get_include(), cython_gsl.get_include()],
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
