from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy as np

ext_modules = [Extension("bajari.dists.dists", ["bajari/dists/dists.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("bajari.fsm.fsm_internal", ["bajari/fsm/fsm_internal.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("bajari.ppm.ppm_internal", ["bajari/ppm/ppm_internal.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("dm.fsm.fsm_internal", ["dm/fsm/fsm_internal.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("dm.ppm.ppm_internal", ["dm/ppm/ppm_internal.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("dm.efsm.efsm_internal", ["dm/efsm/efsm_internal.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("util.polyfit", ["util/polyfit.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()]),
               Extension("util.interpolate", ["util/interpolate.pyx"],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         cython_include_dirs=[cython_gsl.get_cython_include_dir()])]

setup(
  name = 'DM Numerical Analyser',
  include_dirs = [np.get_include(), cython_gsl.get_include()],
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
