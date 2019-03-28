from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("correlate", ["correlate.pyx"],
                include_dirs=[numpy.get_include()], annotate=True)

setup(ext_modules=[ext],
      cmdclass={'build_ext': build_ext})


#
# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules = cythonize("correlate.pyx", annotate=True),
# )