from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("weighted_surprise_prediction_algorithms.pyx"),
    include_dirs=[numpy.get_include()]
    )
