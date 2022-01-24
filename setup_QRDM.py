# Run with: "python setup_DMQR.py build_ext -i"
#
# Builds a "DMQR.pyd"/"DMQR.so" module from "yyy.c" and "xxx.c", taken from "./src"

from setuptools import setup, Extension
import shutil, os, glob, sys
from distutils.core import setup
from distutils.extension import Extension
#from Cython.Build import cythonize
import numpy
import numpy.distutils.misc_util

static_libraries = ['blas','lapacke','lapack']
static_lib_dir = '/usr/lib'
libraries = ['z', 'xml2', 'gmp']

extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

extension_mod = Extension(name = "QRDM", 
		          sources = [
			      "QRDM_wrapper.c",
                              #"./src/dlapy2.c",
			      "./src/dlarfb.c",
			      "./src/dlarf.c",
                              "./src/dlarfg.c",
                              "./src/dgeqr2.c",
			      "./src/dgeqp3.c",
			      "./src/dgeqrdm_work.c",
			      "./src/dgeqrdm.c"

			  ], 
			  include_dirs = [
			      "./include",
			      numpy.get_include(),
			      numpy.distutils.misc_util.get_numpy_include_dirs()
			  ],
			  language = "c",
              libraries=static_libraries
              )

setup(name="QRDM", 
      version = '0.2',
      author = 'M. Dessole and F. Marcuzzi',
      description = "Execute QRDM algorithm in a Python notebook",
      ext_modules=[extension_mod],
      include_dirs = [
          "./include",
          numpy.get_include(),
          numpy.distutils.misc_util.get_numpy_include_dirs()
          ]
      )
