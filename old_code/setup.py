# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

# Define compiler flags
extra_compile_args = ["-O0","-ggdb","-ffast-math","-g1"]

extensions = [
    Extension(
        "cyth_funcs",                            # Module name
        ["cyth_funcs.pyx"],                      # Cython source files
        extra_compile_args=extra_compile_args,  # Compiler flags
        include_dirs=[np.get_include()]        # Include NumPy headers
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)