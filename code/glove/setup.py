from distutils.core import setup    
from distutils.extension import Extension                                     
from Cython.Build import cythonize

ext_modules = [
    Extension("corpus_cython",
              sources=["corpus_cython.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(
    name='corpus_cython',
    ext_modules=cythonize(ext_modules),
)
