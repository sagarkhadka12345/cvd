from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Determine the target architecture based on your Mac
# If your Mac is Intel-based (x86_64)
# For example, for Intel-based architecture (macOS-x86_64):
# extra_compile_args = ["-O3", "-arch", "x86_64"]

# If your Mac is Apple Silicon-based (arm64)
# For example, for Apple Silicon M1 or M1 Pro/Max (macOS-arm64):
extra_compile_args = ["-O3", "-arch", "arm64"]

ext_modules = [
    Extension(
        "custom_cosine_similarity",
        ["custom_cosine_similarity.pyx"],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={
                          "language_level": "3"}),
)
