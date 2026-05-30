import sys
import os
import subprocess
from setuptools import setup, Extension
import pybind11

# Helper to find OpenCV using pkg-config
def get_opencv_flags():
    try:
        # Get compile flags (includes)
        cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv4']).decode().strip().split()
        # Get link flags (libs)
        ldflags = subprocess.check_output(['pkg-config', '--libs', 'opencv4']).decode().strip().split()
        return cflags, ldflags
    except Exception as e:
        print("Warning: pkg-config could not find opencv4. Assuming default paths.")
        return [], ['-lopencv_core', '-lopencv_imgcodecs', '-lopencv_imgproc']

cv_cflags, cv_ldflags = get_opencv_flags()

# Compiler flags
cpp_args = ['-std=c++17', '-O3', '-fopenmp'] + cv_cflags
link_args = ['-fopenmp'] + cv_ldflags

if sys.platform == 'darwin':
    cpp_args = ['-std=c++17', '-O3', '-Xpreprocessor', '-fopenmp'] + cv_cflags
    link_args = ['-lomp'] + cv_ldflags

ext_modules = [
    Extension(
        'deepcpp',
        sources=[
            'src/bindings.cpp',
            'src/tensor.cpp',
            'src/layers.cpp',
            'src/dataset.cpp'  # Added dataset.cpp
        ],
        include_dirs=[
            pybind11.get_include(),
            'include'
        ],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=link_args,
    ),
]

setup(
    name='deepcpp',
    version='0.1',
    author='VibeCoder',
    description='A C++ CNN with Python Bindings',
    ext_modules=ext_modules,
    zip_safe=False,
)
