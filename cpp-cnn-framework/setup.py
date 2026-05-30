"""
TinyLearn — Lightweight C++ Deep Learning Framework
Python bindings via Pybind11
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build TinyLearn")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_TESTS=OFF',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', '-j4']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'tinylearn'] + build_args, cwd=self.build_temp)


setup(
    name='tinylearn',
    version='1.0.0',
    author='Yash Sarang, Sarvesh Shashidhar, Anirban Saha',
    author_email='yash.sarang@example.com',
    description='Lightweight C++ deep learning framework with Python bindings',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/deep-learning-research-suite',
    ext_modules=[CMakeExtension('tinylearn')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='deep-learning cnn pytorch-alternative education',
    project_urls={
        'Documentation': 'https://github.com/yourusername/deep-learning-research-suite/wiki',
        'Source': 'https://github.com/yourusername/deep-learning-research-suite',
        'Bug Reports': 'https://github.com/yourusername/deep-learning-research-suite/issues',
    },
)
