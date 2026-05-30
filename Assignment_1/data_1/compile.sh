#!/bin/bash
set -e

echo "Compiling train executable..."
g++ -O2 -std=c++17 -fopenmp -pthread main.cpp -o train $(pkg-config --cflags --libs opencv4)
echo "Compilation successful: ./train"

echo "Compiling Python bindings..."

if ! python3 -m pybind11 --includes > /dev/null 2>&1; then
    echo "pybind11 module not found. Installing locally..."
    pip3 install pybind11 --break-system-packages --user || true
fi
PY_INCLUDE=$(python3 -m pybind11 --includes)
SUFFIX=$(python3-config --extension-suffix)
OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)

g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp ${PY_INCLUDE} bindings.cpp -o assignment_1${SUFFIX} ${OPENCV_FLAGS}

echo "Bindings compilation successful: assignment_1${SUFFIX}"