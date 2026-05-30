#!/bin/bash
set -e

echo "Updating package list..."
sudo apt-get update

echo "Installing build tools..."
sudo apt-get install -y build-essential cmake pkg-config

echo "Installing OpenCV..."
sudo apt-get install -y libopencv-dev

echo "Installing Python tools..."
sudo apt-get install -y python3-dev python3-pip

echo "Installing Python packages..."

pip3 install pybind11 numpy opencv-python --user

echo "Environment setup complete!"
echo "You can now run compilation using ./compile.sh"