# Deep Learning Framework (Assignment 1 - Data 1)

This project implements a lightweight Convolutional Neural Network (CNN) framework in C++ with Python bindings. It is optimized for the `data_1` dataset (10 classes).

## Directory Structure

- `Code_Files/data_1/`
  - `main.cpp`: Standalone C++ training and evaluation entry point.
  - `model.h`: Core CNN architecture (Conv2D, ReLU, MaxPool2D, Linear).
  - `tensor.h`: 3D and 1D Tensor implementations.
  - `data_loader.h`: Multi-threaded image loader (OpenCV-based).
  - `optimizer.h`: SGD with Momentum and Weight Decay.
  - `utils.h`: Helper functions for MACs/FLOPs calculation and argmax.
  - `bindings.cpp`: Pybind11 bindings for C++ components.
  - `compile.sh`: Compilation script for executable and Python module.
  - `test_bindings.py`: Python script to verify and use the bindings.
  - `data/`: Contains the dataset (expecting 0-9 subfolders).
  - `cnn_weights.bin`: Saved model parameters.

## Requirements

- **C++ Compiler**: GCC 7+ (with C++17 support).
- **OpenCV 4**: Required for image processing and loading (`libopencv-dev`).
- **OpenMP**: For multi-threaded training and data loading.
- **Python 3**: For using the bindings.
- **pybind11**: For building the Python module (`pip install pybind11`).
- **NumPy**: For handling data in Python.

## How to Build

### 1. Standalone C++ Executable
To build the `train` executable:
```bash
g++ -O2 -std=c++17 -fopenmp -pthread main.cpp -o train $(pkg-config --cflags --libs opencv4)
```
Or simply use the provided script:
```bash
bash compile.sh
```

### 2. Python Bindings
To build the `assignment_1` Python module:
```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp $(python3 -m pybind11 --includes) bindings.cpp -o assignment_1$(python3-config --extension-suffix) $(pkg-config --cflags --libs opencv4)
```
Alternatively, you can use:
```bash
pip install .
```

## How to Evaluate and Tune

### 1. Training and Evaluation (C++)
Run the `train` executable to train the model and generate a log of accuracy and performance metrics:
```bash
./train
```
This will produce `training_log.txt` and save weights to `cnn_weights.bin`.

### 2. Hyperparameter Tuning
You can tune the training process in `main.cpp`:
- **Learning Rate (`lr`)**: Change in `SGD optimizer(1e-2f, 0.9f);` (Line 117).
- **Momentum**: Second argument to `SGD` constructor.
- **Batch Size**: Change in `DataLoader loader(train_dataset, 128, true);` (Line 116).
- **Epochs**: Change `int epochs = 2;` (Line 143).
- **Weight Decay**: Pass as the third argument to `SGD` constructor (default is `5e-4f`).

### 3. Model Parameters Tuning
The architecture is defined in `SimpleCNN` (within `model.h`):
- **Convolutional Layers**: Modify `conv1(3, 8, 3)` where args are `(in_channels, out_channels, kernel_size)`.
- **Hidden Units**: Modify `Linear(flat_size, 64)` to change the size of the first fully connected layer.
- **Lazy Initialization**: Note that `fc1` is initialized based on the flattened output of the previous layer, making it easy to change input dimensions or pool sizes without manual calculation.

### 4. Using Python for Evaluation
You can use the compiled `.so` module in Python for custom evaluation loops or prototyping:
```python
import assignment_1
model = assignment_1.SimpleCNN(10)
model.load_model("cnn_weights.bin")

# Forward pass on a custom tensor
input_tensor = assignment_1.Tensor(3, 32, 32)
output = model.forward(input_tensor)
prediction = assignment_1.argmax(output)
```

## Performance Metrics
The framework automatically logs:
- **Loss and Accuracy** per epoch.
- **Model Parameters**: Total count of weights and biases.
- **MACs and FLOPs**: Computational complexity per inference and per epoch.
- **Throughput**: Samples processed per second.
