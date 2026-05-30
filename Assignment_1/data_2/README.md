# DeepCpp: High-Performance C++ CNN with Python Bindings

DeepCpp is a custom-built Convolutional Neural Network (CNN) framework implemented from scratch in C++ (using OpenMP for parallelism and OpenCV for image processing). It exposes a Python interface using `pybind11`, allowing you to define architectures, train models, and evaluate results using Python scripts while the heavy lifting is done in C++.

## 1. Project Structure

```text
DeepCpp/
├── Makefile                # (Optional) For pure C++ compilation
├── setup.py                # Build script for Python extension
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Main script for Training
├── evaluate.py             # Script for Evaluation
├── logs.txt                # Training logs (generated automatically)
├── final_weights.bin       # Saved model weights (generated after training)
├── include/
│   ├── tensor.hpp          # Tensor math and memory management
│   ├── dataset.hpp         # Image loading and batching
│   └── layers.hpp          # Layer definitions (Conv2d, ReLU, Linear, etc.)
└── src/
    ├── tensor.cpp          # Tensor implementation
    ├── dataset.cpp         # OpenCV image processing implementation
    ├── layers.cpp          # Forward/Backward pass logic
    └── bindings.cpp        # pybind11 wrapper code
```

## 2. Prerequisites & Installation

### System Dependencies

You must have a C++ compiler and OpenCV development libraries installed.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install build-essential libopencv-dev python3-dev
```

**Fedora/RedHat:**
```bash
sudo dnf install opencv-devel python3-devel gcc-c++
```

**Conda Users:**
If using Conda, install OpenCV inside the environment to avoid library conflicts:
```bash
conda install -c conda-forge opencv pkg-config
```

### Python Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```


## 4. How to Load Data

The framework expects data in a standard Image Folder format.

**Directory Structure:**
```text
data/
├── class_0_name/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── class_1_name/
│   ├── img1.jpg
│   └── ...
└── ...
```

**In the Code:**
1.  Open `run_pipeline.py`.
2.  Locate the `main()` function.
3.  Change the `DATA_PATH` variable:

```python
# run_pipeline.py
DATA_PATH = "/absolute/path/to/your/dataset"
```

## 5. How to Configure the Model

You can modify the neural network architecture (layers) and hyperparameters (learning rate, batch size) in the Python scripts.

### Modifying Architecture

Edit the `create_model()` function in both `run_pipeline.py` and `evaluate.py`.
**Note:** The architecture must be identical in both files for weight loading to work.

```python
def create_model(num_classes=100):
    model = deepcpp.CppModel()
    
    # Example: Add a Convolutional Layer
    # Args: (Input Channels, Output Channels, Kernel Size, Stride, Padding)
    model.add_conv2d(3, 16, 3, 1, 1) 
    model.add_relu()
    model.add_maxpool2d(2, 2)
    
    # Example: Add Fully Connected Layer
    # Args: (Input Features, Output Classes)
    # Calculation: 32 (channels) * 8 (height) * 8 (width)
    model.add_linear(32 * 8 * 8, num_classes)
    
    return model
```

### Modifying Hyperparameters

Edit the constants in the `main()` function of `run_pipeline.py`:

```python
BATCH_SIZE = 64      # Number of images per step
EPOCHS = 20          # Number of full passes over dataset
LR = 0.01          # Learning Rate
NUM_CLASSES = 100    # Set to 10 for MNIST/CIFAR-10, 100 for CIFAR-100
```
## 3. Compilation

Before running any scripts, you must compile the C++ source code into a Python extension module (`deepcpp.so`).

Run the following command in the root directory:

```bash
python3 setup.py build_ext --inplace
```

*   **Success:** This will generate a file named roughly `deepcpp.cpython-312-x86_64-linux-gnu.so`.
*   **Failure:** If it fails due to OpenCV, ensure `pkg-config --cflags opencv4` works in your terminal.

## 6. Training the Model

To start training, simply run:

```bash
python3 run_pipeline.py
```

**What happens:**
1.  C++ loads images from the disk (multi-threaded).
2.  Training runs for the specified epochs.
3.  **Logs:** Real-time progress is printed to console and saved to `logs.txt`.
4.  **Weights:** After training, parameters are saved to `final_weights.bin`.

## 7. Evaluating the Model

To calculate accuracy on a test dataset using saved weights.

**Command Syntax:**
```bash
python3 evaluate.py <path_to_test_data> <path_to_weights_file>
```

**Example:**
```bash
python3 evaluate.py ./data/test ./final_weights.bin
```

**Important:**
*   The `test_data` folder must have the same class-folder structure as the training data.
*   Ensure `create_model()` in `evaluate.py` matches the trained model's architecture exactly.

## 8. Output Locations

| Output Type | File Name | Description |
| :--- | :--- | :--- |
| Model Weights | `final_weights.bin` | Binary file containing all trained parameters (weights & biases). Saved in the root folder. |
| Training Logs | `logs.txt` | Text file containing loss, accuracy (if validation enabled), and timing per epoch. Appends to existing file. |
| Compiled Lib | `deepcpp.*.so` | The compiled C++ library importable by Python. |

