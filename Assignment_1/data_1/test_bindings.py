import sys
import os

build_dir = "./" 
if os.path.exists(build_dir):
    sys.path.append(build_dir)

try:
    import assignment_1
    print("Successfully imported assignment_1!")
except ImportError:
    print("Could not import assignment_1. Did you build/install it?")
    sys.exit(1)

import numpy as np

t = assignment_1.Tensor1D(10)
t.data = [float(i) for i in range(10)]
print("Tensor1D data:", t.numpy())

model = assignment_1.SimpleCNN(10)

if os.path.exists("cnn_weights.bin"):
    print("Loading trained model from cnn_weights.bin...")
    model.load_model("cnn_weights.bin")
else:
    print("Warning: cnn_weights.bin not found. Using random weights!")
    print("Run ./train first to train and save the model.")

print("Model initialized.")

data_path = "data"
if os.path.exists(data_path):
    print(f"Loading data from {data_path}...")
    dataset = assignment_1.ImageFolderDataset(data_path)
    if len(dataset.samples) > 0:
        loader = assignment_1.DataLoader(dataset, 50, True)
        if loader.has_next():
            batch = loader.next_batch()
            print(f"Loaded batch of size {len(batch)}")

            total_loss = 0
            correct = 0
            for sample in batch:
                logits = model.forward(sample.image)

                loss, _ = assignment_1.compute_loss_and_grad(logits, sample.label)
                total_loss += loss

                pred = assignment_1.argmax(logits)
                if pred == sample.label:
                    correct += 1

            print(f"Batch Average Loss: {total_loss / len(batch)}")
            print(f"Batch Accuracy: {100.0 * correct / len(batch)}%")
    else:
        print("Dataset empty.")
else:
    print("Data directory not found. Using random input.")

    input_tensor = assignment_1.Tensor(3, 32, 32)
    dummy_data = [0.5] * (3 * 32 * 32)
    input_tensor.data = dummy_data

    output = model.forward(input_tensor)
    print("Forward pass output size:", output.size)
    print("Forward pass output:", output.numpy())

    opt = assignment_1.SGD(0.01, 0.9, 0.0005)

    opt.step_model(model)
    print("Optimizer step successful.")

    loss, grad = assignment_1.compute_loss_and_grad(output, 5)
    print("Loss:", loss)
    print("Gradient size:", grad.size)