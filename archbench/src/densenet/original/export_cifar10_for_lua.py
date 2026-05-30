import os
import numpy as np
from torchvision import datasets

def extract():
    print("Exporting CIFAR10 for Lua...")
    os.makedirs('Densenet_Lua/gen', exist_ok=True)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    with open('Densenet_Lua/gen/train_data.bin', 'wb') as f:
        # np.transpose makes it CHW format, required by torch images
        f.write(np.transpose(train_dataset.data, (0, 3, 1, 2)).tobytes()) 
    with open('Densenet_Lua/gen/train_labels.bin', 'wb') as f:
        np.array(train_dataset.targets, dtype=np.int32).tofile(f)

    with open('Densenet_Lua/gen/test_data.bin', 'wb') as f:
        f.write(np.transpose(test_dataset.data, (0, 3, 1, 2)).tobytes())
    with open('Densenet_Lua/gen/test_labels.bin', 'wb') as f:
        np.array(test_dataset.targets, dtype=np.int32).tofile(f)
    print("Export complete.")

if __name__ == '__main__':
    extract()
