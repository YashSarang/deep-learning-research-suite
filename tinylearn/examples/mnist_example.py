"""
TinyLearn MNIST Example
=======================

Demonstrates how to use the TinyLearn Python bindings for MNIST digit classification.

Usage:
    python mnist_example.py --data-dir ./data/mnist --weights ./mnist_weights.bin
"""

import argparse
import numpy as np
import cv2
import tinylearn


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for TinyLearn inference.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed tensor in CHW format (3, 32, 32)
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Resize to 32x32
    img = cv2.resize(img, (32, 32))
    
    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert HWC → CHW
    img = np.transpose(img, (2, 0, 1))
    
    return img


def main():
    parser = argparse.ArgumentParser(description='TinyLearn MNIST Inference')
    parser.add_argument('--data-dir', type=str, default='./data/mnist',
                        help='Path to MNIST dataset')
    parser.add_argument('--weights', type=str, default='./mnist_weights.bin',
                        help='Path to trained model weights')
    parser.add_argument('--image', type=str, default=None,
                        help='Optional: single image to classify')
    args = parser.parse_args()
    
    # Create model
    print("Loading TinyLearn model...")
    model = tinylearn.SimpleCNN(num_classes=10)
    
    # Load weights
    try:
        model.load_weights(args.weights)
        print(f"✓ Loaded weights from {args.weights}")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        print("Run C++ training first: ./build/tinylearn_mnist --data-dir ./data/mnist")
        return
    
    # Single image inference
    if args.image:
        img_tensor = preprocess_image(args.image)
        output = model.forward(img_tensor)
        prediction = tinylearn.argmax(output)
        
        print(f"\n📸 Image: {args.image}")
        print(f"🔢 Predicted digit: {prediction}")
        print(f"📊 Confidence scores: {output.data[:10]}")  # Top 10 class scores
        return
    
    # Batch inference on dataset
    print(f"\n📁 Loading dataset from {args.data_dir}...")
    # Note: Batch inference requires implementing DataLoader in Python
    # For now, demonstrate single-image inference loop
    
    import os
    import glob
    
    # Find all images in first class folder (digit 0)
    test_images = glob.glob(os.path.join(args.data_dir, '0', '*.png'))[:10]
    
    if not test_images:
        print(f"✗ No images found in {args.data_dir}/0/")
        return
    
    print(f"✓ Found {len(test_images)} test images")
    print("\n🧪 Running inference...")
    
    correct = 0
    total = len(test_images)
    
    for img_path in test_images:
        # Ground truth is 0 (folder name)
        true_label = 0
        
        # Preprocess and predict
        img_tensor = preprocess_image(img_path)
        output = model.forward(img_tensor)
        prediction = tinylearn.argmax(output)
        
        # Check correctness
        is_correct = (prediction == true_label)
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Image: {os.path.basename(img_path)} | Predicted: {prediction} | True: {true_label}")
    
    accuracy = 100.0 * correct / total
    print(f"\n📊 Accuracy on {total} test images: {accuracy:.2f}%")
    
    # Compute model complexity
    macs = tinylearn.compute_macs(model, input_height=32, input_width=32)
    print(f"🔢 Model complexity: {macs / 1e6:.2f}M MACs per image")


if __name__ == '__main__':
    main()
