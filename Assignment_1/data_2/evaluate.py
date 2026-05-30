import deepcpp
import sys
import os


NUM_CLASSES = 100
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3

def create_model():
    model = deepcpp.CppModel()

    model.add_conv2d(3, 16, 3, 1, 1)
    model.add_relu()
    model.add_maxpool2d(2, 2)

    model.add_conv2d(16, 32, 3, 1, 1)
    model.add_relu()
    model.add_maxpool2d(2, 2)

    model.add_linear(32 * 8 * 8, NUM_CLASSES)

    return model

def calculate_accuracy(model, dataset, batch_size=32):
    total_samples = dataset.size()
    if total_samples == 0:
        return 0.0

    correct_predictions = 0

    print(f"Evaluating on {total_samples} images...")

    indices = list(range(total_samples))

    for i in range(0, total_samples, batch_size):
        batch_idx = indices[i : min(i + batch_size, total_samples)]
        current_bs = len(batch_idx)

        inputs, targets = dataset.get_batch(batch_idx)

        raw_output = model.forward(inputs, current_bs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        for b in range(current_bs):
            start_idx = b * NUM_CLASSES
            end_idx = start_idx + NUM_CLASSES
            logits = raw_output[start_idx : end_idx]

            pred_label = logits.index(max(logits))
            true_label = int(targets[b])

            if pred_label == true_label:
                correct_predictions += 1

        sys.stdout.write(f"\rProcessed: {min(i + batch_size, total_samples)}/{total_samples}")
        sys.stdout.flush()

    print("\n")
    return (correct_predictions / total_samples) * 100.0

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate.py <path_to_test_data> <path_to_weights.bin>")
        sys.exit(1)

    data_path = sys.argv[1]
    weights_path = sys.argv[2]

    if not os.path.exists(data_path):
        print(f"Error: Dataset path '{data_path}' not found.")
        sys.exit(1)

    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        sys.exit(1)

    print("Loading Dataset Structure...")
    ds = deepcpp.PyDataset()
    ds.load(data_path)

    if ds.size() == 0:
        print("Error: No images found in dataset folder.")
        sys.exit(1)

    print(f"Initializing Model and loading {weights_path}...")
    try:
        model = create_model()
        model.load_weights(weights_path)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        print("-" * 50)
        print("Possible causes:")
        print("1. 'Arch mismatch': The layers in evaluate.py do not match run_pipeline.py.")
        print("2. The weights file is from an old training run with a different architecture.")
        print("-" * 50)
        sys.exit(1)

    accuracy = calculate_accuracy(model, ds)

    print("========================================")
    print(f"RESULTS")
    print("========================================")
    print(f"Weights  : {weights_path}")
    print(f"Dataset  : {data_path}")
    print(f"Accuracy : {accuracy:.2f}%")
    print("========================================")

if __name__ == "__main__":
    main()
