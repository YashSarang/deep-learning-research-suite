import deepcpp
import time
import random
import os
import sys
import datetime


class Logger(object):
    def __init__(self, filename="logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.log.write(f"\n\n=== NEW SESSION STARTED: {datetime.datetime.now()} ===\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def format_metric(n):
    if n >= 1e9: return f"{n / 1e9:.2f}G"
    elif n >= 1e6: return f"{n / 1e6:.2f}M"
    elif n >= 1e3: return f"{n / 1e3:.2f}K"
    return str(n)


def evaluate_accuracy(model, dataset, indices, batch_size, num_classes, h, w, c):
    total = len(indices)
    if total == 0: return 0.0

    correct = 0

    for i in range(0, total, batch_size):
        batch_idx = indices[i : min(i + batch_size, total)]
        current_bs = len(batch_idx)

        inputs, targets = dataset.get_batch(batch_idx)

        raw_output = model.forward(inputs, current_bs, c, h, w)

        for b in range(current_bs):
            start = b * num_classes
            logits = raw_output[start : start + num_classes]

            pred_label = logits.index(max(logits))
            true_label = int(targets[b])

            if pred_label == true_label:
                correct += 1

    return (correct / total) * 100.0


def create_model(num_classes=100):
    model = deepcpp.CppModel()
    model.add_conv2d(3, 16, 3, 1, 1)
    model.add_relu()
    model.add_maxpool2d(2, 2)

    model.add_conv2d(16, 32, 3, 1, 1)
    model.add_relu()
    model.add_maxpool2d(2, 2)

    model.add_linear(32 * 8 * 8, num_classes)
    return model


def main():
    sys.stdout = Logger("logs.txt")

    DATA_PATH = "./data_2"
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 0.01
    NUM_CLASSES = 100

    # Image dims
    IMG_C, IMG_H, IMG_W = 3, 32, 32

    print("="*100)
    print("                                TRAINING CONFIGURATION")
    print("="*100)
    print(f"Dataset Path  : {os.path.abspath(DATA_PATH)}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Learning Rate : {LR}")
    print("="*100)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Path '{DATA_PATH}' does not exist.")
        return

    print("\n[System] Loading Dataset...")
    ds = deepcpp.PyDataset()
    ds.load(DATA_PATH)
    total_images = ds.size()
    print(f"[System] Found {total_images} images.")

    if total_images == 0: return

    all_indices = list(range(total_images))
    random.shuffle(all_indices)
    split_pt = int(0.9 * total_images)
    train_indices = all_indices[:split_pt]
    val_indices = all_indices[split_pt:]

    print(f"[System] Split: {len(train_indices)} Training | {len(val_indices)} Validation")

    print("[System] Initializing Model...")
    model = create_model(NUM_CLASSES)

    params, macs = model.get_complexity_info(IMG_C, IMG_H, IMG_W)
    flops = macs * 2

    p_str = format_metric(params)
    m_str = format_metric(macs)
    f_str = format_metric(flops)

    print("\n" + "="*110)
    header = f"{'Epoch':<6} | {'Avg Loss':<10} | {'Acc %':<8} | {'Time(s)':<8} | {'Params':<10} | {'Tot MACs':<10} | {'Tot FLOPs':<10} | {'Status':<10}"
    print(header)
    print("-" * 110)

    for epoch in range(EPOCHS):
        start_time = time.time()
        random.shuffle(train_indices)

        total_loss = 0.0
        batches = 0

        for i in range(0, len(train_indices), BATCH_SIZE):
            batch_idx = train_indices[i : min(i + BATCH_SIZE, len(train_indices))]
            current_bs = len(batch_idx)

            inputs, targets = ds.get_batch(batch_idx)

            loss = model.train_step(inputs, targets, current_bs, IMG_C, IMG_H, IMG_W, LR)

            total_loss += loss
            batches += 1

            sys.stdout.terminal.write(".")
            sys.stdout.terminal.flush()

        sys.stdout.terminal.write("\r")

        val_acc = evaluate_accuracy(model, ds, val_indices, BATCH_SIZE, NUM_CLASSES, IMG_H, IMG_W, IMG_C)

        avg_loss = total_loss / batches if batches > 0 else 0
        elapsed = time.time() - start_time

        row = f"{epoch+1:<6} | {avg_loss:<10.4f} | {val_acc:<8.2f} | {elapsed:<8.2f} | {p_str:<10} | {m_str:<10} | {f_str:<10} | Completed"
        print(row)

        if (epoch + 1) % 5 == 0:
            LR *= 0.5
            print(f"             > Scheduler: Reducing LR to {LR}")

    print("=" * 110)

    model.save_weights("final_weights.bin")
    print("\n[System] Weights saved to final_weights.bin")
    print("Done.")

if __name__ == "__main__":
    main()
