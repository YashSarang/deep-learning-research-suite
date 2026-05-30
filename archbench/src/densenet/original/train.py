import argparse
import time
import os
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataset import get_cifar10_dataloaders
from model import get_densenet_100_12_cifar

def parse_args():
    parser = argparse.ArgumentParser(description='DenseNet Training/Comparison on CIFAR-10')
    parser.add_argument('--model', type=str, default='scratch', choices=['scratch', 'official', 'pytorch_official'],
                        help='Which model to train: scratch, official (Lua/Docker), or pytorch_official (torchvision DenseNet121)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--subset', action='store_true', 
                        help='Use a small subset of the dataset for fast verification')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable cudnn benchmark for faster convolutions if using static input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.model == 'official':
        import subprocess
        import re

        print("Using Official LUA DenseNet Model via Docker...")
        print("Note: The Torch7 code outputs error rate. This script will convert it to Accuracy for the JSON logs.")
        history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
        
        densenet_lua_path = os.path.join(os.getcwd(), 'Densenet_Lua')
        docker_cmd = [
            'docker', 'run', '--rm', 
            '-v', f"{densenet_lua_path}:/data", 
            '-w', '/data', 
            'nagadomi/torch7:latest', 
            'th', 'main.lua', 
            '-dataset', 'cifar10', 
            '-depth', '40', 
            '-growthRate', '12', 
            '-nEpochs', str(args.epochs),
            '-batchSize', str(args.batch_size)
        ]
        
        print(f"Running command: {' '.join(docker_cmd)}")
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            current_epoch_train_loss = 0.0
            
            for line in process.stdout:
                print(line, end='')
                # Parse Lua script training error to get loss
                err_match = re.search(r"Err\s+([0-9.]+)", line)
                if "Epoch: [" in line and err_match:
                    current_epoch_train_loss = float(err_match.group(1))
                
                # Parse Lua script final epoch top1 error to log accuracy
                epoch_match = re.search(r"\*\s*Finished epoch\s*#\s*\d+\s+top1:\s+([0-9.]+)", line)
                if epoch_match:
                    top1_err = float(epoch_match.group(1))
                    test_acc = 100.0 - top1_err
                    history['train_loss'].append(current_epoch_train_loss)
                    history['test_acc'].append(test_acc)
                    
            process.wait()
            total_time = time.time() - start_time
            if process.returncode != 0 or len(history['test_acc']) == 0:
                raise Exception("Docker execution failed or produced no metrics.")
                
            print(f"\nLua Training Complete in {total_time/60:.2f} mins.")

            log_filename = "history_lua_official.json"
            with open(log_filename, 'w') as f:
                json.dump(history, f)
            print(f"Metrics saved to {log_filename}")
            return
            
        except Exception as e:
            print("\n" + "="*80)
            print("WARNING: Official Lua Torch7 Implementation failed to execute.")
            print(f"Reason: {str(e)}")
            print("Note: The 2017 CVPR code requires CUDA 8.0 and fails on modern systems.")
            print("Falling back to the PyTorch officially endorsed version for Assignment completion...")
            print("="*80 + "\n")
            
            trainloader, testloader = get_cifar10_dataloaders(
                batch_size=args.batch_size, 
                num_workers=args.workers, 
                use_subset=args.subset
            )
            net = models.densenet121(weights=None)
            net.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.features.pool0 = nn.Identity() 
            net.classifier = nn.Linear(net.classifier.in_features, 10)
            args.model = 'pytorch_official' # ensure we save logs correctly for pytorch
            net = net.to(device)
            # Below code shares same training loop as scratch. So we just set net and let it fall through.
            
    if args.model == 'pytorch_official':
        print("Using PyTorch Official DenseNet (torchvision DenseNet121, CIFAR-10 adapted)...")
        trainloader, testloader = get_cifar10_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_subset=args.subset
        )
        net = models.densenet121(weights=None)
        net.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.features.pool0 = nn.Identity()
        net.classifier = nn.Linear(net.classifier.in_features, 10)
        net = net.to(device)

    if args.model == 'scratch' or 'net' in locals():
        if args.model == 'scratch':
            print("Using From-Scratch DenseNet Model...")
            
            trainloader, testloader = get_cifar10_dataloaders(
                batch_size=args.batch_size, 
                num_workers=args.workers, 
                use_subset=args.subset
            )
            
            net = get_densenet_100_12_cifar(num_classes=10)
            net = net.to(device)

        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        scaler = torch.amp.GradScaler('cuda')

        history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

        start_time = time.time()
        
        for epoch in range(args.epochs):
            net.train()
            train_loss = 0.0
            correct = 0
            total = 0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            epoch_start_time = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_time = time.time() - epoch_start_time
            peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

            net.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                 for batch_idx, (inputs, targets) in enumerate(testloader):
                     inputs, targets = inputs.to(device), targets.to(device)
                     outputs = net(inputs)
                     loss = criterion(outputs, targets)
                     
                     test_loss += loss.item()
                     _, predicted = outputs.max(1)
                     test_total += targets.size(0)
                     test_correct += predicted.eq(targets).sum().item()

            epoch_train_loss = train_loss/len(trainloader)
            epoch_test_acc = 100. * test_correct / test_total
            
            print(f"Epoch: {epoch+1:02d}/{args.epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Train Acc: {100.*correct/total:.2f}% | "
                  f"Test Acc: {epoch_test_acc:.2f}% | "
                  f"Max VRAM: {peak_gpu_mem:.0f} MB")
            
            history['train_loss'].append(epoch_train_loss)
            history['test_acc'].append(epoch_test_acc)

        total_time = time.time() - start_time
        print(f"\nTraining Complete in {total_time/60:.2f} mins.")

        log_filename = f"history_{args.model}.json"
        with open(log_filename, 'w') as f:
            json.dump(history, f)
        print(f"Metrics saved to {log_filename}")

if __name__ == '__main__':
    main()
