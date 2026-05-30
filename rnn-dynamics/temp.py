import torch, time

assert torch.cuda.is_available()
device = "cuda"

# warmup
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(50):
    z = x @ y
torch.cuda.synchronize()
t1 = time.time()

print("Ran matmul on:", z.device)
print("Time (s):", round(t1 - t0, 3))
print("Max allocated (MB):", round(torch.cuda.max_memory_allocated() / 1024**2, 1))
