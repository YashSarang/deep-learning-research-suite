"""EC9: Max-GPU extra credit run for temporal order task.
nhid=500, bs=200, maxiters=200000, smart_tanh + clip + Omega.
"""
import subprocess, sys, time, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cmd = [
    sys.executable, "-m", "trainingRNNs_torch.train",
    "--task", "torder",
    "--model", "rnn",
    "--init", "smart_tanh",
    "--nhid", "500",
    "--lr", "0.01",
    "--bs", "200",
    "--min_length", "50",
    "--max_length", "200",
    "--ebs", "10000",
    "--cbs", "1000",
    "--checkFreq", "50",
    "--seed", "42",
    "--valid_seed", "12345",
    "--clipstyle", "rescale",
    "--cutoff", "1.0",
    "--alpha", "0.5",
    "--maxiters", "200000",
    "--collectDiags",
    "--diagBins", "60",
    "--satThresh", "0.05",
    "--device", "cuda",
    "--name", "EC9_torder_h500_bs200_clip1_a05",
]

print(f"EC9 starting at {time.strftime('%H:%M:%S')}", flush=True)
print(f"Config: nhid=500, bs=200, maxiters=200k, alpha=0.5, clip=1.0", flush=True)
t0 = time.time()

with open("EC9.log", "w") as logf:
    r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

elapsed = time.time() - t0
lines = [l.strip() for l in open("EC9.log") if l.strip()]
last = lines[-1] if lines else "(empty)"
print(f"EC9 finished in {elapsed/60:.1f} min (exit={r.returncode})")
print(f"  -> {last}")
