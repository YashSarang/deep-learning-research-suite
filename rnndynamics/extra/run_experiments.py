"""Run experiments (A1-A5, B1-B2, EC1-EC8) sequentially.
Usage:
    python run_experiments.py          # run all standard (A1-B2)
    python run_experiments.py A1       # run only A1
    python run_experiments.py EC5 EC6  # run extra credit
"""
import subprocess, sys, time, os, argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Shared base flags (from the assignment spec — do NOT change these for A/B runs)
BASE_STD = [
    "--nhid", "50", "--lr", "0.01", "--bs", "20",
    "--min_length", "50", "--max_length", "200",
    "--ebs", "10000", "--cbs", "1000", "--checkFreq", "20",
    "--seed", "52", "--valid_seed", "12345",
    "--collectDiags", "--diagBins", "60", "--satThresh", "0.05",
    "--device", "cuda",
]

# Extra credit base — free to tune hyperparams
BASE_EC = [
    "--collectDiags", "--diagBins", "60", "--satThresh", "0.05",
    "--device", "cuda", "--valid_seed", "12345",
]

EXPERIMENTS = {
    # ===== STANDARD RUNS (spec-mandated params) =====
    "A1": {"base": BASE_STD, "extra": ["--task", "mem", "--model", "rnn", "--alpha", "0.0",
                     "--clipstyle", "nothing", "--maxiters", "50000",
                     "--name", "A1_mem_rnn_tanh_noclip"]},
    "A2": {"base": BASE_STD, "extra": ["--task", "mem", "--model", "rnn", "--alpha", "0.0",
                     "--clipstyle", "rescale", "--cutoff", "0.05", "--maxiters", "50000",
                     "--name", "A2_mem_rnn_tanh_clip005"]},
    "A3": {"base": BASE_STD, "extra": ["--task", "mem", "--model", "rnn", "--alpha", "0.0",
                     "--clipstyle", "rescale", "--cutoff", "0.01", "--maxiters", "50000",
                     "--name", "A3_mem_rnn_tanh_clip001"]},
    "A4": {"base": BASE_STD, "extra": ["--task", "mem", "--model", "gru", "--alpha", "0.0",
                     "--clipstyle", "nothing", "--diagGates", "--maxiters", "50000",
                     "--name", "A4_mem_gru_noclip"]},
    "A5": {"base": BASE_STD, "extra": ["--task", "mem", "--model", "gru", "--alpha", "0.0",
                     "--clipstyle", "rescale", "--cutoff", "0.05", "--diagGates", "--maxiters", "50000",
                     "--name", "A5_mem_gru_clip005"]},
    "B1": {"base": BASE_STD, "extra": ["--task", "mul", "--model", "rnn", "--alpha", "0.0",
                     "--clipstyle", "nothing", "--maxiters", "50000",
                     "--name", "B1_mul_rnn_tanh_noclip"]},
    "B2": {"base": BASE_STD, "extra": ["--task", "mul", "--model", "gru", "--alpha", "0.0",
                     "--clipstyle", "nothing", "--diagGates", "--maxiters", "50000",
                     "--name", "B2_mul_gru_noclip"]},

    # ===== EXTRA CREDIT: Temporal Order + smart_tanh + clip + Omega =====
    # For extra credit we're free to tune. Key: need to break NLL < 1.386

    # EC5: nhid=100, alpha=0.5, clip=1.0, longer training, seed=42
    "EC5": {"base": BASE_EC, "extra": [
        "--task", "torder", "--model", "rnn", "--init", "smart_tanh",
        "--nhid", "100", "--lr", "0.01", "--bs", "20",
        "--min_length", "50", "--max_length", "200",
        "--ebs", "10000", "--cbs", "1000", "--checkFreq", "20",
        "--clipstyle", "rescale", "--cutoff", "1.0",
        "--alpha", "0.5", "--seed", "42",
        "--maxiters", "100000",
        "--name", "EC5_torder_h100_clip1_a05_s42"]},

    # EC6: nhid=200, alpha=0.5, clip=1.0, seed=42
    "EC6": {"base": BASE_EC, "extra": [
        "--task", "torder", "--model", "rnn", "--init", "smart_tanh",
        "--nhid", "200", "--lr", "0.01", "--bs", "20",
        "--min_length", "50", "--max_length", "200",
        "--ebs", "10000", "--cbs", "1000", "--checkFreq", "20",
        "--clipstyle", "rescale", "--cutoff", "1.0",
        "--alpha", "0.5", "--seed", "42",
        "--maxiters", "100000",
        "--name", "EC6_torder_h200_clip1_a05_s42"]},

    # EC7: nhid=100, shorter sequences to make it easier, alpha=0.5
    "EC7": {"base": BASE_EC, "extra": [
        "--task", "torder", "--model", "rnn", "--init", "smart_tanh",
        "--nhid", "100", "--lr", "0.01", "--bs", "20",
        "--min_length", "20", "--max_length", "50",
        "--ebs", "10000", "--cbs", "1000", "--checkFreq", "20",
        "--clipstyle", "rescale", "--cutoff", "1.0",
        "--alpha", "0.5", "--seed", "42",
        "--maxiters", "100000",
        "--name", "EC7_torder_h100_short_clip1_a05"]},

    # EC8: nhid=100, medium sequences, higher lr, alpha=1.0
    "EC8": {"base": BASE_EC, "extra": [
        "--task", "torder", "--model", "rnn", "--init", "smart_tanh",
        "--nhid", "100", "--lr", "0.05", "--bs", "20",
        "--min_length", "50", "--max_length", "200",
        "--ebs", "10000", "--cbs", "1000", "--checkFreq", "20",
        "--clipstyle", "rescale", "--cutoff", "1.0",
        "--alpha", "1.0", "--seed", "42",
        "--maxiters", "100000",
        "--name", "EC8_torder_h100_lr05_clip1_a1"]},
}

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="*", default=None,
                    help="Which experiments to run (default: A1-B2)")
args = parser.parse_args()

if args.runs is None:
    runs = [k for k in EXPERIMENTS if k.startswith(("A", "B"))]
else:
    runs = args.runs

for name in runs:
    if name not in EXPERIMENTS:
        print(f"Unknown experiment {name}. Choose from: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

for name in runs:
    exp = EXPERIMENTS[name]
    base = exp.get("base", BASE_STD)
    ts = time.strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  {name} started at {ts}")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    cmd = [sys.executable, "-m", "trainingRNNs_torch.train"] + base + exp["extra"]
    logfile = f"{name}.log"
    with open(logfile, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    with open(logfile, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        last = lines[-1] if lines else "(empty)"
    print(f"  {name} finished in {elapsed/60:.1f} min (exit={r.returncode})")
    print(f"  -> {last}", flush=True)

print(f"\nAll done! Experiments run: {runs}")
