# =============================================================================
# run_all.ps1 - Assignment 3: DenseNet Replication Pipeline
# =============================================================================
#
# CROSS-PLATFORM USAGE:
#   Windows  : .\run_all.ps1
#   Linux    : pwsh ./run_all.ps1
#   macOS    : pwsh ./run_all.ps1
#
# REQUIREMENTS:
#   - Python 3.8+ with PyTorch, torchvision installed (see requirements.txt)
#   - CUDA-capable GPU is strongly recommended (CPU fallback is very slow)
#   - For the Lua/Torch7 implementation, use run_lua_official.ps1 separately
#
# NOTE: On Linux/macOS, if 'pwsh' is not found, install PowerShell Core:
#   https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell
# =============================================================================

Write-Host "Starting Assignment 3 Execution Pipeline: DenseNet Replication" -ForegroundColor Green
Write-Host "(Lua/Torch7 official implementation can be run separately via run_lua_official.ps1)" -ForegroundColor DarkGray

# --- Pre-flight: Verify Python is available ---
Write-Host "`nChecking Python availability..." -ForegroundColor DarkGray
python --version 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found. Make sure Python 3.8+ is installed and on your PATH." -ForegroundColor Red
    Write-Host "        On Linux/macOS, try 'python3 --version'. If found, create a 'python' alias or symlink." -ForegroundColor Yellow
    exit 1
}

# --- Pre-flight: Warn if CUDA is unavailable ---
$cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaCheck -ne "True") {
    Write-Host "[WARNING] CUDA not detected. Training will run on CPU and may take significantly longer." -ForegroundColor Yellow
    Write-Host "          On Linux, ensure nvidia-container-toolkit and matching CUDA drivers are installed." -ForegroundColor Yellow
    Write-Host "          On macOS (Apple Silicon), MPS backend may be available but is not configured here." -ForegroundColor Yellow
}

# =============================================================================
# Step 1: From-Scratch DenseNet (custom implementation)
# =============================================================================
Write-Host "`n[1/3] Training Custom (From-Scratch) DenseNet Architecture on CIFAR-10" -ForegroundColor Cyan
python train.py --model scratch --epochs 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Custom model training failed (exit code: $LASTEXITCODE)." -ForegroundColor Red
    Write-Host "        Check that all dependencies are installed: pip install -r requirements.txt" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

# =============================================================================
# Step 2: PyTorch Official DenseNet (torchvision DenseNet121, CIFAR-10 adapted)
# This is the PyTorch-endorsed reference implementation.
# It is NOT the original 2017 Lua/Torch7 implementation —
# for that, use run_lua_official.ps1 (requires Docker with GPU passthrough).
# =============================================================================
Write-Host "`n[2/3] Training PyTorch Official DenseNet (torchvision DenseNet121, CIFAR-10 adapted)" -ForegroundColor Cyan
python train.py --model pytorch_official --epochs 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] PyTorch official model training failed (exit code: $LASTEXITCODE)." -ForegroundColor Red
    Write-Host "        Ensure torchvision is installed: pip install torchvision" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

# =============================================================================
# Step 3: Report Generation
# =============================================================================
Write-Host "`n[3/3] Generating Metrics Report and Visualizations" -ForegroundColor Cyan
# Uncomment the line below to generate the report (requires generate_report.py):
# python generate_report.py

Write-Host "`nPipeline Complete! Please check 'assignment_3_metrics.png' for the report graphs." -ForegroundColor Green
