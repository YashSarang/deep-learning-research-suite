# =============================================================================
# run_lua_official.ps1 - Assignment 3: Official Lua/Torch7 DenseNet (CVPR 2017)
# =============================================================================
#
# CROSS-PLATFORM USAGE:
#   Windows  : .\run_lua_official.ps1
#   Linux    : pwsh ./run_lua_official.ps1
#   macOS    : pwsh ./run_lua_official.ps1
#
# REQUIREMENTS:
#   - Docker must be installed and running with the Linux engine enabled
#   - An NVIDIA GPU is required for --gpus all flag (GPU passthrough)
#
# DOCKER STARTUP INSTRUCTIONS (by OS):
#   Windows  : Open Docker Desktop from the Start Menu / system tray.
#              Ensure "Use the WSL2 based engine" or "Linux containers" is selected.
#   Linux    : Run: sudo systemctl start docker
#              Verify: docker info
#              GPU support requires nvidia-container-toolkit:
#                https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   macOS    : Open Docker Desktop from Applications.
#              NOTE: macOS does NOT support --gpus all (no NVIDIA GPU support in Docker on macOS).
#              The Lua/Torch7 container will likely fail or run on CPU-only.
#              Consider running this on a Linux machine or cloud instance instead.
#
# NOTE: This script runs the original CVPR 2017 Lua/Torch7 DenseNet code inside
#       the 'nagadomi/torch7' Docker container (CUDA 8.0 environment).
#       It may fail on very modern GPU drivers due to CUDA version incompatibility.
# =============================================================================

# --- Pre-flight: Check if Docker daemon is reachable ---
Write-Host "Checking Docker availability..." -ForegroundColor DarkGray
docker info 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
#### Docker Prerequisites (by OS)
    Write-Host "`n[ERROR] Docker is not running or not accessible. Please refer to the DOCKER STARTUP INSTRUCTIONS (by OS) section in the script for more information." -ForegroundColor Red

    # Provide OS-specific startup instructions
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        Write-Host "        Start Docker Desktop from the Start Menu or system tray." -ForegroundColor Yellow
        Write-Host "        Ensure the Linux engine is active (Settings > General > Use WSL2 based engine)." -ForegroundColor Yellow
    } elseif ($IsLinux) {
        Write-Host "        Run: sudo systemctl start docker" -ForegroundColor Yellow
        Write-Host "        Then verify: docker info" -ForegroundColor Yellow
        Write-Host "        For GPU support, install nvidia-container-toolkit:" -ForegroundColor Yellow
        Write-Host "          https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Yellow
    } elseif ($IsMacOS) {
        Write-Host "        Open Docker Desktop from Applications and wait for it to fully start." -ForegroundColor Yellow
        Write-Host "        WARNING: macOS does not support NVIDIA GPU passthrough (--gpus all)." -ForegroundColor Yellow
        Write-Host "        The Torch7 container requires CUDA 8.0; this will likely fail on macOS." -ForegroundColor Yellow
        Write-Host "        Consider running on a Linux machine or cloud GPU instance instead." -ForegroundColor Yellow
    }

    exit 1
}

# --- Warn macOS users about GPU passthrough incompatibility ---
if ($IsMacOS) {
    Write-Host "`n[WARNING] macOS detected. Docker on macOS does not support --gpus all (NVIDIA GPU passthrough)." -ForegroundColor Yellow
    Write-Host "          The Lua/Torch7 container is built for CUDA 8.0 and will likely fail without a GPU." -ForegroundColor Yellow
    Write-Host "          Proceeding anyway (may run on CPU or fail)..." -ForegroundColor Yellow
}

# --- Resolve the Lua source directory (cross-platform safe via Join-Path) ---
$densenet_path = Join-Path (Get-Location).Path "Densenet_Lua"

if (-not (Test-Path $densenet_path)) {
    Write-Host "`n[ERROR] Lua source directory not found: $densenet_path" -ForegroundColor Red
    Write-Host "        Ensure the 'Densenet_Lua' folder exists in the same directory as this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "`nRunning Official Lua DenseNet via Torch7 Docker container..." -ForegroundColor Cyan
Write-Host "Source directory: $densenet_path" -ForegroundColor DarkGray

# --gpus all: Passes the host NVIDIA GPU into the legacy CUDA 8 container environment.
#             On Linux, this requires nvidia-container-toolkit to be installed.
#             On macOS/Windows without WSL2 GPU support, remove '--gpus all' to attempt CPU-only.
docker run --rm --gpus all `
    -v "$($densenet_path):/data" `
    -w /data `
    -e TERM=xterm `
    nagadomi/torch7:latest `
    th main.lua -netType densenet -dataset cifar10 -depth 40 -growthRate 12 -nEpochs 1 -batchSize 64

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Lua training failed (exit code: $LASTEXITCODE)." -ForegroundColor Red
    Write-Host "        Possible causes:" -ForegroundColor Yellow
    Write-Host "          - Docker GPU passthrough is not configured (check --gpus all support)" -ForegroundColor Yellow
    Write-Host "          - CUDA 8.0 incompatibility with modern NVIDIA drivers (>= 520)" -ForegroundColor Yellow
    Write-Host "          - Missing or malformed Lua source files in Densenet_Lua/" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host "`nLua training execution complete!" -ForegroundColor Green
