# Assignment 3: IResNet Pipeline Orchestration
Write-Host "Starting IResNet Assignment 3 Execution Pipeline" -ForegroundColor Green

# 1. Train Scratch Version
Write-Host "`n[1/3] Training From-Scratch iResNet-18 on CIFAR-10..." -ForegroundColor Cyan
python train.py --model scratch --epochs 100

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Scratch model training failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Train Official Version
Write-Host "`n[2/3] Training Official iResNet-18 (Adapted for CIFAR-10)..." -ForegroundColor Cyan
python train.py --model official --epochs 100

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Official model training failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Generate Report
Write-Host "`n[3/3] Generating Metrics Report and Visualizations..." -ForegroundColor Cyan
python generate_report.py

Write-Host "`nPipeline Complete! Please check 'Report.md' and 'iresnet_metrics.png'." -ForegroundColor Green
