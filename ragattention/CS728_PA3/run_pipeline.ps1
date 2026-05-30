<#
.SYNOPSIS
Master pipeline script to execute all parts of CS728_PA3.
.DESCRIPTION
This script sequentially runs:
1. Part 1 (Classical Retrieval)
2. Part 2 & Part 3 (Attention-based Retrieval & Head Selection) using the highly optimized, combined execution loop.

It logs the standard output and errors into appropriate log files so that you can view the metrics and execution paths at your convenience.
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting CS728_PA3 Master Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Run Part 1
Write-Host "`n[1/2] Executing Part 1: Classical Retrieval..." -ForegroundColor Yellow
Write-Host "Logs will be saved to run1_output.log"
python run1.py 2>&1 | Tee-Object -FilePath "run1_output.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Part 1 failed! Check run1_output.log for details." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Run Parts 2 & 3 (Combined)
Write-Host "`n[2/2] Executing Parts 2 & 3: Attention-based Retrieval and Retrieval Heads..." -ForegroundColor Yellow
Write-Host "This step will process 5000 queries in a highly memory-optimized loop."
Write-Host "Logs will be saved to run_all_output.log"
python run_all.py 2>&1 | Tee-Object -FilePath "run_all_output.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Parts 2 & 3 execution failed! Check run_all_output.log for details." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Generate Final Report
Write-Host "`n[3/3] Generating Final MD Report from Aggregated Metrics..." -ForegroundColor Yellow
python generate_report.py

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Pipeline Execution Complete!" -ForegroundColor Green
Write-Host "Check the 'final_report.md' for your submission." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
