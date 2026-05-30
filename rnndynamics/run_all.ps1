# run_all.ps1
# Exact commands from the CS728 PA2 spec (Section 3)
# Added --device cuda for GPU acceleration

# Set-Location "c:\Code\CS728-CourseWork\Assignment-2\PA2_code"

# ============================================================
# Task 1: Memorization (A1–A5)
# ============================================================

Write-Host "Starting Experiment A1: RNN (tanh), no clipping..."
python -m trainingRNNs_torch.train --task mem --model rnn --alpha 0.0 --clipstyle nothing --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name A1_mem_rnn_tanh_noclip > A1.log 2>&1

Write-Host "Starting Experiment A2: RNN (tanh), clipping cutoff=0.05..."
python -m trainingRNNs_torch.train --task mem --model rnn --alpha 0.0 --clipstyle rescale --cutoff 0.05 --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name A2_mem_rnn_tanh_clip005 > A2.log 2>&1

Write-Host "Starting Experiment A3: RNN (tanh), clipping cutoff=0.01..."
python -m trainingRNNs_torch.train --task mem --model rnn --alpha 0.0 --clipstyle rescale --cutoff 0.01 --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name A3_mem_rnn_tanh_clip001 > A3.log 2>&1

Write-Host "Starting Experiment A4: GRU, no clipping..."
python -m trainingRNNs_torch.train --task mem --model gru --alpha 0.0 --clipstyle nothing --diagGates --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name A4_mem_gru_noclip > A4.log 2>&1

Write-Host "Starting Experiment A5: GRU, clipping cutoff=0.05..."
python -m trainingRNNs_torch.train --task mem --model gru --alpha 0.0 --clipstyle rescale --cutoff 0.05 --diagGates --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name A5_mem_gru_clip005 > A5.log 2>&1

# ============================================================
# Task 2: Multiplication (B1–B2)
# ============================================================

Write-Host "Starting Experiment B1: RNN (tanh), no clipping..."
python -m trainingRNNs_torch.train --task mul --model rnn --alpha 0.0 --clipstyle nothing --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name B1_mul_rnn_tanh_noclip > B1.log 2>&1

Write-Host "Starting Experiment B2: GRU, no clipping..."
python -m trainingRNNs_torch.train --task mul --model gru --alpha 0.0 --clipstyle nothing --diagGates --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 --device cuda --name B2_mul_gru_noclip > B2.log 2>&1

Write-Host "All experiments completed!"
