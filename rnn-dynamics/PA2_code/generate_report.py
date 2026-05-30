import numpy as np
import os
import re

def extract_last_stats(log_file):
    if not os.path.exists(log_file):
        return None
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract last diagnostic summary
    matches = re.findall(r"Iter (\d+): train nll ([\d.]+), valid error ([\d.]+)%, best valid error ([\d.]+)%, avg grad norm ([\d.]+), avg grad norm \(post clip\) ([\d.]+), rho_Whh ([\d.]+)", content)
    if matches:
        return matches[-1]
    return None

def update_report(report_file, run_data):
    if not os.path.exists(report_file):
        return
    
    with open(report_file, 'r') as f:
        report = f.read()
    
    for run, stats in run_data.items():
        if stats:
            iter, train_nll, valid_err, best_err, grad_norm, post_norm, rho = stats
            replacement = (f"Experiment {run} completed at iteration {iter}. "
                           f"Final Train NLL: {train_nll}, Final Valid Error: {valid_err}%. "
                           f"Spectral Radius (rho): {rho}.")
            report = report.replace(f"[{run} Results Analysis]", replacement)
    
    with open(report_file, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    runs = ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]
    data = {}
    for run in runs:
        data[run] = extract_last_stats(f"{run}.log")
    
    update_report("../Report.md", data)
    print("Report updated with latest stats.")
