import os
import glob
import time

runs = ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]
for run in runs:
    log = f"{run}.log"
    npz = f"{run}_state.npz"
    final = f"{run}_final_state.npz"
    
    status = "MISSING"
    if os.path.exists(final):
        status = "FINISHED"
    elif os.path.exists(npz):
        status = f"RUNNING (updated {time.ctime(os.path.getmtime(npz))})"
    elif os.path.exists(log):
        size = os.path.getsize(log)
        if size > 0:
            status = f"LOGGING ({size} bytes)"
        else:
            status = "LOGGING (Empty)"
            
    print(f"{run}: {status}")
