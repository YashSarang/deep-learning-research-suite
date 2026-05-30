"""Detailed check of EC1 log for Omega regularizer activity."""
import sys, os
logfile = sys.argv[1] if len(sys.argv) > 1 else "EC1.log"
lines = open(logfile).readlines()
for l in lines:
    s = l.strip()
    if s.startswith("Iter "):
        parts = s.split(",")
        iter_part = parts[0].strip()
        nll = ""
        verr = ""
        vnll = ""
        omega = ""
        steps = ""
        for p in parts:
            p2 = p.strip()
            if "valid error" in p2 and "best" not in p2:
                verr = p2
            if "valid nll" in p2:
                vnll = p2
            if "Omega" in p2:
                omega = p2
            if "steps in the past" in p2:
                steps = p2
        print(f"{iter_part} | {verr} | {vnll} | {omega} | {steps}")
