# GNR638 — Submission Testing Plan & Next Steps

## Finetuning Status — Verdict

> [!IMPORTANT]
> **The 7B finetuned stage (Stage 2.5) has been DISABLED.**
> 
> **Why:** The LoRA training ran and saved only config/tokenizer files, but **no `.safetensors` weight file** was saved to `Current_implementation/qwen_mcq_finetuned/final/`. The `models/finetuned_7b/` directory is also empty. There are literally no trained weights to load.
>
> **Impact:** None. The 72B alone achieves **95.4% accuracy** on 2000 images. Stage 2.5 was only ever triggered when the 72B was uncertain (<0.75 confidence), which is already a rare case. The pipeline now goes directly from Stage 2 → Stage 3 (CPU fallbacks) for uncertain cases.
>
> `config.yaml` already updated: `finetuned_7b.enabled: false`

---

## What to Do Next

### Priority 1 — Test `inference.py` end-to-end on the cluster

This is the most critical thing. The grading system will run it blindly.

```bash
# On the cluster (GPU node, with models already downloaded)
conda activate gnr_project_env
cd ~/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition

# Test on the 2-image official sample
python inference.py --test_dir data/data/sample_test_project_2/

# Verify:
cat submission.csv          # must show image_name,option
ls -la submission.csv       # must be in CWD, not in test_dir
```

**What to check:**
- [ ] `submission.csv` appears in the current directory (not inside `data/data/sample_test_project_2/`)
- [ ] Two rows: `image_1,X` and `image_2,Y` where X,Y ∈ {1,2,3,4,5}
- [ ] Script runs to completion without crash

---

### Priority 2 — Test setup.bash in isolation (simulate grader)

The grader will run `bash setup.bash` from a fresh directory. Simulate this:

```bash
# From a CLEAN directory (not inside the repo)
cd /tmp/gnr_test_setup
bash /path/to/setup.bash

# After it finishes:
conda activate gnr_project_env
cd GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition
python inference.py --test_dir data/data/sample_test_project_2/
```

**What to check:**
- [ ] `gnr_project_env` conda env is created with Python 3.11
- [ ] All pip installs succeed
- [ ] `models/vlm/` gets populated (72B weights)
- [ ] `models/faiss/index.faiss` is built
- [ ] `inference.py` runs after `setup.bash`

---

### Priority 3 — Verify submission.csv column format

The grader uses a grading script that reads `submission.csv`. The format must be exactly:

```csv
image_name,option
image_1,3
image_2,1
```

> [!CAUTION]
> The `option` column must be an **integer** (1/2/3/4/5), not a string or float.
> Our `inference.py` already casts with `int(answer)` ✅

---

### Priority 4 — Update the submission zip

After confirming `inference.py` runs clean, regenerate the zip:

```python
import zipfile
with zipfile.ZipFile('project_2_24m2152_24m2160_25d1598.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('setup.bash')
```

The zip is at: `/users/student/pg/pg24/yash.sarang/GNR/project_2_24m2152_24m2160_25d1598.zip`

---

## Full Testing Checklist

### A. Structure Tests (no GPU needed)

```bash
cd ~/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition

# 1. Verify test data structure is parsed correctly
python3 -c "
import csv, os
d = 'data/data/sample_test_project_2'
print('test.csv cols:', open(d+'/test.csv').readline().strip())
print('images:', os.listdir(d+'/images'))
"

# 2. Verify inference.py has correct argparse
python3 inference.py --help

# 3. Check config.yaml disabled finetuned stage
python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print('ft enabled:', c['finetuned_7b']['enabled'])"
# Expected: ft enabled: False

# 4. Verify submission.csv goes to CWD (not test_dir)
# (Dry run — patch solve() to return '1')
python3 -c "
import sys, os, csv
sys.path.insert(0, '.')
# Patch pipeline
import unittest.mock as m
with m.patch('src.pipeline.initialize_models'), \
     m.patch('src.pipeline.solve', return_value='3'):
    import importlib
    import inference as inf_mod
    inf_mod.run_inference(
        test_dir='data/data/sample_test_project_2',
        config_path='config.yaml',
        output_path='/tmp/test_submission.csv'
    )
print('Output:')
print(open('/tmp/test_submission.csv').read())
assert os.path.exists('/tmp/test_submission.csv'), 'FAIL: submission not created'
print('PASS: submission.csv created correctly')
"
```

### B. GPU Tests (on cluster)

```bash
# 5. Test with real model — 2 images only (~30s)
python inference.py --test_dir data/data/sample_test_project_2/
cat submission.csv

# 6. Verify GPU VRAM fits (should show ~38-42 GB allocated)
nvidia-smi

# 7. Check pipeline stages (look at stdout)
# Should see: [Pipeline] Stage 1 answer: X (conf=Y.YY)
```

### C. Stress Test (optional, on cluster)

```bash
# 8. Quick 10-image test from synthetic data
python scripts/evaluate_offline.py --n 10 --verbose
```

---

## Known Risks to Watch For

| Risk | Mitigation |
|---|---|
| `src/` not in sys.path when run from different CWD | `inference.py` inserts `script_dir` into `sys.path` on startup |
| Model path not found | `_resolve_path()` in `pipeline.py` tries local + `/kaggle/input/` |
| Missing image in test_dir | Defaults to skip (`5`) instead of crash |
| Config not found | `inference.py` searches both CWD and script directory |
| `submission.csv` written to test_dir | Fixed: always written to `args.output` (default: CWD) |
| VRAM OOM | 72B NF4 ≈ 38-42 GB, fits within 48 GB L40s ✅ |
| Flash Attention not installed | Using SDPA fallback — configured in `config.yaml` |

---

## Final Submission Checklist

- [ ] `inference.py` tested end-to-end on sample data
- [ ] `submission.csv` format verified (`image_name,option` with int values)
- [ ] `setup.bash` tested from fresh directory
- [ ] `gnr_project_env` created with Python 3.11 ✅ (in setup.bash)
- [ ] `config.yaml` → `finetuned_7b.enabled: false` ✅
- [ ] Notebook removed ✅
- [ ] All scripts use `gnr_project_env` ✅
- [ ] Zip file: `project_2_24m2152_24m2160_25d1598.zip` ✅
- [ ] GitHub repo is public ✅
- [ ] Repo URL in `setup.bash` is correct ✅
