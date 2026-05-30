import re

with open('output.md', 'r') as f:
    lines = f.readlines()

new_lines = []
in_scratch_logs = False
in_official_logs = False

for line in lines:
    if line.startswith('Starting Assignment 3'):
        new_lines.append('# Assignment 3 Execution Pipeline: DenseNet Replication\n\n')
    elif '[1/3] Training Custom' in line:
        new_lines.append('## [1/3] Custom (From-Scratch) DenseNet Architecture\n\n')
    elif '[2/3] Training Official PyTorch' in line:
        new_lines.append('---\n\n## [2/3] Official PyTorch DenseNet Architecture\n\n')
    elif '[3/3] Generating Metrics Report' in line:
        new_lines.append('---\n\n## [3/3] Generating Metrics Report and Visualizations\n\n')
    elif line.startswith('Epoch: 01/100') and not in_scratch_logs and not in_official_logs:
        in_scratch_logs = True
        new_lines.append('<details>\n<summary><b>View Training Logs (100 Epochs)</b></summary>\n\n```text\n')
        new_lines.append(line)
    elif line.startswith('Epoch: 01/100') and not in_official_logs:
        # Wait, if we see another Epoch 01/100, it's the second execution (official)
        in_official_logs = True
        new_lines.append('<details>\n<summary><b>View Training Logs (100 Epochs)</b></summary>\n\n```text\n')
        new_lines.append(line)
    elif line.startswith('Training Complete') and in_scratch_logs:
        in_scratch_logs = False
        new_lines.append('```\n</details>\n\n')
        new_lines.append(f'**Result**: {line.strip()}\n')
    elif line.startswith('Training Complete') and in_official_logs:
        in_official_logs = False
        new_lines.append('```\n</details>\n\n')
        new_lines.append(f'**Result**: {line.strip()}\n')
    elif line.startswith('Epoch: '):
        new_lines.append(line)
    elif line.startswith('Using device:'):
        new_lines.append(f'**Device**: {line.split(":")[1].strip().upper()}\n')
    elif line.startswith('Total trainable parameters:'):
        new_lines.append(f'**Total Trainable Parameters**: {line.split(":")[1].strip()}\n\n')
    elif 'Pipeline Complete!' in line:
        new_lines.append(f'> **{line.strip()}**\n')
    else:
        new_lines.append(line)

with open('output.md', 'w') as f:
    f.writelines(new_lines)

print("Formatting complete! output.md has been styled.")
