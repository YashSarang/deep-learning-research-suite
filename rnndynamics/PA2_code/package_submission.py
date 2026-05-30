import zipfile
import os

def package():
    zip_name = "submission_assignment2.zip"
    files_to_include = [
        "trainingRNNs_torch/model.py",
        "trainingRNNs_torch/train.py",
        "commands.txt",
        "../Report.md",
        "../ToDo.md"
    ]
    
    # Include all logs and npz files
    for f in os.listdir("."):
        if f.endswith(".log") or f.endswith(".npz"):
            files_to_include.append(f)
            
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file, arcname=os.path.basename(file))
            else:
                print(f"Warning: {file} not found")
                
    print(f"Submission packaged into {zip_name}")

if __name__ == "__main__":
    package()
