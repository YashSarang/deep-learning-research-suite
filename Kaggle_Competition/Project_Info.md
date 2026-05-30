# Deep Learning-Based Quiz (Visual MCQ Solver)

## Problem Statement
You will be provided with PNG images containing multiple-choice questions (MCQs) related to deep learning. The goal is to build an automated reasoning system to process these images and accurately predict the correct answer.

### Core Challenges
- **Image-to-Answer Pipeline:** Directly process and understand mathematical equations, text, and concepts embedded in the image.
- **Problem Types:**
  - Solving mathematical equations (derivatives, optimizations, etc.).
  - Answering theoretical or conceptual questions.
  - Computing specific metrics (e.g., neural network size, FLOPs computation).
- **Format:** Each question has 4 options (A, B, C, D) with exactly 1 correct answer.
- **Constraints:** No tables or figures will be present in the questions. MLE (Maximum Likelihood Estimation) level complexity is excluded.

## Platform & Hardware Constraints

- **Hardware:** Models will infer on a L40s (providing 48GB VRAM).
- **Submission Format:** A single Jupyter Notebook (`inference.ipynb`).

## Critical Rules
- **No Internet Access:** The inference environment strictly forbids internet access. All models, weights, and pip wheels MUST be loaded via offline Kaggle Datasets.

- **No Training Data Provided:** You must rely on pre-trained models or external knowledge bases. Training from scratch is not viable. We have manually curated data in the data folder

## Evaluation
- **Metric:** Multi-class classification accuracy.
- **Grading:** Final grades and rankings are determined by performance on a hidden test dataset.
- **Sanity Check:** Sample test cases and images are provided in the data folder to validate the inference pipeline.
- Refer to sample_test_project_2 for submission format and stuff

## Important Notes from the Instructor
- Pre-trained, open-source models are highly encouraged.
- Focus on prompting and inference strategies rather than attempting to train a model from scratch.
- The server provides sufficient GPU memory (48GB VRAM) to run robust open-source models natively.

## Q&A from the Instructor

**Q: Will we have different kinds of fonts, color grading, and background color?**  
**Ans:** No. The formatting will be consistent.

**Q: Will our code (whatever we use) be answering the questions?**  
**Ans:** Yes, it can be an LLM, VLM, or any other model or pipeline that you think is capable.

**Q: In this project, can the MCQs be on any topic and any type?**  
**Ans:** They are strictly deep learning specific. They consist of text and equations. No tables or figures will be included in the question prompt.

**Q: Can you give an example of the type/difficulty level of the questions that can be asked? Will the model be required to solve math equations? If yes, what level of complexity?**  
**Ans:** It can involve solving mathematical equations, theoretical concepts, computing the size of a network, FLOPs computation, and so on. Mathematical equations do need to be solved, but extremely rigorous math (like MLE sort of stuff) is not expected. No figures/tables will need to be passed.

**Q: What are the specifications regarding the project? Can we add any pre-trained models from Kaggle, or should we train everything from scratch?**  
**Ans:** Up to you. There are no restrictions, except that your notebook must run on Kaggle (adhering to inference runtime limits and GPU limits) and must work offline without internet.


# Submission Information 
You can submit a zip containing a single setup.bash file
The naming of the zip file should be project_num_student1roll_student2roll.zip (For Eg. If your project is project 2 and the team members in your group are having roll number 22m2162, 22m2152, your file name should be project_2_22m2162_22m2152.zip)
Your conda environment name should be: gnr_project_env
Environment python version should be 3.11
While setup.bash file is run, internet will be available therefore in the bash file you should clone your repository (make it public), download weights, create environment, activate environment or perform any operation that requires internet and setup of your environment. 

No internet will be provided when your inference python script is run.
inference.py must accept --test_dir as a command-line argument and use it to read the test data.
You should create submission.csv file in your directory itself and not in test directory while inference.py is run
Following commands will be run while grading your project: 
cd ./your_directory
bash setup.bash (Your setup file)
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
Note: The target system will be Linux containing L40s with cuda 12.6 and 48GB VRAM (16GB RAM). If at any point your program fails to run, 0 marks will directly be awarded since it is an automated system, and nothing can be done about it, therefore thoroughly test your system before submitting your bash file. 

Test dataset will follow the same structure as provided to you in the sample test dataset. It'll consist of folder containing images, test.csv file and a dummy submission.csv (For more details refer to shared sample test set)

---

## Our Submission

**Zip filename:** `project_2_24m2152_24m2160_25d1598.zip`  
**Contents:** `setup.bash`

**Group members:**
- Yash Sarang — 24M2152
- Sarvesh Shashidhar — 24M2160
- Anirban Saha — 25D1598

**Grading commands that will be run:**
```bash
bash setup.bash
conda activate gnr_project_env
cd GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition
python inference.py --test_dir <absolute_path_to_test_dir>
```

**Offline evaluation result (2000-image synthetic dataset):** 95.4% accuracy


