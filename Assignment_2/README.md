# Pre-trained CNN Representation Transfer and Robustness Analysis
This repository contains the implementation and experimental pipeline for analyzing **GNR 638 - Coding Assignment 2: Pre-trained CNN Representation Transfer and Robustness Analysis**. The project evaluates how different CNN backbones behave when transferred to a new dataset under several controlled experimental conditions. The experiments are conducted using the **Aerial Image Dataset (AID)** and follow the experimental design as illustrated in the assignment statement.

---

# Models Evaluated
As per the assignment requirements, models pre-trained in ImageNet (taken from the **Timm** library) were used. The work evaluated **three** models:

- ResNet-50
- ConvNeXT-Tiny
- DenseNet-121

Each model is evaluated under the same training protocols to ensure **fair comparison**.

---

# Experimental Scenarios
The project implements five experimental studies:

- **Linear Probe Transfer** - Evaluate frozen backbone with only linear classifier trained.
- **Fine-Tuning Strategies** - Compare different backbone unfreezing strategies.
- **Few-Shot Learning** - Evaluate model performance under limited training data.
- **Corruption Robustness** - Analyze model behavior under input distribution shifts.
- **Layer-wise Feature Probing** - Study semantic abstraction across network depth.

Each experiment produces **quantitative metrics, plots, and visualizations**.

---
# Repository Structure
```
root/
│
├── resnet_50/
│ ├── experiments.py
│ ├── plots.py
│ ├── utils.py
│ ├── ResNet50_LP.py
│ ├── main.py
│ ├── resnet_50.log
│ └── run_resnet.sh
│
├── convNext/
│ ├── experiments.py
│ ├── plots.py
│ ├── utils.py
│ ├── ConvNeXtTiny_LP.py
│ ├── main.py
│ ├── convNext.log
│ └── run_convNext.sh
│
├── densenet/
│ ├── experiments.py
│ ├── plots.py
│ ├── utils.py
│ ├── DenseNet121_LP.py
│ ├── main.py
│ ├── densenet.log
│ └── run_densenet.sh
│
├── Figures/
│ ├── resnet_50/
│ ├── convNext/
│ └── densenet/
│
├── train_data/
│
├── gnr_638.yml
│
└── README.md
```

---

# Directory Description
- Each model has its own directory containing all experiment code.
    - `experiments.py` - Contains implementations of all experimental scenarios.
    - `plots.py` - Contains visualization functions used to generate plots/graphs.
    - `utils.py` - Provides shared helper functions for data_loading, feature extraction etc.
    - `<model_name>_LP.py` - Defines the model wrapper (class) for the specific CNN backbone.
    - `main.py` - Main code file that runs all the experiments.
    - `<model_name>.log` - Stores experiment logs.
    - `run_<model_name>.sh` - One shell script to run the entire pipeline. It has the required arguments and the code to call `main.py`
- Each model stores the plots and figures under the `root/Figures/<model_name>` directory. Each of these plots is stored in a `.pdf` extension (vectorized) to ensure clarity.
- The AID images that are used for training and evaluation are present under `root/train_data`. The images are divided into **30 classes** with approximately *250 images per class*.
- The conda environment used to run the experiments can be re-created using `root/gnr_638.yml` which has the details of the libraries and modules needed.
- The detailed report for the assignment and highlights the various aspects of this work can be found in `root/Report.pdf`.


---

# How to run?
Here is a step-by-step guide to run the code and re-create the findings of the experiments done. We assume that your directory looks similar to the structure mentioned previously and you are currently in the `root` directory.

**Step 1**

Create a conda environment that install all the required modules for running the experiment.

```bash
conda env create -f gnr_638.yml
conda activate gnr_638
```
---

**Step 2**

Navigate to the model directory and update the parameters in the shell script for your experiment. For example, if I want to change the batch size for ResNet-50 from 1024 (default) to 256, I would run the following commands.

```bash
cd /resnet_50
nano run_resnet.sh
```

Then, I would update `BATCH_SIZE` from 1024 to 256 and save the file. Similarly, one can update the other arguments as well.

---

**Step 3**

Finally, just run the shell script to run all the experiments.

```bash
bash run_resnet.sh
```

---

**NOTE**

In case you want to just run a specific experiment (instead of all 5), then go to `main.py` in the model directory and comment out the code for the other experiments. Each experiment is preceeded by a comment of the form `Exp 4.X` which indicates that this code block is for experiment 4.X (as per the assignment nomenclature).

---

# Results
To view the results, you can:
- Navigate to `root/Figures/<model_name>` and view the plots (that have been named appropriately)
- Navigate to `root/<model_name>` and open the `<model_name>.log` file to check the logs with time stamps.

More insight into the results has been highlighted in `Report.pdf`.