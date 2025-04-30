# <p align="center"> 🍑 APRICOT-Mamba: Acuity Prediction in Intensive Care Unit (ICU)</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.02026-b31b1b)](https://arxiv.org/abs/2311.02026)


📝 Paper is under review on [*Nature Communications*](https://www.researchsquare.com/article/rs-4790824/v1) ([📄PDF](https://www.researchsquare.com/article/rs-4790824/v1.pdf?c=1722920524000)).

---

## 📘 Overview

**APRICOT-Mamba** is a deep learning framework designed to continuously predict patient acuity in the ICU using Electronic Health Records (EHR). It extends the APRICOT family by integrating Mamba-based state space models and Transformer architectures, enabling real-time, interpretable predictions of patient stability and transitions.

This repository includes:

- Data preprocessing pipelines for retrospective and prospective ICU cohorts.
- Training and evaluation scripts for APRICOT-Mamba, APRICOT-Transformer, GRU, CatBoost, and Transformer baselines.
- Post-hoc analysis tools for calibration, feature attribution, and prospective validation.

---

## 📂 Project Structure

```
├── README.md
└── main/
    ├── analyses/             # Post-training analyses (calibration, performance, etc.)
    │   ├── calibration/
    │   ├── confusion_matrix/
    │   ├── integrated_gradients/  # Feature importance analysis
    │   └── ...
    ├── baseline_models/      # Baseline models (CatBoost, GRU, Transformer)
    │   ├── catboost/
    │   ├── gru/
    │   └── transformer/
    ├── datasets/             # Data loading and description
    │   ├── README.md
    │   ├── eicu/
    │   ├── mimic/
    │   └── uf/
    ├── models/               # Core model implementations (APRICOT-Mamba, APRICOT-T)
    │   ├── apricotm/         # APRICOT-Mamba model
    │   ├── apricott/         # APRICOT-Transformer model
    │   ├── model_comparison.py
    │   └── variables.py      # Configuration variables
    ├── prospective_cohort/   # Prospective cohort data processing
    ├── retrospective_cohort/ # Retrospective cohort data processing
    ├── sofa_baseline/        # SOFA score baseline calculation
    └── summary/              # Summary generation scripts
```

---

## ⚙️ Requirements

### Software

- **Python** ≥ 3.8
- **Package Manager**: `pip` or `conda`
- **Key Python Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `h5py`
  - `torch` (PyTorch)
  - `optuna`
  - `catboost`
  - `captum`

Install all dependencies with:

```bash 
pip install -r requirements.txt
```

*Note*: For GPU support with PyTorch, refer to the [official installation guide](https://pytorch.org/get-started/locally/).

### Hardware

- **CPU**: Multi-core processor
- **RAM**: ≥ 16GB
- **GPU**: NVIDIA GPU with CUDA support (recommended for training deep learning models)

---

## 🏥 Data Sources

This project utilizes EHR data from:

- **eICU Collaborative Research Database**: A multi-center ICU database with high granularity data for over 200,000 admissions. Access requires credentialed approval.

  - [Apply for access](https://physionet.org/content/eicu-crd/)
  - [Documentation](https://eicu-crd.mit.edu/)


- **MIMIC-IV**: A large, freely accessible critical care database comprising de-identified health-related data associated with over 60,000 ICU admissions.

  - [Apply for access](https://physionet.org/content/mimiciv/)
  - [Documentation](https://mimic.mit.edu/docs/iv/)

- **University of Florida Health (UFH)**: Internal EHR data from UF Health. *Note*: This dataset is not publicly available at this time.

Data processing scripts are located in:

- `main/datasets/`
- `main/retrospective_cohort/`
- `main/prospective_cohort/`

The primary data format for training and evaluation is HDF5 (`.h5`). The script `main/retrospective_cohort/5_build_hdf5.py` demonstrates the structure of the final `dataset.h5` file, which includes training, validation, external test, and temporal test sets with features (`X`), static data (`static`), and labels (`y_main`, `y_trans`).

Refer to `main/datasets/README.md` for detailed information on data sources and initial setup.

---

## 🚀 Getting Started

### 1. Data Preparation

Process raw EHR data to generate the required `dataset.h5` file:

```bash
python main/retrospective_cohort/5_build_hdf5.py
```

*Note*: Adjust paths and parameters as needed in the script.

### 2. Model Training

Navigate to the desired model directory and run the training script:

```bash
cd main/models/apricotm/
python 1_train.py
```

This script performs hyperparameter optimization using `optuna`, trains the model with PyTorch, and saves:

- Best hyperparameters: `best_params.pkl`
- Model weights: `apricotm_weights.pth`
- Model architecture: `apricotm_architecture.pth`

Training duration is approximately 2 hours on an NVIDIA A100 GPU.

Repeat the process for other models as needed.

### 3. Model Evaluation

Evaluate the trained model on test sets:

```bash
python 2_eval.py
```

Evaluation results are saved in the `results` subdirectory within the model's directory.

### 4. Prospective Run 

If prospective data is prepared, apply the trained model:

```bash
python 3_prospective.py
```

### 5. Post-hoc Analyses

Perform analyses on model predictions:

```bash
python main/analyses/calibration/1_calibration.py
python main/analyses/integrated_gradients/1_integrated_gradients_table.py
```

### 6. Expected Output

Results are generated under the user-defined home directory (`HOME_DIR`), time window (`time_window`), and model:

```
{HOME_DIR}/deepacu/main/{time_window}h_window/model/{model}/results
```

---

## 📊 Results & Performance

APRICOT-Mamba demonstrates high performance in predicting patient acuity, with AUROC scores comparable to state-of-the-art models. Detailed performance metrics, calibration plots, and feature importance analyses are available in the `results` directories and can be visualized using the provided analysis scripts.

---

## 🧑‍💻 Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Submit a pull request detailing your changes.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## 📚 Citation

If you use this work in your research, please cite:

```
@article{contreras2024apricotmamba_rs,
  author    = {Contreras, Miguel and Silva, Brandon and Shickel, Benjamin and Davidson, Alex and Ozrazgat-Baslanti, Tezcan and Ren, Yuan and Guan, Zihan and Balch, Justin and Zhang, Jian and Bandyopadhyay, Sudeep and Loftus, Thomas and Khezeli, Kiarash and Nerella, Sandeep and Bihorac, Azra and Rashidi, Parisa},
  title     = {APRICOT-Mamba: Acuity Prediction in Intensive Care Unit (ICU): Development and Validation of a Stability, Transitions, and Life-Sustaining Therapies Prediction Model},
  journal   = {Research Square [Preprint]},
  year      = {2024},
  month     = {Aug},
  day       = {6},
  pages     = {rs.3.rs-4790824},
  doi       = {10.21203/rs.3.rs-4790824/v1},
  pmid      = {39149454},
  pmcid     = {PMC11326394},
  note      = {Preprint}
}
```

---

## 📬 Contact

- Dr. Parisa Rashidi: [parisa.rashidi@bme.ufl.edu](mailto:parisa.rashidi@bme.ufl.edu)

---