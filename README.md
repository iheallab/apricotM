# APRICOT-Mamba: Acuity Prediction in Intensive Care Unit (ICU)

## Introduction
The repository contains the code implementation used for the paper APRICOT-Mamba: Acuity Prediction in Intensive Care Unit (ICU). We used EHR data to make continuous predictions of patient acuity in the ICU. The repository contains the codes for all data processing, model training and validation, and post-analyses.

## Requirements

### Software
*   Python (version 3.8 or higher recommended)
*   Package manager: `pip` or `conda`
*   Key Python libraries:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `h5py`
    *   `torch` (PyTorch)
    *   `optuna`
    *   `catboost`
    *   `captum`
    *   All required Python libraries are listed in the [requirements.txt](requirements.txt)    file. You can install them using:
        ```bash
        pip install -r requirements.txt
        ```
        (Note: Depending on your system and whether you need GPU support for PyTorch, you might need specific installation commands, especially for `torch`. Refer to the official PyTorch installation guide: https://pytorch.org/get-started/locally/)


### Hardware
*   **CPU:** Standard multi-core processor.
*   **RAM:** Minimum 16GB recommended, more may be needed depending on dataset size.
*   **GPU:** An NVIDIA GPU with CUDA support is highly recommended for training the deep learning models (APRICOT-Mamba, APRICOT-T, GRU, Transformer) efficiently. The code checks for CUDA availability (e.g., in [`main/models/apricotm/1_train.py`](main/models/apricotm/1_train.py)).



## Project Structure

```
├── README.md
└── main/
    ├── analyses/             # Scripts for post-training analyses (calibration, 
    |   |                     # performance, etc.)
    │   ├── calibration/
    │   ├── confusion_matrix/
    │   ├── integrated_gradients/ # Feature importance analysis
    │   └── ...
    ├── baseline_models/      # Implementations of baseline models (CatBoost, GRU, 
    |   |                     # Transformer)
    │   ├── catboost/
    │   ├── gru/
    │   └── transformer/
    ├── datasets/             # Data loading and description (see datasets/README.md)
    │   ├── README.md
    │   ├── eicu/
    │   ├── mimic/
    │   └── uf/
    ├── models/               # Core model implementations (APRICOT-Mamba, APRICOT-T)
    │   ├── apricotm/         # APRICOT-Mamba model training, evaluation, prospective run
    │   ├── apricott/         # APRICOT-Transformer model training, evaluation,     
    |   |                     # prospective run
    │   ├── model_comparison.py
    │   └── variables.py      # Configuration variables (paths, etc.)
    ├── prospective_cohort/   # Scripts for processing prospective cohort data
    ├── retrospective_cohort/ # Scripts for processing retrospective cohort data (e.g., 
    |                         # building HDF5 dataset)
    ├── sofa_baseline/        # SOFA score baseline calculation
    └── summary/              # Scripts for generating summaries
```

## Data

This project uses Electronic Health Record (EHR) data. The specific datasets used (e.g., eICU, MIMIC, UF) are processed by scripts within the `main/datasets/`, `main/retrospective_cohort/`, and `main/prospective_cohort/` directories.

The primary data format used for training and evaluation is HDF5 (`.h5`). The script [`main/retrospective_cohort/.ipynb_checkpoints/5_build_hdf5-checkpoint.py`](main/retrospective_cohort/.ipynb_checkpoints/5_build_hdf5-checkpoint.py) shows how the final `dataset.h5` file is structured, containing training, validation, external test, and temporal test sets with features (`X`), static data (`static`), and labels (`y_main`, `y_trans`).

Refer to [main/datasets/README.md](main/datasets/README.md) for more details on data sources and initial setup.

## Usage Workflow

1.  **Data Preparation:**
    *   Run the necessary scripts in `main/datasets/` and `main/retrospective_cohort/` (or `main/prospective_cohort/`) to process the raw EHR data and generate the required `dataset.h5` file (e.g., in `OUTPUT_DIR/final_data/`).
2.  **Model Training:**
    *   Navigate to the desired model directory (e.g., `main/models/apricotm/`).
    *   Run the training script (e.g., [`1_train.py`](main/models/apricotm/1_train.py)). This script uses `optuna` for hyperparameter search, trains the model using PyTorch, and saves the best hyperparameters (`best_params.pkl`), model weights (`apricotm_weights.pth`), and architecture (`apricotm_architecture.pth`) to the specified `MODEL_DIR`.
    *   Repeat for other models (e.g., [`main/models/apricott/1_train.py`](main/models/apricott/1_train.py), baseline models).
    *   The model training takes approximatley 2 hours using a NIVIDIA A100 GPU.
3.  **Model Evaluation:**
    *   Run the evaluation script (e.g., `2_eval.py` inferred from [`main/models/apricotm/.ipynb_checkpoints/2_eval-checkpoint.py`](main/models/apricotm/.ipynb_checkpoints/2_eval-checkpoint.py)) to load the trained model and evaluate its performance on the test sets defined in `dataset.h5`. Results are typically saved in a `results` subdirectory within the model's directory.
4.  **Prospective Run (Optional):**
    *   If prospective data is prepared, run the corresponding script (e.g., [`main/models/apricotm/3_prospective.py`](main/models/apricotm/3_prospective.py)) to apply the trained model.
5.  **Analyses:**
    *   Run scripts within the `main/analyses/` subdirectories (e.g., [`main/analyses/calibration/1_calibration.py`](main/analyses/calibration/1_calibration.py), [`main/analyses/integrated_gradients/.ipynb_checkpoints/1_integrated_gradients_table-checkpoint.py`](main/analyses/integrated_gradients/.ipynb_checkpoints/1_integrated_gradients_table-checkpoint.py)) to perform post-hoc analyses on the saved model predictions and results.
6.  **Expected Output:**
    *   Results will be generated under the user defined home directory (HOME_DIR), time window (time_window), and model `/{HOME_DIR}/deepacu/main/{time_window}h_window/model/{model}/results`
 
