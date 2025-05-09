HOME_DIR = "/orange/parisa.rashidi"
time_window = 4
DATA_DIR = f"{HOME_DIR}/deepacu/main/datasets"
OUTPUT_DIR = f"{HOME_DIR}/deepacu/main/{time_window}h_window"
ANALYSIS_DIR = f"{HOME_DIR}/deepacu/main/analyses/ablations"
SCALERS_DIR = f"{HOME_DIR}/deepacu/main/{time_window}h_window/model"
VAR_MAP = f"{HOME_DIR}/deepacu/main/datasets"
# PROSP_DATA_DIR = f"{HOME_DIR}/deepacu/main/analyses/apricot_comparison"
PROSP_DATA_DIR = f"{HOME_DIR}/deepacu/main/prospective_cohort/{time_window}h_window"
model = "apricotm"
MODEL_DIR = f"{HOME_DIR}/deepacu/main/{time_window}h_window/model/{model}"