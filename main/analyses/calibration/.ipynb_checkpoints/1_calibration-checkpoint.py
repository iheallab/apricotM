#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import os

from variables import (
    time_window,
    OUTPUT_DIR,
    BASELINE_DIR,
    MODELS_DIR,
    MODELS_NAME_DIR,
)

baselines = ["catboost", "gru", "transformer"]
models = ["apricott", "apricotm"]

cohorts = ["int", "ext", "temp", "prosp"]

#%%

# Calibrate each model

for baseline in baselines:

    for cohort in cohorts:

        model_true = pd.read_csv(
            f"{BASELINE_DIR}/{baseline}/results/{cohort}_true_labels.csv"
        )

        model_probs = pd.read_csv(
            f"{BASELINE_DIR}/{baseline}/results/{cohort}_pred_labels.csv"
        )

        random_sample = np.random.choice(
            len(model_true), size=int(0.1 * len(model_true)), replace=False
        )

        model_true_train = model_true.iloc[random_sample]
        model_probs_train = model_probs.iloc[random_sample]

        calibrator = {}

        for column in model_true.columns:
            # Fit isotonic regression on the training data

            iso_reg = IsotonicRegression(out_of_bounds="clip")

            iso_reg.fit(
                model_probs_train.loc[:, column], model_true_train.loc[:, column]
            )

            calibrator[f"calibrator_{column}"] = iso_reg

        # Calibrate model

        for column in model_probs.columns:
            model_probs.loc[:, column] = calibrator[f"calibrator_{column}"].transform(
                model_probs.loc[:, column]
            )
            
        if not os.path.exists(f"{BASELINE_DIR}/{baseline}/calibration"):
            os.makedirs(f"{BASELINE_DIR}/{baseline}/calibration")

        model_probs.to_csv(f"{BASELINE_DIR}/{baseline}/calibration/{cohort}_cal_probs.csv", index=False)
        model_true.to_csv(f"{BASELINE_DIR}/{baseline}/calibration/{cohort}_true_labels.csv", index=False)


for model in models:

    for cohort in cohorts:

        model_true = pd.read_csv(
            f"{MODELS_DIR}/{model}/results/{cohort}_true_labels.csv"
        ).iloc[:, 1:]

        model_probs = pd.read_csv(
            f"{MODELS_DIR}/{model}/results/{cohort}_pred_labels.csv"
        ).iloc[:, 1:]

        random_sample = np.random.choice(
            len(model_true), size=int(0.1 * len(model_true)), replace=False
        )

        model_true_train = model_true.iloc[random_sample]
        model_probs_train = model_probs.iloc[random_sample]

        calibrator = {}

        for column in model_true.columns:
            # Fit isotonic regression on the training data

            iso_reg = IsotonicRegression(out_of_bounds="clip")

            iso_reg.fit(
                model_probs_train.loc[:, column], model_true_train.loc[:, column]
            )

            calibrator[f"calibrator_{column}"] = iso_reg

        # Calibrate model

        for column in model_probs.columns:
            model_probs.loc[:, column] = calibrator[f"calibrator_{column}"].transform(
                model_probs.loc[:, column]
            )
            
        if not os.path.exists(f"{MODELS_DIR}/{model}/calibration"):
            os.makedirs(f"{MODELS_DIR}/{model}/calibration")

        model_probs.to_csv(f"{MODELS_DIR}/{model}/calibration/{cohort}_cal_probs.csv", index=False)
        model_true.to_csv(f"{MODELS_DIR}/{model}/calibration/{cohort}_true_labels.csv", index=False)
