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

DATA_DIR = "/home/contreras.miguel/deepacu"

#%%

cohorts = ["int", "ext", "temp"]

for cohort in cohorts:

    model_true = pd.read_csv(f"{DATA_DIR}/results/mamba_tuned/{cohort}_true_labels.csv")
    model_true = model_true.iloc[:, 1:]

    random_sample = np.random.choice(
        len(model_true), size=int(0.1 * len(model_true)), replace=False
    )

    model_true_train = model_true.iloc[random_sample]

    # CatBoost Calibration

    cb_probs = pd.read_csv(
        f"{DATA_DIR}/baseline/catboost/catboost_results_{cohort}.csv"
    )

    cb_probs_train = cb_probs.iloc[random_sample]

    calibrator_cb = {}

    for column in model_true.columns:
        # Fit isotonic regression on the training data

        iso_reg = IsotonicRegression(out_of_bounds="clip")

        iso_reg.fit(cb_probs_train.loc[:, column], model_true_train.loc[:, column])

        calibrator_cb[f"calibrator_{column}"] = iso_reg

    # GRU Calibration

    gru_probs = pd.read_csv(f"{DATA_DIR}/baseline/gru/{cohort}_pred_labels.csv")

    gru_probs = gru_probs.iloc[:, 1:]

    gru_probs_train = gru_probs.iloc[random_sample]

    calibrator_gru = {}

    for column in model_true.columns:
        # Fit isotonic regression on the training data

        iso_reg = IsotonicRegression(out_of_bounds="clip")

        iso_reg.fit(gru_probs_train.loc[:, column], model_true_train.loc[:, column])

        calibrator_gru[f"calibrator_{column}"] = iso_reg

    # Transformer Calibration

    trans_probs = pd.read_csv(
        f"{DATA_DIR}/baseline/transformer/{cohort}_pred_labels.csv"
    )

    trans_probs = trans_probs.iloc[:, 1:]

    trans_probs_train = trans_probs.iloc[random_sample]

    calibrator_trans = {}

    for column in model_true.columns:
        # Fit isotonic regression on the training data

        iso_reg = IsotonicRegression(out_of_bounds="clip")

        iso_reg.fit(trans_probs_train.loc[:, column], model_true_train.loc[:, column])

        calibrator_trans[f"calibrator_{column}"] = iso_reg

    # APRICOT Calibration

    apricot_probs = pd.read_csv(
        f"{DATA_DIR}/results/apricot_tuned/{cohort}_pred_labels.csv"
    )

    apricot_probs = apricot_probs.iloc[:, 1:]

    apricot_probs_train = apricot_probs.iloc[random_sample]

    calibrator_apricot = {}

    for column in model_true.columns:
        # Fit isotonic regression on the training data

        iso_reg = IsotonicRegression(out_of_bounds="clip")

        iso_reg.fit(apricot_probs_train.loc[:, column], model_true_train.loc[:, column])

        calibrator_apricot[f"calibrator_{column}"] = iso_reg

    # Mamba Calibration

    mamba_probs = pd.read_csv(
        f"{DATA_DIR}/results/mamba_tuned/{cohort}_pred_labels.csv"
    )

    mamba_probs = mamba_probs.iloc[:, 1:]

    mamba_probs_train = mamba_probs.iloc[random_sample]

    calibrator_mamba = {}

    for column in model_true.columns:
        # Fit isotonic regression on the training data

        iso_reg = IsotonicRegression(out_of_bounds="clip")

        iso_reg.fit(mamba_probs_train.loc[:, column], model_true_train.loc[:, column])

        calibrator_mamba[f"calibrator_{column}"] = iso_reg

    # Calibrate Models

    # CatBoost

    for column in cb_probs.columns:
        cb_probs.loc[:, column] = calibrator_cb[f"calibrator_{column}"].transform(
            cb_probs.loc[:, column]
        )

    # GRU

    for column in gru_probs.columns[1:]:
        gru_probs.loc[:, column] = calibrator_gru[f"calibrator_{column}"].transform(
            gru_probs.loc[:, column]
        )

    # Transformer

    for column in trans_probs.columns[1:]:
        trans_probs.loc[:, column] = calibrator_trans[f"calibrator_{column}"].transform(
            trans_probs.loc[:, column]
        )

    # APRICOT

    for column in apricot_probs.columns[1:]:
        apricot_probs.loc[:, column] = calibrator_apricot[
            f"calibrator_{column}"
        ].transform(apricot_probs.loc[:, column])

    # Mamba

    for column in mamba_probs.columns[1:]:
        mamba_probs.loc[:, column] = calibrator_mamba[f"calibrator_{column}"].transform(
            mamba_probs.loc[:, column]
        )

    outcomes = [
        ["discharge", "Discharge"],
        ["stable", "Stable"],
        ["unstable", "Unstable"],
        ["dead", "Deceased"],
        ["unstable-stable", "Unstable-Stable"],
        ["stable-unstable", "Stable-Unstable"],
        ["no mv-mv", "MV"],
        ["no vp- vp", "VP"],
        ["no crrt-crrt", "CRRT"],
    ]

    mamba_probs.to_csv(
        f"{DATA_DIR}/results/mamba_tuned/calibration/{cohort}_apricot_cal_probs.csv",
        index=None,
    )

    for outcome in outcomes:

        models = [
            cb_probs[outcome[0]].values,
            gru_probs[outcome[0]].values,
            trans_probs[outcome[0]].values,
            apricot_probs[outcome[0]].values,
            mamba_probs[outcome[0]].values,
        ]  # Replace with actual model instances
        labels = ["CatBoost", "GRU", "Transformer", "APRICOT-T", "APRICOT-M"]
        colors = ["#FFC374", "#F9E8C9", "#98ABEE", "#1D24CA", "#201658"]

        plt.figure(figsize=(10, 8))

        for model, label, color in zip(models, labels, colors):

            # Calculate the reliability (calibration) curve
            prob_true, prob_pred = calibration_curve(
                model_true[outcome[0]], model, n_bins=10, strategy="uniform"
            )

            # Calculate Brier score (optional)
            brier_score = brier_score_loss(model_true[outcome[0]], model)

            # Plot the calibration curve
            plt.plot(
                prob_pred,
                prob_true,
                marker="o",
                color=color,
                label=f"{label} {brier_score:.4f}",
            )

        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )
        plt.xlabel("Mean Predicted Probability", fontsize=15)
        plt.ylabel("Fraction of Positives", fontsize=15)
        plt.title(outcome[1], fontsize=24)
        plt.legend(fontsize=15)

        plt.savefig(
            f"{DATA_DIR}/results/mamba_tuned/calibration/{cohort}_calibration_{outcome[1]}.png",
            dpi=400,
        )

        plt.show()
        plt.clf()


# %%
