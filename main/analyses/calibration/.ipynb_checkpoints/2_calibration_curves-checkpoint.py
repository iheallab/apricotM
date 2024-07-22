#%%

# # Import libraries

# import pandas as pd
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import brier_score_loss
# from sklearn.calibration import IsotonicRegression
# from sklearn.linear_model import LogisticRegression
# import os

# from variables import (
#     BASELINE_DIR,
#     CALIB_DIR,
#     MODELS_NAME_DIR,
#     MODELS_DIR,
# )

# if not os.path.exists(f"{CALIB_DIR}"):
#     os.makedirs(f"{CALIB_DIR}")

# baselines = os.listdir(BASELINE_DIR)
# models = os.listdir(MODELS_NAME_DIR)

# all_models = baselines + models

# cohorts = ["int", "ext", "temp", "prosp"]

# #%%

# # Get calibration curve for each model and outcome
# outcomes = [
#     ["discharge", "Discharge"],
#     ["stable", "Stable"],
#     ["unstable", "Unstable"],
#     ["dead", "Deceased"],
#     ["unstable-stable", "Unstable-Stable"],
#     ["stable-unstable", "Stable-Unstable"],
#     ["no mv-mv", "MV"],
#     ["no vp- vp", "VP"],
#     ["no crrt-crrt", "CRRT"],
# ]

# for cohort in cohorts:

#     cal_probs = []
#     true_labels = []

#     for model in all_models:

#         probs = pd.read_csv(f"{CALIB_DIR}/{model}_{cohort}_cal_probs.csv")
#         label = pd.read_csv(f"{CALIB_DIR}/{model}_{cohort}_true_labels.csv")

#         cal_probs.append(probs)
#         true_labels.append(label)

#     for outcome in outcomes:

#         labels = ["CatBoost", "GRU", "Transformer", "APRICOT-M", "APRICOT-T"]
#         colors = ["#FFC374", "#F9E8C9", "#98ABEE", "#1D24CA", "#201658"]

#         plt.figure(figsize=(10, 8))

#         for model, label, color, model_true in zip(
#             cal_probs, labels, colors, true_labels
#         ):

#             # Calculate the reliability (calibration) curve
#             prob_true, prob_pred = calibration_curve(
#                 model_true[outcome[0]], model[outcome[0]], n_bins=10, strategy="uniform"
#             )

#             # Calculate Brier score (optional)
#             brier_score = brier_score_loss(model_true[outcome[0]], model[outcome[0]])

#             # Plot the calibration curve
#             plt.plot(
#                 prob_pred,
#                 prob_true,
#                 marker="o",
#                 color=color,
#                 label=f"{label} {brier_score:.4f}",
#             )

#         plt.plot(
#             [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
#         )
#         plt.xlabel("Mean Predicted Probability", fontsize=15)
#         plt.ylabel("Fraction of Positives", fontsize=15)
#         plt.title(outcome[1], fontsize=24)
#         plt.legend(fontsize=15)

#         plt.savefig(
#             f"{CALIB_DIR}/{cohort}_calibration_{outcome[1]}.png",
#             dpi=400,
#         )

#         plt.show()
#         plt.clf()



# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import os

from variables import (
    BASELINE_DIR,
    CALIB_DIR,
    MODELS_DIR,
)

if not os.path.exists(f"{CALIB_DIR}"):
    os.makedirs(f"{CALIB_DIR}")

baselines = ['catboost', 'gru', 'transformer']
models = ['apricott', 'apricotm']

all_models = baselines + models

cohorts = ["int", "ext", "temp", "prosp"]

# Get calibration curve for each model and outcome
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

for cohort in cohorts:

    cal_probs = []
    true_labels = []

    for model in all_models:
        
        if model in baselines:

            probs = pd.read_csv(f"{BASELINE_DIR}/{model}/calibration/{cohort}_cal_probs.csv")
            label = pd.read_csv(f"{BASELINE_DIR}/{model}/calibration/{cohort}_true_labels.csv")

            cal_probs.append(probs)
            true_labels.append(label)
            
        else:

            probs = pd.read_csv(f"{MODELS_DIR}/{model}/calibration/{cohort}_cal_probs.csv")
            label = pd.read_csv(f"{MODELS_DIR}/{model}/calibration/{cohort}_true_labels.csv")

            cal_probs.append(probs)
            true_labels.append(label)

    # Create a 3x3 plot for each cohort
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for i, outcome in enumerate(outcomes):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        labels = ["CatBoost", "GRU", "Transformer", "APRICOT-M", "APRICOT-T"]
        colors = ["#FFC374", "#F9E8C9", "#98ABEE", "#1D24CA", "#201658"]

        for model, label, color, model_true in zip(
            cal_probs, labels, colors, true_labels
        ):

            # Calculate the reliability (calibration) curve
            prob_true, prob_pred = calibration_curve(
                model_true[outcome[0]], model[outcome[0]], n_bins=10, strategy="uniform"
            )

            # Calculate Brier score (optional)
            brier_score = brier_score_loss(model_true[outcome[0]], model[outcome[0]])

            # Plot the calibration curve
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                color=color,
                label=f"{label} {brier_score:.4f}",
            )

        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(outcome[1], fontsize=18, fontweight="bold")
        ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        f"{CALIB_DIR}/{cohort}_calibration_plots.png",
        dpi=400,
    )

    plt.show()

