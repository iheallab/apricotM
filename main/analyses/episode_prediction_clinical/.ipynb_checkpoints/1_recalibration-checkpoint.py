#%%

# Import libraries

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import os

from variables import time_window, MODEL_DIR

#%%

# # Mortality episode prediction

# models = ["sofa", "sofa_criteria", "apricott", "apricotm"]
# # models = ["sofa_criteria"]

# outcomes = ["dead"]

# print("Running mortality episode recalibration")

# for model in models:

#     if not os.path.exists(f"{MODEL_DIR}/{model}/episode_prediction"):
#         os.makedirs(f"{MODEL_DIR}/{model}/episode_prediction")

#     for outcome in outcomes:

#         cohorts = ["int", "ext", "temp", "prosp"]

#         table = pd.DataFrame()

#         for cohort in cohorts:

#             true_ext = pd.read_csv(
#                 f"{MODEL_DIR}/{model}/results/{cohort}_true_labels.csv"
#             )
#             pred_ext = pd.read_csv(
#                 f"{MODEL_DIR}/{model}/results/{cohort}_pred_labels.csv"
#             )

#             y_true_labels = true_ext[outcome].values
#             y_pred_labels = pred_ext[outcome].values

#             y_ids = true_ext["icustay_id"].values

#             predictions = pd.DataFrame(
#                 {
#                     "icustay_id": y_ids,
#                     "true": true_ext[outcome].values,
#                     "prob": y_pred_labels,
#                 }
#             )

#             predictions.insert(0, "time", np.nan)

#             def time_calc(group):
#                 time = np.arange(0, int(time_window * len(group)), time_window)
#                 group["time"] = time
#                 return group

#             predictions = (
#                 predictions.groupby("icustay_id").apply(time_calc).reset_index(drop=True)
#             )

#             group_pred = predictions.groupby("icustay_id")

#             true_label = []
#             probs = []

#             for name, group in group_pred:

#                 state = group["true"].max()

#                 prob_state = group["prob"].max()

#                 if state == 1:

#                     pos = group[group["true"] == 1]
#                     pos = pos["time"].min()

#                     early = group[(group["time"] <= pos)]

#                     if len(early) > 0:

#                         prob_state = early["prob"].max()

#                     else:

#                         prob_state = 0

#                 true_label.append(state)
#                 probs.append(prob_state)

#             true_label = np.array(true_label)
#             probs = np.array(probs)
            
#             recal = pd.DataFrame(data={"true": true_label, "pred": probs})

#             recal.to_csv(
#                 f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome}_episodes.csv",
#                 index=False,
#             )

#             predictions.to_csv(
#                 f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome}_step_predictions.csv",
#                 index=False,
#             )


#%%

# Unstable episode prediction
# models = ["sofa_criteria", "sofa", "apricott", "apricotm"]

models = ["apricott"]
# models = ["sofa_criteria"]

# outcomes = [["stable-unstable", "unstable-stable"],
#            ["no mv-mv", "mv-no mv"],
#            ["no vp- vp", "vp-no vp"],
#            ["no crrt-crrt", "crrt-no crrt"]]

outcomes = [["stable-unstable", "unstable-stable"]]

print("Running unstable episode recalibration")

for model in models:

    for outcome in outcomes:

        cohorts = ["int", "ext", "temp", "prosp"]

        table = pd.DataFrame()

        for cohort in cohorts:

            true_ext = pd.read_csv(
                f"{MODEL_DIR}/{model}/results/{cohort}_true_labels.csv"
            )
            pred_ext = pd.read_csv(
                f"{MODEL_DIR}/{model}/results/{cohort}_pred_labels.csv"
            )

            y_true_labels = true_ext[outcome[0]].values
            y_pred_labels = pred_ext[outcome[0]].values

            y_ids = true_ext["icustay_id"].values

            predictions = pd.DataFrame(
                {
                    "icustay_id": y_ids,
                    "true": true_ext["unstable"].values,
                    "prob": y_pred_labels,
                    "transition": true_ext[outcome[0]].values,
                    "transition_before": true_ext[outcome[0]].values,
                    "transition_after": true_ext[outcome[1]].values,
                }
            )

            predictions["no change"] = 0

            condition = (predictions["transition"] == 0) & (predictions["true"] == 1)

            predictions.loc[condition, "no change"] = 1

            predictions.insert(0, "time", np.nan)

            def time_calc(group):
                time = np.arange(0, int(time_window * len(group)), time_window)
                group["time"] = time
                return group

            predictions = (
                predictions.groupby("icustay_id").apply(time_calc).reset_index(drop=True)
            )

            predictions = predictions[predictions["no change"] == 0]

            predictions = predictions.reset_index(drop=True)

            group_pred = predictions.groupby("icustay_id")

            true_label = []
            probs = []

            for name, group in group_pred:

                state = group["true"].max()

                prob_state = group["prob"].max()

                if state == 1:

                    pos = group[group["true"] == 1]
                    pos = pos["time"].min()

                    early = group[(group["time"] <= pos)]

                    if len(early) > 0:

                        prob_state = early["prob"].max()

                    else:

                        prob_state = 0

                true_label.append(state)
                probs.append(prob_state)

            true_label = np.array(true_label)
            probs = np.array(probs)

            recal = pd.DataFrame(data={"true": true_label, "pred": probs})

            recal.to_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome[0]}_episodes.csv",
                index=False,
            )
            
            predictions.to_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome[0]}_step_predictions.csv",
                index=False,
            )

# %%
