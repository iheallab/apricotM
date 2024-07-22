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

from variables import time_window, MODELS_DIR, BASELINE_DIR

#%%

directories = [[BASELINE_DIR, ["catboost", "gru", "transformer"]], [MODELS_DIR, ["apricott", "apricotm"]]]

# Mortality episode prediction

outcomes = ["dead"]

print("Running mortality episode recalibration")

for directory in directories:
    
    for model in directory[1]:

        if not os.path.exists(f"{directory[0]}/{model}/episode_prediction"):
            os.makedirs(f"{directory[0]}/{model}/episode_prediction")
    
        for outcome in outcomes:
    
            cohorts = ["int", "ext", "temp", "prosp"]
    
            table = pd.DataFrame()
    
            for cohort in cohorts:
            
                true_ext = pd.read_csv(f"{MODELS_DIR}/apricotm/results/{cohort}_true_labels.csv")

                y_ids = true_ext["icustay_id"].values
                
                print(len(y_ids))
    
                true_ext = pd.read_csv(
                    f"{directory[0]}/{model}/results/{cohort}_true_labels.csv"
                )
                pred_ext = pd.read_csv(
                    f"{directory[0]}/{model}/results/{cohort}_pred_labels.csv"
                )
            
                print(len(true_ext))
    
                y_true_labels = true_ext[outcome].values
                y_pred_labels = pred_ext[outcome].values
    
                # y_ids = true_ext["icustay_id"].values
    
                predictions = pd.DataFrame(
                    {
                        "icustay_id": y_ids,
                        "true": true_ext[outcome].values,
                        "prob": y_pred_labels,
                    }
                )
    
                predictions.insert(0, "time", np.nan)
    
                def time_calc(group):
                    time = np.arange(0, int(time_window * len(group)), time_window)
                    group["time"] = time
                    return group
    
                predictions = (
                    predictions.groupby("icustay_id").apply(time_calc).reset_index(drop=True)
                )
    
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
                    f"{directory[0]}/{model}/episode_prediction/{cohort}_{outcome}_episodes.csv",
                    index=False,
                )
    
                predictions.to_csv(
                    f"{directory[0]}/{model}/episode_prediction/{cohort}_{outcome}_step_predictions.csv",
                    index=False,
                )


#%%

# Transition episode prediction

outcomes = [["stable-unstable", "unstable-stable"],
            ["no mv-mv", "mv-no mv"],
            ["no vp- vp", "vp-no vp"],
            ["no crrt-crrt", "crrt-no crrt"]]

print("Running transition episode recalibration")

for directory in directories:
    
    for model in directory[1]:

        for outcome in outcomes:
    
            cohorts = ["int", "ext", "temp", "prosp"]
    
            table = pd.DataFrame()
    
            for cohort in cohorts:
            
                true_ext = pd.read_csv(f"{MODELS_DIR}/apricotm/results/{cohort}_true_labels.csv")

                y_ids = true_ext["icustay_id"].values
                
                print(len(y_ids))
    
                true_ext = pd.read_csv(
                    f"{directory[0]}/{model}/results/{cohort}_true_labels.csv"
                )
                pred_ext = pd.read_csv(
                    f"{directory[0]}/{model}/results/{cohort}_pred_labels.csv"
                )
            
                print(len(true_ext))
    
                y_true_labels = true_ext[outcome[0]].values
                y_pred_labels = pred_ext[outcome[0]].values
    
                # y_ids = true_ext["icustay_id"].values
    
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
                    f"{directory[0]}/{model}/episode_prediction/{cohort}_{outcome[0]}_episodes.csv",
                    index=False,
                )
                
                predictions.to_csv(
                    f"{directory[0]}/{model}/episode_prediction/{cohort}_{outcome[0]}_step_predictions.csv",
                    index=False,
                )

# %%
