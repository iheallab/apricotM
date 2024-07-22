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
import re
import math
from scipy.stats import t

from variables import time_window, MODEL_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/summary/episode_prediction"):
    os.makedirs(f"{OUTPUT_DIR}/summary/episode_prediction")

# Mortality episode prediction

models = ["sofa_criteria", "apricotm"]

outcomes = ["dead", "stable-unstable"]

cohorts = ["int", "ext", "prosp"]

tasks = ["Sensitivity", "Specificity", "Sensitivity (step)", "Specificity (step)", "PPV (step)"]

for outcome in outcomes:
    
    table_comp = pd.DataFrame()

    for model in models:
        
        
        table = pd.read_csv(f"{MODEL_DIR}/{model}/episode_prediction/{outcome}_episode_metrics.csv", index_col=["Unnamed: 0"])
        
        table["model"] = model
        
        table_comp = pd.concat([table_comp, table], axis=0)
        
    table_comp = table_comp.reset_index().rename({"index": "cohort"}, axis=1)
    
    table_comp = table_comp[table_comp["cohort"] != "temp"]
    
    table_comp["cohort"] = pd.Categorical(table_comp["cohort"], categories=cohorts, ordered=True)
    
    table_comp["model"] = pd.Categorical(table_comp["model"], categories=models, ordered=True)
    
    table_comp = table_comp.sort_values(by=["cohort", "model"])
    
    table_comp["Precision"] = table_comp["Precision"].round(2)
    
    table_comp.drop("PPV", axis=1, inplace=True)
    
    
    table_comp = table_comp.astype(str)
    
    # Perform statistical test


    def extract_value(s):

        matches = re.findall(r"\d+\.\d+", s)

        mean = float(matches[0])
        ci = float(matches[2]) - float(matches[1])

        return mean, ci


    for cohort in cohorts:

        for task in tasks:

            sample_1 = table_comp.loc[(table_comp["cohort"] == cohort)&(table_comp["model"] == "sofa_criteria"), task].values[0]
            sample_2 = table_comp.loc[(table_comp["cohort"] == cohort)&(table_comp["model"] == "apricotm"), task].values[0]
            
            mean_1, ci_1 = extract_value(sample_1)
            mean_2, ci_2 = extract_value(sample_2)

            std_1 = (math.sqrt(100) * ci_1) / 3.92
            std_2 = (math.sqrt(100) * ci_2) / 3.92

            dof = 100 + 100 - 2

            if std_1 > 0 or std_2 > 0:

                t_statistic = (mean_1 - mean_2) / (
                    (std_1**2 / 100) + (std_2**2 / 100)
                ) ** 0.5
                p_value = 2 * (1 - t.cdf(abs(t_statistic), dof))

            elif mean_2 > mean_1:

                p_value = 0

            else:

                p_value = 1.00

            if p_value < 0.05:

                table_comp.loc[(table_comp["cohort"] == cohort)&(table_comp["model"] == "apricotm"), task] = (
                    table_comp.loc[(table_comp["cohort"] == cohort)&(table_comp["model"] == "apricotm"), task] + "*"
                )    
    
    cols = ["cohort", "model", "Precision"] + tasks
    table_comp = table_comp.loc[:, cols]
    
    print(table_comp)
    
    table_comp.to_csv(f"{OUTPUT_DIR}/summary/episode_prediction/{outcome}_sofa_criteria_comp.csv")

