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

# Episode prediction

models = ["apricotm"]

outcomes = ["dead", "stable-unstable", "no mv-mv", "no vp- vp"]

cohorts = ["int", "ext", "prosp"]

tasks = ["Sensitivity", "Specificity", "Sensitivity (step)", "Specificity (step)", "PPV (step)"]

all_episodes = pd.DataFrame()

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
    
    
    
    table_comp.drop("PPV", axis=1, inplace=True)
    
    table_comp = table_comp[table_comp["Precision"] == 0.33]
    
    cols = ["cohort", "model", "Earliest prediction", "Number of alerts"] + tasks
    table_comp = table_comp.loc[:, cols]

        
    all_episodes = pd.concat([all_episodes, table_comp])
    
print(all_episodes)
    
all_episodes.to_csv(f"{OUTPUT_DIR}/summary/episode_prediction/all_episodes.csv")