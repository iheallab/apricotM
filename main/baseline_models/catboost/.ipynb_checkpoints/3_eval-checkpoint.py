#%%
# Import libraries

import pandas as pd
import numpy as np
import h5py
import pickle
import os

from variables import time_window, MODEL_DIR


#%%

# Evaluate Catboost model

groups = [
    ["validation", "int"],
    ["external_test", "ext"],
    ["temporal_test", "temp"],
    ["prospective", "prosp"],
]

with open("%s/catboost.pkl" % MODEL_DIR, "rb") as f:
    binary_classifiers = pickle.load(f)

for group in groups:

    with h5py.File("%s/dataset_catboost.h5" % MODEL_DIR, "r") as f:
        data = f[group[0]]
        X = data["X"][:]
        y_trans = data["y_trans"][:]
        y_main = data["y_main"][:]

    class_probabilities = []

    for model in binary_classifiers:
        # Predict the probability of positive class (1) for each sample in the test set
        y_pred = model.predict_proba(X)[:, 1]
        class_probabilities.append(y_pred)

    class_probabilities = pd.DataFrame(class_probabilities).T

    cols = [
        "discharge",
        "stable",
        "unstable",
        "dead",
        "unstable-stable",
        "stable-unstable",
        "mv-no mv",
        "no mv-mv",
        "vp-no vp",
        "no vp- vp",
        "crrt-no crrt",
        "no crrt-crrt",
    ]

    class_probabilities.columns = cols

    cols = [
        "discharge",
        "stable",
        "unstable",
        "dead",
        "unstable-stable",
        "stable-unstable",
        "mv-no mv",
        "no mv-mv",
        "vp-no vp",
        "no vp- vp",
        "crrt-no crrt",
        "no crrt-crrt",
    ]

    true_labels = np.concatenate([y_main, y_trans], axis=1)
    true_labels = pd.DataFrame(true_labels, columns=cols)

    class_probabilities.to_csv(
        f"{MODEL_DIR}/results/{group[1]}_pred_labels.csv", index=None
    )

    true_labels.to_csv(f"{MODEL_DIR}/results/{group[1]}_true_labels.csv", index=None)

# %%
