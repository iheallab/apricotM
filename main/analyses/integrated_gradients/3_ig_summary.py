#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

import torch
import captum.attr as attr

from variables import MODEL_DIR

#%%

# Interpretability all sets

cohorts = ["validation", "external_test", "prosp"]

for cohort in cohorts:

    all_ig = pd.DataFrame(data=[], columns=["variable"])

    targets = ["unstable", "deceased", "stable-unstable", "no mv-mv", "no vp-vp", "no crrt-crrt"]

    for i in range(len(targets)):

        ig = pd.read_csv(
            f"{MODEL_DIR}/integrated_gradients/{cohort}_integrated_gradients_{targets[i]}.csv"
        )

        ig = ig.rename({"ig_variable": f"ig_{targets[i]}"}, axis=1)

        all_ig = all_ig.merge(ig, on="variable", how="outer")

    all_ig["mean"] = all_ig.iloc[:, 1:].mean(axis=1)
    
    # Sort dataframe by the 'sum' column in descending order
    df_sorted = all_ig.sort_values(by="mean", ascending=False)

    df_sorted = df_sorted.iloc[:15]

    # Create a horizontal bar plot
    plt.figure(figsize=(8, 8))
    plt.barh(df_sorted["variable"], df_sorted["mean"], color="skyblue")

    plt.yticks(fontsize=20)
    
    plt.xticks(np.arange(0, np.max(df_sorted["mean"].values), 0.1), fontsize=20)

    # Optional: Customize the plot further (e.g., add labels and titles)
    plt.xlabel("IG attribution", fontsize=20)

    plt.tight_layout()

    plt.savefig(
        f"{MODEL_DIR}/integrated_gradients/{cohort}_ig_acuity_summary.png",
        dpi=400,
    )

    # plt.show()

    plt.clf()

