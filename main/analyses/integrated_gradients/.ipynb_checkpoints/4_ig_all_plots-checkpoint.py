#%%

# # Import libraries

# import pandas as pd
# import numpy as np
# import h5py
# import pickle
# import matplotlib.pyplot as plt

# import torch
# import captum.attr as attr

# from variables import MODEL_DIR

# #%%

# # Interpretability all sets

# cohorts = ["validation", "external_test", "temporal_test", "prosp"]

# targets = [
#     "discharge",
#     "stable",
#     "unstable",
#     "deceased",
#     "unstable-stable",
#     "stable-unstable",
#     "no mv-mv",
#     "no vp-vp",
#     "no crrt-crrt",
# ]

# for cohort in cohorts:

#     for target in targets:

#         df_sorted = pd.read_csv(
#             f"{MODEL_DIR}/integrated_gradients/{cohort}_integrated_gradients_{target}.csv"
#         )

#         df_sorted = df_sorted.iloc[:15]

#         # Create a horizontal bar plot
#         plt.figure(figsize=(8, 8))
#         plt.barh(df_sorted["variable"], df_sorted["ig_variable"], color="skyblue")

#         plt.yticks(fontsize=20)

#         plt.xticks(np.arange(0, np.max(df_sorted["ig_variable"].values), 0.1), fontsize=20)

#         # Optional: Customize the plot further (e.g., add labels and titles)
#         plt.xlabel("IG attribution", fontsize=20)

#         plt.tight_layout()

#         plt.savefig(
#             f"{MODEL_DIR}/integrated_gradients/{cohort}_{target}_ig.png",
#             dpi=400,
#         )

#         # plt.show()

#         plt.clf()



# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from variables import MODEL_DIR

# Interpretability all sets
cohorts = ["validation", "external_test", "temporal_test", "prosp"]

targets = [
    "discharge",
    "stable",
    "unstable",
    "deceased",
    "unstable-stable",
    "stable-unstable",
    "no mv-mv",
    "no vp-vp",
    "no crrt-crrt",
]

targets_titles = [
    "Discharge",
    "Stable",
    "Unstable",
    "Deceased",
    "Unstable-Stable",
    "Stable-Unstable",
    "MV",
    "VP",
    "CRRT",
]


for cohort in cohorts:
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))  # Create a 3x3 grid of subplots
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    
    count = 0

    for i, target in enumerate(targets):
        df_sorted = pd.read_csv(
            f"{MODEL_DIR}/integrated_gradients/{cohort}_integrated_gradients_{target}.csv"
        )

        df_sorted = df_sorted.iloc[:15]

        ax = axes[i]  # Select the current subplot
        ax.barh(df_sorted["variable"], df_sorted["ig_variable"], color="skyblue")

        ax.set_yticks(df_sorted["variable"])
        ax.set_yticklabels(df_sorted["variable"], fontsize=14)

        ax.set_xticks(np.arange(0, np.max(df_sorted["ig_variable"].values) + 0.1, 0.1))
        ax.tick_params(axis='x', labelsize=14)

        ax.set_xlabel("IG attribution", fontsize=12)
        ax.set_title(targets_titles[count], fontsize=16, fontweight='bold')
        
        count += 1

    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/integrated_gradients/{cohort}_ig_grid.png", dpi=400)
    plt.clf()

