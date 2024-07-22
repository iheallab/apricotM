#%%

# Import libraries

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from pywaffle import Waffle

import os

from variables import MODEL_DIR

if not os.path.exists(f"{MODEL_DIR}/confusion_matrix"):
    os.makedirs(f"{MODEL_DIR}/confusion_matrix")


#%%

cohorts = ["int", "ext", "prosp"]

figures = []

for cohort in cohorts:

    # Load predictions

    y_true = pd.read_csv(f"{MODEL_DIR}/results/{cohort}_true_labels.csv").iloc[:, 1:]
    y_pred = pd.read_csv(f"{MODEL_DIR}/results/{cohort}_pred_labels.csv").iloc[:, 1:]

    labels = y_true.columns.tolist()

    thresh = []

    for label in labels:

        fpr, tpr, thresholds = roc_curve(
            y_true.loc[:, label].values, y_pred.loc[:, label].values
        )
        roc_auc_class = auc(fpr, tpr)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        
        precision, recall, thresholds = precision_recall_curve(
            y_true.loc[:, label].values, y_pred.loc[:, label].values
        )
        pr_auc_class = auc(recall, precision)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        best_thresh_pr = thresholds[ix]
        
        thresh.append(best_thresh)

    y_pred_class = y_pred.copy()
    for i in range(len(thresh)):
        y_pred_class.iloc[:, i] = (y_pred.iloc[:, i] >= thresh[i]).astype("int")

    y_true_label = y_true[["discharge", "stable", "unstable", "dead"]].idxmax(axis=1)

    condition_deceased = y_pred_class["dead"] == 1
    condition_unstable = (
        (y_pred_class["no mv-mv"] == 1)
        | (y_pred_class["no vp- vp"] == 1)
        | (y_pred_class["no crrt-crrt"] == 1)
        | (y_pred_class["unstable"] == 1)
        | (y_pred_class["stable-unstable"] == 1)
    )
    condition_stable = (
        (y_pred_class["stable"] == 1) | (y_pred_class["unstable-stable"] == 1)
    ) & (
        (y_pred_class["no mv-mv"] == 0)
        & (y_pred_class["no vp- vp"] == 0)
        & (y_pred_class["no crrt-crrt"] == 0)
    )
    condition_discharge = (
        (y_pred_class["discharge"] == 1)
        & (y_pred_class["no mv-mv"] == 0)
        & (y_pred_class["no vp- vp"] == 0)
        & (y_pred_class["no crrt-crrt"] == 0)
    )

    y_pred_class["predicted_state"] = "stable"
    y_pred_class.loc[condition_stable, "predicted_state"] = "stable"
    y_pred_class.loc[condition_discharge, "predicted_state"] = "discharge"
    y_pred_class.loc[condition_unstable, "predicted_state"] = "unstable"
    y_pred_class.loc[condition_deceased, "predicted_state"] = "dead"

    y_pred_label = y_pred_class["predicted_state"]

    map_states = {
        "discharge": 0,
        "stable": 1,
        "unstable": 2,
        "dead": 3,
    }

    y_true_label = y_true_label.map(map_states).values
    y_pred_label = y_pred_label.map(map_states).values

    labels = ["discharge", "stable", "unstable", "dead"]

    clf_report_print = classification_report(
        y_true_label, y_pred_label, target_names=labels
    )
    cf_matrix = confusion_matrix(y_true_label, y_pred_label)
    print(clf_report_print)
    print(cf_matrix)

    # Predicted vs true classes

    prop_df = pd.DataFrame(data=[], columns=labels, index=labels)
    num_df = pd.DataFrame(data=[], columns=labels, index=labels)

    for i in range(len(labels)):
        true_class = np.sum(cf_matrix[i, :])

        for j in range(len(labels)):

            prop_df.loc[labels[i], labels[j]] = cf_matrix[i, j] / true_class
            num_df.loc[labels[i], labels[j]] = cf_matrix[i, j]

    percentages = prop_df.applymap(lambda x: round(x, 2))

    percentages = percentages * 100

    percentages = percentages.div(percentages.sum(axis=1) / 100, axis=0)
    
    figures_cohort = []
    
    # Create a waffle chart for each index
    for index in prop_df.index:

        correct_prediction = index

        discharge = percentages.loc[index, "discharge"]
        stable = percentages.loc[index, "stable"]
        unstable = percentages.loc[index, "unstable"]
        dead = percentages.loc[index, "dead"]

        cols = percentages.loc[index].sort_values(ascending=False).index

        outcomes = {
            "Discharge": discharge,
            "Stable": stable,
            "Unstable": unstable,
            "Deceased": dead,
        }

        outcomes = dict(
            sorted(outcomes.items(), key=lambda item: item[1], reverse=True)
        )

        keys = list(outcomes.keys())
        values = list(outcomes.values())

        colors = [
            "lightgreen" if col == correct_prediction else "red" for col in cols
        ]

        values = [round(value) for value in values]

        while sum(values) != 100:
            if sum(values) < 100:
                values[0] += 1
            else:
                values[0] = values[0] - 1

        legend_labels = [
            f"{keys[0]}: {values[0]:.0f}%",
            f"{keys[1]}: {values[1]:.0f}%",
            f"{keys[2]}: {values[2]:.0f}%",
            f"{keys[3]}: {values[3]:.0f}%",
        ]

        fig = plt.figure(
            FigureClass=Waffle,
            rows=10,
            values=values,
            colors=colors,
            icons="person",
            font_size=30,
            legend={
                "loc": "upper center",
                "bbox_to_anchor": (0.5, -0.1),
                "labels": legend_labels,
                "fontsize": 17,
            },
            figsize=(6, 6),
        )
        
        plt.tight_layout()
                
        figures_cohort.append(fig)
        
    figures.append(figures_cohort)
    
# Create a figure with a 3x3 grid of subplots
fig, axs = plt.subplots(3, 4, figsize=(20, 20))

for i in range(len(figures)):
        
    for j in range(len(figures[i])):
        
        # Clear the current axes
        axs[i, j].clear()
        # Remove the axis for individual plots
        axs[i, j].axis('off')
        # Transfer the contents of the figure to the current subplot
        axs[i, j].figure = figures[i][j]
        figures[i][j].canvas.draw()
        # Copy the contents of the figure to the subplot
        axs[i, j].imshow(figures[i][j].canvas.renderer._renderer)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure as a PNG file
plt.savefig(
    f"{MODEL_DIR}/confusion_matrix/cf_proportions.png",
    format="png",
    dpi=400,
)

#         plt.savefig(
#             f"{MODEL_DIR}/confusion_matrix/{cohort}_{index}_proportions.png",
#             format="png",
#             dpi=400,
#         )

#         plt.show()
#         plt.clf()

# %%
