#%%

# Import libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from variables import MODEL_DIR

import os

if not os.path.exists(f"{MODEL_DIR}/transition_diagram"):
    os.makedirs(f"{MODEL_DIR}/transition_diagram")

#%%

cohorts = ["int", "ext", "prosp"]

for cohort in cohorts:

    y_true = pd.read_csv(f"{MODEL_DIR}/results/{cohort}_true_labels.csv")

    y_ids = y_true["icustay_id"].values

    y_true["interval"] = y_true.groupby("icustay_id").cumcount()
    y_true["hours"] = y_true["interval"] * 4

    y_pred = pd.read_csv(f"{MODEL_DIR}/results/{cohort}_pred_labels.csv")

    y_pred["interval"] = y_pred.groupby("icustay_id").cumcount()
    y_pred["hours"] = y_pred["interval"] * 4

    y_true_hours = y_true["hours"].values
    y_pred_hours = y_pred["hours"].values

    y_true.drop(labels=["icustay_id", "interval", "hours"], inplace=True, axis=1)
    y_pred.drop(labels=["icustay_id", "interval", "hours"], inplace=True, axis=1)

    labels = y_true.columns.tolist()

    thresh = []

    for label in labels:

        fpr, tpr, thresholds = roc_curve(
            y_true.loc[:, label].values, y_pred.loc[:, label].values
        )
        roc_auc_class = auc(fpr, tpr)
        print(roc_auc_class)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        print("Best Threshold=%f" % (best_thresh))
        thresh.append(best_thresh)

    y_pred_class = y_pred.copy()
    for i in range(len(thresh)):
        y_pred_class.iloc[:, i] = (y_pred.iloc[:, i] >= thresh[i]).astype("int")

    y_true_label = y_true[["discharge", "stable", "unstable", "dead"]].idxmax(axis=1)

    true_labels = pd.DataFrame(
        {"hours": y_true_hours, "final_state": y_true_label, "icustay_id": y_ids}
    )

    transition_diagram = true_labels.copy()

    new_val = (transition_diagram["hours"] / 24).apply(lambda x: math.floor(x))

    transition_diagram.insert(loc=0, column="day", value=new_val)

    # Define admission status priority
    status_priority = {"stable": 0, "unstable": 1, "discharge": 2, "dead": 3}

    # Add a priority column based on the admission status
    transition_diagram["priority"] = transition_diagram["final_state"].map(
        status_priority
    )

    # Sort the DataFrame by date and priority
    transition_diagram = transition_diagram.sort_values(
        by=["icustay_id", "day", "priority"]
    )

    transition_diagram.drop_duplicates(
        subset=["day", "icustay_id"], keep="last", inplace=True
    )

    transition_diagram = transition_diagram.loc[:, ["icustay_id", "day", "final_state"]]

    grouped = transition_diagram.groupby("day")

    icustay_ids = transition_diagram["icustay_id"].unique()

    transition_diagram = pd.DataFrame(
        columns=["day", "Stable", "Unstable", "Deceased", "Discharge"]
    )

    count_disch = 0
    count_dead = 0
    for _, group in grouped:
        counts = group["final_state"].value_counts()
        if "dead" in group["final_state"].tolist():
            prop_dead = counts["dead"]
            # prop_dead = (counts['dead'] + count_dead)
            # count_dead += counts['dead']
        else:
            prop_dead = 0
        if "unstable" in group["final_state"].tolist():
            prop_unstable = counts["unstable"]
        else:
            prop_unstable = 0
        if "stable" in group["final_state"].tolist():
            prop_stable = counts["stable"]
        else:
            prop_stable = 0
        if "discharge" in group["final_state"].tolist():
            prop_disch = counts["discharge"]
            # prop_disch = (counts['discharge'] + count_disch)
            # count_disch += counts['discharge']
        else:
            prop_disch = 0

        # prop_disch = len(icustay_ids) - prop_stable - prop_dead - prop_unstable

        proportions = pd.DataFrame(
            data={
                "day": group["day"].values[0],
                "Stable": prop_stable,
                "Unstable": prop_unstable,
                "Deceased": prop_dead,
                "Discharge": prop_disch,
            },
            index=[group["day"].values[0]],
        )

        transition_diagram = pd.concat([transition_diagram, proportions], axis=0)

    transition_diagram = transition_diagram.iloc[:15]

    # Set the x-values (time points)
    x = transition_diagram["day"].values.astype(int)

    # Define RGB values for custom colors
    color_rgb1 = (12 / 255, 53 / 255, 106 / 255)
    color_rgb2 = (39 / 255, 158 / 255, 255 / 255)
    color_rgb3 = (64 / 255, 248 / 255, 255 / 255)
    color_rgb4 = (213 / 255, 108 / 255, 255 / 255)
    color_rgb5 = (253 / 255, 226 / 255, 255 / 255)

    rgb = [color_rgb1, color_rgb2, color_rgb3, color_rgb4, color_rgb5]

    import matplotlib.colors as mcolors

    status_labels = ["Stable", "Unstable", "Deceased", "Discharge"]
    bar_positions = range(1, len(x) + 1)

    bottom = [0] * len(x)
    count = 0
    for status in status_labels:
        proportions = transition_diagram.loc[:, status]
        plt.bar(
            bar_positions, proportions, bottom=bottom, label=status, color=rgb[count]
        )
        bottom = [b + p for b, p in zip(bottom, proportions)]
        count += 1

    # Define a range of gray colors
    num_colors = 5
    gray_colors = mcolors.LinearSegmentedColormap.from_list(
        "gray_colors",
        [(x / num_colors, x / num_colors, x / num_colors) for x in range(num_colors)],
    )

    # plt.figure(figsize=(16, 8))  # Adjust the size as needed

    # Create a stacked area chart
    # plt.stackplot(x, y_stable, y_unstable, y_dead, y_disch, y_transfer, colors=rgb, labels=['Stable', 'Unstable', 'Deceased', 'Discharged', 'Transferred'], alpha=0.7)

    # Adding labels and title
    plt.xlabel("Days in ICU", fontsize=16)
    plt.ylabel("Number of ICU stays", fontsize=16)
    plt.xlim((0, 16))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=(0.5, 1), fontsize=16)
    # plt.ylim((0, 1))

    plt.savefig(
        f"{MODEL_DIR}/transition_diagram/{cohort}_true_transitions.png",
        format="png",
        dpi=400,
        bbox_inches="tight",
    )

    # Show the plot
    plt.show()
    plt.clf()

    condition_deceased = y_pred_class["dead"] == 1
    condition_unstable = (
        (y_pred_class["no mv-mv"] == 1)
        | (y_pred_class["no vp- vp"] == 1)
        | (y_pred_class["no crrt-crrt"] == 1)
        | (y_pred_class["unstable"] == 1)
        | (y_pred_class["stable-unstable"] == 1)
    )
    condition_discharge = (
        (y_pred_class["discharge"] == 1)
        & (y_pred_class["no mv-mv"] == 0)
        & (y_pred_class["no vp- vp"] == 0)
        & (y_pred_class["no crrt-crrt"] == 0)
    )
    condition_stable = (
        (y_pred_class["stable"] == 1) | (y_pred_class["unstable-stable"] == 1)
    ) & (
        (y_pred_class["no mv-mv"] == 0)
        & (y_pred_class["no vp- vp"] == 0)
        & (y_pred_class["no crrt-crrt"] == 0)
    )

    y_pred_class["predicted_state"] = "stable"
    y_pred_class.loc[condition_stable, "predicted_state"] = "stable"
    y_pred_class.loc[condition_discharge, "predicted_state"] = "discharge"
    y_pred_class.loc[condition_unstable, "predicted_state"] = "unstable"
    y_pred_class.loc[condition_deceased, "predicted_state"] = "dead"

    y_pred_label = y_pred_class["predicted_state"]

    pred_labels = pd.DataFrame(
        {"hours": y_pred_hours, "final_state": y_pred_label, "icustay_id": y_ids}
    )

    transition_diagram = pred_labels.copy()

    new_val = (transition_diagram["hours"] / 24).apply(lambda x: math.floor(x))

    transition_diagram.insert(loc=0, column="day", value=new_val)

    # Define admission status priority
    status_priority = {"stable": 0, "unstable": 2, "discharge": 1, "dead": 3}

    # Add a priority column based on the admission status
    transition_diagram["priority"] = transition_diagram["final_state"].map(
        status_priority
    )

    # Sort the DataFrame by date and priority
    transition_diagram = transition_diagram.sort_values(
        by=["icustay_id", "day", "priority"]
    )

    transition_diagram.drop_duplicates(
        subset=["day", "icustay_id"], keep="last", inplace=True
    )

    transition_diagram = transition_diagram.loc[:, ["icustay_id", "day", "final_state"]]

    grouped = transition_diagram.groupby("day")

    icustay_ids = transition_diagram["icustay_id"].unique()

    transition_diagram = pd.DataFrame(
        columns=["day", "Stable", "Unstable", "Deceased", "Discharge"]
    )

    count_disch = 0
    count_dead = 0
    for _, group in grouped:
        counts = group["final_state"].value_counts()
        if "dead" in group["final_state"].tolist():
            prop_dead = counts["dead"]
            # prop_dead = (counts['dead'] + count_dead)
            # count_dead += counts['dead']
        else:
            prop_dead = 0
        if "unstable" in group["final_state"].tolist():
            prop_unstable = counts["unstable"]
        else:
            prop_unstable = 0
        if "stable" in group["final_state"].tolist():
            prop_stable = counts["stable"]
        else:
            prop_stable = 0
        if "discharge" in group["final_state"].tolist():
            prop_disch = counts["discharge"]
            # prop_disch = (counts['discharge'] + count_disch)
            # count_disch += counts['discharge']
        else:
            prop_disch = 0

        # prop_disch = 1 - prop_stable - prop_dead - prop_unstable

        proportions = pd.DataFrame(
            data={
                "day": group["day"].values[0],
                "Stable": prop_stable,
                "Unstable": prop_unstable,
                "Deceased": prop_dead,
                "Discharge": prop_disch,
            },
            index=[group["day"].values[0]],
        )

        transition_diagram = pd.concat([transition_diagram, proportions], axis=0)

    transition_diagram = transition_diagram.iloc[:15]

    # Set the x-values (time points)
    x = transition_diagram["day"].values.astype(int)

    # Define RGB values for custom colors
    color_rgb1 = (12 / 255, 53 / 255, 106 / 255)
    color_rgb2 = (39 / 255, 158 / 255, 255 / 255)
    color_rgb3 = (64 / 255, 248 / 255, 255 / 255)
    color_rgb4 = (213 / 255, 108 / 255, 255 / 255)
    color_rgb5 = (253 / 255, 226 / 255, 255 / 255)

    rgb = [color_rgb1, color_rgb2, color_rgb3, color_rgb4, color_rgb5]

    import matplotlib.colors as mcolors

    status_labels = ["Stable", "Unstable", "Deceased", "Discharge"]
    bar_positions = range(1, len(x) + 1)

    bottom = [0] * len(x)
    count = 0
    for status in status_labels:
        proportions = transition_diagram.loc[:, status]
        plt.bar(
            bar_positions, proportions, bottom=bottom, label=status, color=rgb[count]
        )
        bottom = [b + p for b, p in zip(bottom, proportions)]
        count += 1

    # Define a range of gray colors
    num_colors = 5
    gray_colors = mcolors.LinearSegmentedColormap.from_list(
        "gray_colors",
        [(x / num_colors, x / num_colors, x / num_colors) for x in range(num_colors)],
    )

    # plt.figure(figsize=(16, 8))  # Adjust the size as needed

    # Create a stacked area chart
    # plt.stackplot(x, y_stable, y_unstable, y_dead, y_disch, y_transfer, colors=rgb, labels=['Stable', 'Unstable', 'Deceased', 'Discharged', 'Transferred'], alpha=0.7)

    # Adding labels and title
    plt.xlabel("Days in ICU", fontsize=16)
    plt.ylabel("Number of ICU stays", fontsize=16)
    plt.xlim((0, 16))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=(0.5, 1), fontsize=16)
    # plt.ylim((0, 1))

    plt.savefig(
        f"{MODEL_DIR}/transition_diagram/{cohort}_pred_transitions.png",
        format="png",
        dpi=400,
        bbox_inches="tight",
    )

    # Show the plot
    plt.show()
    plt.clf()

# %%
