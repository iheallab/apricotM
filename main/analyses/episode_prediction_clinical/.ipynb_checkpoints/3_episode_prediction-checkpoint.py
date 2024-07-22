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

# models = ["sofa", "apricott", "apricotm"]
# # models = ["sofa"]

# outcomes = ["dead"]

# for model in models:

#     for outcome in outcomes:

#         cohorts = ["int", "ext", "temp", "prosp"]

#         table = pd.DataFrame()

#         for cohort in cohorts:

#             recal = pd.read_csv(
#                 f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome}_episodes.csv"
#             )

#             predictions = pd.read_csv(
#                 f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome}_step_predictions.csv"
#             )
            
#             true_label = recal["true"].values
#             probs = recal["pred"].values
            
#             true_label_step = predictions["true"].values
#             probs_step = predictions["prob"].values

#             n_iterations = 100

#             # Desired precision level
#             desired_precision = [0.20, 0.25, 0.33, 0.50, 0.75]

#             precision, recall, thresholds = precision_recall_curve(true_label, probs)

#             table_thresh = pd.DataFrame()
            
#             thresholds_prec = []

#             for prec in desired_precision:

#                 table_bootstrap = pd.DataFrame()

#                 for i in range(n_iterations):

#                     random_sample = np.random.choice(
#                         len(true_label), len(true_label), replace=True
#                     )

#                     sample_true = true_label[random_sample]
#                     sample_prob = probs[random_sample]

#                     while (sample_true == 0).all(axis=0).any():
#                         random_sample = np.random.choice(
#                             len(true_label), len(true_label), replace=True
#                         )

#                         sample_true = true_label[random_sample]
#                         sample_prob = probs[random_sample]

#                     # Find the threshold corresponding to the desired precision
#                     threshold_index = np.argmax(precision >= prec)

#                     if threshold_index == len(thresholds):

#                         threshold = thresholds[threshold_index - 1]

#                     else:

#                         threshold = thresholds[threshold_index]

#                     sample_pred = np.array(
#                         (np.array(sample_prob) >= threshold).astype("int")
#                     )

#                     adj_sens = recall_score(
#                         np.array(sample_true), np.array(sample_pred)
#                     )
#                     adj_ppv = precision_score(
#                         np.array(sample_true), np.array(sample_pred)
#                     )
#                     adj_cf = confusion_matrix(
#                         np.array(sample_true), np.array(sample_pred)
#                     )
#                     adj_spec = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[0, 1])
#                     adj_npv = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[1, 0])

#                     table_episode = pd.DataFrame(
#                         {
#                             "Sensitivity": adj_sens,
#                             "Specificity": adj_spec,
#                             "PPV": adj_ppv,
#                         },
#                         index=[cohort],
#                     )
                    
#                     random_sample = np.random.choice(
#                         len(true_label_step), len(true_label_step), replace=True
#                     )
                    
#                     pred_labels_step = (probs_step >= threshold).astype(
#                         "int"
#                     )

#                     sample_true = true_label_step[random_sample]
#                     sample_pred = pred_labels_step[random_sample]

#                     while (sample_true == 0).all(axis=0).any():
#                         random_sample = np.random.choice(
#                             len(true_label_step), len(true_label_step), replace=True
#                         )

#                         sample_true = true_label_step[random_sample]
#                         sample_pred = pred_labels_step[random_sample]

#                     adj_sens = recall_score(
#                         np.array(sample_true), np.array(sample_pred)
#                     )
#                     adj_ppv = precision_score(
#                         np.array(sample_true), np.array(sample_pred)
#                     )
#                     adj_cf = confusion_matrix(
#                         np.array(sample_true), np.array(sample_pred)
#                     )
#                     adj_spec = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[0, 1])
#                     adj_npv = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[1, 0])

#                     table_step = pd.DataFrame(
#                         {
#                             "Sensitivity (step)": adj_sens,
#                             "Specificity (step)": adj_spec,
#                             "PPV (step)": adj_ppv,
#                         },
#                         index=[cohort],
#                     )
                    
#                     table_comb = pd.concat([table_episode, table_step], axis=1)

#                     table_bootstrap = pd.concat([table_bootstrap, table_comb], axis=0)


#                 table_bootstrap = table_bootstrap.apply(
#                     lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
#                     axis=0,
#                 )

#                 table_bootstrap = pd.DataFrame(
#                     data=table_bootstrap.values.reshape((1, len(table_bootstrap))),
#                     columns=table_bootstrap.index.values,
#                     index=[cohort],
#                 )

#                 prevalence = np.unique(np.array(true_label), return_counts=True)[1][
#                     1
#                 ] / len(true_label)

#                 table_bootstrap.insert(
#                     loc=0, column="Prevalence", value=f"{prevalence:.3f}"
#                 )
                
#                 prevalence = np.unique(np.array(true_label_step), return_counts=True)[1][
#                     1
#                 ] / len(true_label_step)

#                 table_bootstrap.insert(
#                     loc=4, column="Prevalence (step)", value=f"{prevalence:.3f}"
#                 )

#                 pred_label = np.array(probs >= threshold).astype("int")

#                 pred_label = pd.Series(pred_label)
#                 sample_true = pd.Series(true_label)

#                 table_bootstrap.insert(
#                     loc=0, column="Threshold", value=f"{threshold:.3f}"
#                 )

#                 table_thresh = pd.concat([table_thresh, table_bootstrap], axis=0)
                
#                 thresholds_prec.append(threshold)

#             table_thresh.insert(loc=0, column="Precision", value=desired_precision)

#             # Calculate lead time and number of alerts

#             group_lead = []
#             group_alert = []

#             group_pred = predictions.groupby("icustay_id")

#             for thresh in thresholds_prec:

#                 predictions["pred"] = (predictions["prob"].values >= thresh).astype(
#                     "int"
#                 )

#                 alert_counts = []
#                 time_lead = []

#                 group_pred = predictions.groupby("icustay_id")

#                 for name, group in group_pred:

#                     state = group["true"].max()

#                     if state == 1:

#                         pos = group[group["true"] == 1]
#                         pos = pos["time"].min()

#                         early = group[(group["time"] <= pos)]

#                         pred_state = early["pred"].max()

#                         if len(early) > 0:

#                             if pred_state == 1:

#                                 time = early[early["pred"] == 1]
#                                 time = (pos - time["time"].min()) + time_window

#                                 time_lead.append(time)

#                                 alert_counts.append(len(early[early["pred"] == 1]))
                                
#                 if len(alert_counts) > 0:

#                     alert_counts = pd.Series(alert_counts).describe()
#                     time_lead = pd.Series(time_lead).describe()
                
#                     alert = f"{alert_counts['50%']:.2f} ({alert_counts['25%']:.2f}-{alert_counts['75%']:.2f})"
#                     lead = f"{time_lead['50%']:.2f} ({time_lead['25%']:.2f}-{time_lead['75%']:.2f})"
                    
#                 else:
                    
#                     alert = "N/A"
#                     lead = "N/A"

#                 group_alert.append(alert)
#                 group_lead.append(lead)

#             table_thresh.insert(loc=1, column="Number of alerts", value=group_alert)
#             table_thresh.insert(loc=1, column="Earliest prediction", value=group_lead)
            
#             print(table_thresh)

#             table = pd.concat([table, table_thresh], axis=0)

#         table.to_csv(f"{MODEL_DIR}/{model}/episode_prediction/{outcome}_episode_metrics.csv")

#%%

# Unstable episode prediction

# models = ["sofa", "apricott", "apricotm"]
models = ["apricotm"]


# outcomes = [["stable-unstable", "unstable-stable"],
#            ["no mv-mv", "mv-no mv"],
#            ["no vp- vp", "vp-no vp"],
#            ["no crrt-crrt", "crrt-no crrt"]]

outcomes = [["no mv-mv", "mv-no mv"],
           ["no vp- vp", "vp-no vp"],
           ["no crrt-crrt", "crrt-no crrt"]]


for model in models:

    for outcome in outcomes:

        cohorts = ["int", "ext", "prosp"]

        table = pd.DataFrame()

        for cohort in cohorts:

            recal = pd.read_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome[0]}_episodes.csv"
            )

            predictions = pd.read_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome[0]}_step_predictions.csv"
            )
            
            true_label = recal["true"].values
            probs = recal["pred"].values
            
            true_label_step = predictions["true"].values
            probs_step = predictions["prob"].values

            n_iterations = 100

            # Desired precision level
            desired_precision = [0.20, 0.25, 0.33, 0.50, 0.75]

            precision, recall, thresholds = precision_recall_curve(true_label, probs)

            table_thresh = pd.DataFrame()
            
            thresholds_prec = []

            for prec in desired_precision:

                table_bootstrap = pd.DataFrame()

                for i in range(n_iterations):

                    random_sample = np.random.choice(
                        len(true_label), len(true_label), replace=True
                    )

                    sample_true = true_label[random_sample]
                    sample_prob = probs[random_sample]

                    while (sample_true == 0).all(axis=0).any():
                        random_sample = np.random.choice(
                            len(true_label), len(true_label), replace=True
                        )

                        sample_true = true_label[random_sample]
                        sample_prob = probs[random_sample]

                    # Find the threshold corresponding to the desired precision
                    threshold_index = np.argmax(precision >= prec)

                    if threshold_index == len(thresholds):

                        threshold = thresholds[threshold_index - 1]

                    else:

                        threshold = thresholds[threshold_index]

                    sample_pred = np.array(
                        (np.array(sample_prob) >= threshold).astype("int")
                    )

                    adj_sens = recall_score(
                        np.array(sample_true), np.array(sample_pred)
                    )
                    adj_ppv = precision_score(
                        np.array(sample_true), np.array(sample_pred)
                    )
                    adj_cf = confusion_matrix(
                        np.array(sample_true), np.array(sample_pred)
                    )
                    adj_spec = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[0, 1])
                    adj_npv = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[1, 0])

                    table_episode = pd.DataFrame(
                        {
                            "Sensitivity": adj_sens,
                            "Specificity": adj_spec,
                            "PPV": adj_ppv,
                        },
                        index=[cohort],
                    )
                    
                    random_sample = np.random.choice(
                        len(true_label_step), len(true_label_step), replace=True
                    )
                    
                    pred_labels_step = (probs_step >= threshold).astype(
                        "int"
                    )

                    sample_true = true_label_step[random_sample]
                    sample_pred = pred_labels_step[random_sample]

                    while (sample_true == 0).all(axis=0).any():
                        random_sample = np.random.choice(
                            len(true_label_step), len(true_label_step), replace=True
                        )

                        sample_true = true_label_step[random_sample]
                        sample_pred = pred_labels_step[random_sample]

                    adj_sens = recall_score(
                        np.array(sample_true), np.array(sample_pred)
                    )
                    adj_ppv = precision_score(
                        np.array(sample_true), np.array(sample_pred)
                    )
                    adj_cf = confusion_matrix(
                        np.array(sample_true), np.array(sample_pred)
                    )
                    adj_spec = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[0, 1])
                    adj_npv = adj_cf[0, 0] / (adj_cf[0, 0] + adj_cf[1, 0])

                    table_step = pd.DataFrame(
                        {
                            "Sensitivity (step)": adj_sens,
                            "Specificity (step)": adj_spec,
                            "PPV (step)": adj_ppv,
                        },
                        index=[cohort],
                    )
                    
                    table_comb = pd.concat([table_episode, table_step], axis=1)

                    table_bootstrap = pd.concat([table_bootstrap, table_comb], axis=0)


                table_bootstrap = table_bootstrap.apply(
                    lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                    axis=0,
                )

                table_bootstrap = pd.DataFrame(
                    data=table_bootstrap.values.reshape((1, len(table_bootstrap))),
                    columns=table_bootstrap.index.values,
                    index=[cohort],
                )

                prevalence = np.unique(np.array(true_label), return_counts=True)[1][
                    1
                ] / len(true_label)

                table_bootstrap.insert(
                    loc=0, column="Prevalence", value=f"{prevalence:.3f}"
                )
                
                prevalence = np.unique(np.array(true_label_step), return_counts=True)[1][
                    1
                ] / len(true_label_step)

                table_bootstrap.insert(
                    loc=4, column="Prevalence (step)", value=f"{prevalence:.3f}"
                )

                pred_label = np.array(probs >= threshold).astype("int")

                pred_label = pd.Series(pred_label)
                sample_true = pd.Series(true_label)

                table_bootstrap.insert(
                    loc=0, column="Threshold", value=f"{threshold:.3f}"
                )

                table_thresh = pd.concat([table_thresh, table_bootstrap], axis=0)
                
                thresholds_prec.append(threshold)

            table_thresh.insert(loc=0, column="Precision", value=desired_precision)

            # Calculate lead time and number of alerts

            group_lead = []
            group_alert = []

            group_pred = predictions.groupby("icustay_id")

            for thresh in thresholds_prec:

                predictions["pred"] = (predictions["prob"].values >= thresh).astype(
                    "int"
                )

                alert_counts = []
                time_lead = []

                group_pred = predictions.groupby("icustay_id")

                for name, group in group_pred:

                    state = group["true"].max()

                    if state == 1:

                        pos = group[group["true"] == 1]
                        pos = pos["time"].min()

                        early = group[(group["time"] <= pos)]

                        pred_state = early["pred"].max()

                        if len(early) > 0:

                            if pred_state == 1:

                                time = early[early["pred"] == 1]
                                time = (pos - time["time"].min()) + time_window

                                time_lead.append(time)

                                alert_counts.append(len(early[early["pred"] == 1]))
                                
                if len(alert_counts) > 0:

                    alert_counts = pd.Series(alert_counts).describe()
                    time_lead = pd.Series(time_lead).describe()
                
                    alert = f"{alert_counts['50%']:.2f} ({alert_counts['25%']:.2f}-{alert_counts['75%']:.2f})"
                    lead = f"{time_lead['50%']:.2f} ({time_lead['25%']:.2f}-{time_lead['75%']:.2f})"
                    
                else:
                    
                    alert = "N/A"
                    lead = "N/A"

                group_alert.append(alert)
                group_lead.append(lead)

            table_thresh.insert(loc=1, column="Number of alerts", value=group_alert)
            table_thresh.insert(loc=1, column="Earliest prediction", value=group_lead)
            
            print(table_thresh)

            table = pd.concat([table, table_thresh], axis=0)

        table.to_csv(f"{MODEL_DIR}/{model}/episode_prediction/{outcome[0]}_episode_metrics.csv")


# %%
