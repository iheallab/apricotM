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

# Mortality episode prediction

# models = ["sofa", "apricott", "apricotm"]
models = ["apricotm"]

outcomes = ["dead"]

for model in models:

    for outcome in outcomes:

        cohorts = ["int", "ext", "temp", "prosp"]

        table = pd.DataFrame()

        for cohort in cohorts:

            recal = pd.read_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome}_episodes.csv"
            )

            true_label = recal["true"].values
            probs = recal["pred"].values

            n_iterations = 100

            # Calculate episode metrics
            table_episode = pd.DataFrame()

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

                
                adj_auroc = roc_auc_score(sample_true, sample_prob)
                prec, rec, _ = precision_recall_curve(sample_true, sample_prob)
                
                adj_auprc = auc(rec, prec)


                table_ind = pd.DataFrame(
                    {
                        "AUROC": adj_auroc,
                        "AUPRC": adj_auprc,
                    },
                    index=[cohort],
                )

                table_episode = pd.concat([table_episode, table_ind], axis=0)

            table_episode = table_episode.apply(
                lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )

            table_episode = pd.DataFrame(
                data=table_episode.values.reshape((1, len(table_episode))),
                columns=table_episode.index.values,
                index=[cohort],
            )

            prevalence = np.unique(np.array(true_label), return_counts=True)[1][
                1
            ] / len(true_label)

            table_episode.insert(
                loc=0, column="Prevalence", value=f"{prevalence:.3f}"
            )

            # Calculate step metrics

            predictions = pd.read_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome}_step_predictions.csv"
            )
                
            true_label = predictions["true"].values
            pred_labels = predictions["prob"].values

            table_step = pd.DataFrame()

            for i in range(n_iterations):

                random_sample = np.random.choice(
                    len(true_label), len(true_label), replace=True
                )

                sample_true = true_label[random_sample]
                sample_pred = pred_labels[random_sample]

                while (sample_true == 0).all(axis=0).any():
                    random_sample = np.random.choice(
                        len(true_label), len(true_label), replace=True
                    )

                    sample_true = true_label[random_sample]
                    sample_pred = pred_labels[random_sample]

                adj_auroc = roc_auc_score(sample_true, sample_pred)
                prec, rec, _ = precision_recall_curve(sample_true, sample_pred)
                
                adj_auprc = auc(rec, prec)


                table_ind = pd.DataFrame(
                    {
                        "AUROC (step)": adj_auroc,
                        "AUPRC (step)": adj_auprc,
                    },
                    index=[cohort],
                )
                
                table_step = pd.concat([table_step, table_ind], axis=0)

            table_step = table_step.apply(
                lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )

            table_step = pd.DataFrame(
                data=table_step.values.reshape((1, len(table_step))),
                columns=table_step.index.values,
                index=[cohort],
            )
            
            prevalence = np.unique(np.array(true_label), return_counts=True)[1][
                1
            ] / len(true_label)

            table_step.insert(
                loc=0, column="Prevalence (step)", value=f"{prevalence:.3f}"
            )

            table_final = pd.concat([table_episode, table_step], axis=1)
            
            print(table_final)

            table = pd.concat([table, table_final], axis=0)

        table.to_csv(f"{MODEL_DIR}/{model}/episode_prediction/{outcome}_auroc_auprc.csv")

# # #%%

# Unstable episode prediction

# models = ["sofa_criteria", "sofa", "apricott", "apricotm"]
models = ["apricotm"]

outcomes = [["stable-unstable", "unstable-stable"],
           ["no mv-mv", "mv-no mv"],
           ["no vp- vp", "vp-no vp"],
           ["no crrt-crrt", "crrt-no crrt"]]

# outcomes = [["no mv-mv", "mv-no mv"],
#            ["no vp- vp", "vp-no vp"],
#            ["no crrt-crrt", "crrt-no crrt"]]

for model in models:

    for outcome in outcomes:

        cohorts = ["int", "ext", "temp", "prosp"]

        table = pd.DataFrame()

        for cohort in cohorts:

            recal = pd.read_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome[0]}_episodes.csv"
            )

            true_label = recal["true"].values
            probs = recal["pred"].values

            n_iterations = 100

            # Calculate episode metrics
            table_episode = pd.DataFrame()

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

                
                adj_auroc = roc_auc_score(sample_true, sample_prob)
                prec, rec, _ = precision_recall_curve(sample_true, sample_prob)
                
                adj_auprc = auc(rec, prec)


                table_ind = pd.DataFrame(
                    {
                        "AUROC": adj_auroc,
                        "AUPRC": adj_auprc,
                    },
                    index=[cohort],
                )

                table_episode = pd.concat([table_episode, table_ind], axis=0)

            table_episode = table_episode.apply(
                lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )

            table_episode = pd.DataFrame(
                data=table_episode.values.reshape((1, len(table_episode))),
                columns=table_episode.index.values,
                index=[cohort],
            )

            prevalence = np.unique(np.array(true_label), return_counts=True)[1][
                1
            ] / len(true_label)

            table_episode.insert(
                loc=0, column="Prevalence", value=f"{prevalence:.3f}"
            )

            # Calculate step metrics

            predictions = pd.read_csv(
                f"{MODEL_DIR}/{model}/episode_prediction/{cohort}_{outcome[0]}_step_predictions.csv"
            )
                
            true_label = predictions["true"].values
            pred_labels = predictions["prob"].values

            table_step = pd.DataFrame()

            for i in range(n_iterations):

                random_sample = np.random.choice(
                    len(true_label), len(true_label), replace=True
                )

                sample_true = true_label[random_sample]
                sample_pred = pred_labels[random_sample]

                while (sample_true == 0).all(axis=0).any():
                    random_sample = np.random.choice(
                        len(true_label), len(true_label), replace=True
                    )

                    sample_true = true_label[random_sample]
                    sample_pred = pred_labels[random_sample]

                adj_auroc = roc_auc_score(sample_true, sample_pred)
                prec, rec, _ = precision_recall_curve(sample_true, sample_pred)
                
                adj_auprc = auc(rec, prec)


                table_ind = pd.DataFrame(
                    {
                        "AUROC (step)": adj_auroc,
                        "AUPRC (step)": adj_auprc,
                    },
                    index=[cohort],
                )
                
                table_step = pd.concat([table_step, table_ind], axis=0)

            table_step = table_step.apply(
                lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )

            table_step = pd.DataFrame(
                data=table_step.values.reshape((1, len(table_step))),
                columns=table_step.index.values,
                index=[cohort],
            )
            
            prevalence = np.unique(np.array(true_label), return_counts=True)[1][
                1
            ] / len(true_label)

            table_step.insert(
                loc=0, column="Prevalence (step)", value=f"{prevalence:.3f}"
            )

            table_final = pd.concat([table_episode, table_step], axis=1)
            
            print(table_final)

            table = pd.concat([table, table_final], axis=0)

        table.to_csv(
            f"{MODEL_DIR}/{model}/episode_prediction/{outcome[0]}_auroc_auprc.csv"
        )
