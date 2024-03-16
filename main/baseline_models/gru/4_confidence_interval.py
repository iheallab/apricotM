#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    auc,
)
import os

from variables import time_window, MODEL_DIR

#%%

groups = [
    ["validation", "int"],
    ["external_test", "ext"],
    ["temporal_test", "temp"],
    ["prospective", "prosp"],
]

for group in groups:

    model_true = pd.read_csv(f"{MODEL_DIR}/results/{group[1]}_true_labels.csv")

    model_probs = pd.read_csv(f"{MODEL_DIR}/results/{group[1]}_pred_labels.csv")

    cols = model_probs.columns.tolist()

    n_iterations = 100

    auroc = pd.DataFrame(data=[], columns=model_true.columns)
    auprc = pd.DataFrame(data=[], columns=model_true.columns)
    sens = pd.DataFrame(data=[], columns=model_true.columns)
    spec = pd.DataFrame(data=[], columns=model_true.columns)
    ppv = pd.DataFrame(data=[], columns=model_true.columns)
    npv = pd.DataFrame(data=[], columns=model_true.columns)

    for i in range(n_iterations):

        random = np.random.choice(len(model_true), len(model_true), replace=True)

        sample_true = model_true.iloc[random]
        sample_pred = model_probs.iloc[random]

        while (sample_true == 0).all(axis=0).any():
            random_sample = np.random.choice(
                len(model_true), len(model_true), replace=True
            )

            sample_true = model_true.iloc[random_sample]
            sample_pred = model_probs.iloc[random_sample]

        ind_auroc = []
        ind_auprc = []
        ind_sens = []
        ind_spec = []
        ind_ppv = []
        ind_npv = []

        for column in model_true.columns:
            score = roc_auc_score(
                sample_true.loc[:, column], sample_pred.loc[:, column]
            )
            ind_auroc.append(score)

            fpr, tpr, thresholds = roc_curve(
                sample_true.loc[:, column], sample_pred.loc[:, column]
            )
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]

            y_pred_class = (sample_pred.loc[:, column] >= best_thresh).astype("int")
            cf_class = confusion_matrix(sample_true.loc[:, column], y_pred_class)
            spec_ind = cf_class[0, 0] / (cf_class[0, 0] + cf_class[0, 1])
            npv_ind = cf_class[0, 0] / (cf_class[0, 0] + cf_class[1, 0])
            ppv_ind = cf_class[1, 1] / (cf_class[1, 1] + cf_class[0, 1])
            sens_ind = cf_class[1, 1] / (cf_class[1, 1] + cf_class[1, 0])

            ind_sens.append(sens_ind)
            ind_spec.append(spec_ind)
            ind_ppv.append(ppv_ind)
            ind_npv.append(npv_ind)

            prec, rec, _ = precision_recall_curve(
                sample_true.loc[:, column], sample_pred.loc[:, column]
            )
            score = auc(rec, prec)
            ind_auprc.append(score)

        ind_auroc = np.array(ind_auroc).reshape(1, -1)
        ind_auprc = np.array(ind_auprc).reshape(1, -1)
        ind_sens = np.array(ind_sens).reshape(1, -1)
        ind_spec = np.array(ind_spec).reshape(1, -1)
        ind_ppv = np.array(ind_ppv).reshape(1, -1)
        ind_npv = np.array(ind_npv).reshape(1, -1)

        auroc = pd.concat(
            [auroc, pd.DataFrame(data=ind_auroc, columns=model_true.columns)], axis=0
        )
        auprc = pd.concat(
            [auprc, pd.DataFrame(data=ind_auprc, columns=model_true.columns)], axis=0
        )
        sens = pd.concat(
            [sens, pd.DataFrame(data=ind_sens, columns=model_true.columns)], axis=0
        )
        spec = pd.concat(
            [spec, pd.DataFrame(data=ind_spec, columns=model_true.columns)], axis=0
        )
        ppv = pd.concat(
            [ppv, pd.DataFrame(data=ind_ppv, columns=model_true.columns)], axis=0
        )
        npv = pd.concat(
            [npv, pd.DataFrame(data=ind_npv, columns=model_true.columns)], axis=0
        )

        print(f"Iteration {i+1}")

    metrics = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV"]

    # Calculate Metrics

    auroc_sum = auroc.apply(
        lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
        axis=0,
    )
    auprc_sum = auprc.apply(
        lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
        axis=0,
    )
    sens_sum = sens.apply(
        lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
        axis=0,
    )
    spec_sum = spec.apply(
        lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
        axis=0,
    )
    ppv_sum = ppv.apply(
        lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
        axis=0,
    )
    npv_sum = npv.apply(
        lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
        axis=0,
    )

    metrics_df = pd.concat(
        [auroc_sum, auprc_sum, sens_sum, spec_sum, ppv_sum, npv_sum], axis=1
    )

    metrics_df.columns = metrics

    metrics_df.to_csv(f"{MODEL_DIR}/results/{group[1]}_metrics.csv", index=None)
