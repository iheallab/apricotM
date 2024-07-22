#%%

# Import libraries

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from variables import OUTPUT_DIR, MODEL_DIR

if not os.path.exists(f"{MODEL_DIR}/subgroup"):
    os.makedirs(f"{MODEL_DIR}/subgroup")


#%%

# Load static data

static = pd.read_csv(f"{OUTPUT_DIR}/final_data/static.csv")

with open("%s/model/scalers_static.pkl" % OUTPUT_DIR, "rb") as f:
    scalers = pickle.load(f)


# %%

# Inverse transform static data

static.iloc[:, 1:] = scalers["scaler_static"].inverse_transform(static.iloc[:, 1:])

static["sex"] = scalers["scaler_gender"].inverse_transform(static["sex"].astype(int))
static["race"] = scalers["scaler_race"].inverse_transform(static["race"].astype(int))

static.loc[static["age"] < 60, "age_group"] = "Young"
static.loc[static["age"] >= 60, "age_group"] = "Old"


#%%

cohorts = ["int", "ext", "temp"]
# cohorts = ["prosp"]

for cohort in cohorts:

    # Load model results

    int_model_probs = pd.read_csv(f"{MODEL_DIR}/results/{cohort}_pred_labels.csv")
    int_model_true = pd.read_csv(f"{MODEL_DIR}/results/{cohort}_true_labels.csv")

    bias_analysis = {
        "sex": ["Male", "Female"],
        "race": ["white", "black", "other"],
        "age_group": ["Young", "Old"],
    }

    for key in bias_analysis.keys():

        for value in bias_analysis[key]:

            gender = static.loc[static[key] == value, "icustay_id"].tolist()

            n_iterations = 100

            auroc_int = pd.DataFrame(data=[], columns=int_model_true.columns[1:])
            auprc_int = pd.DataFrame(data=[], columns=int_model_true.columns[1:])
            sens_int = pd.DataFrame(data=[], columns=int_model_true.columns[1:])
            spec_int = pd.DataFrame(data=[], columns=int_model_true.columns[1:])
            ppv_int = pd.DataFrame(data=[], columns=int_model_true.columns[1:])
            npv_int = pd.DataFrame(data=[], columns=int_model_true.columns[1:])

            for i in range(n_iterations):

                random_int = np.random.choice(
                    len(int_model_true), len(int_model_true), replace=True
                )

                sample_int_true = (
                    int_model_true.iloc[random_int]
                    .loc[int_model_true["icustay_id"].isin(gender)]
                    .iloc[:, 1:]
                )
                sample_int_pred = (
                    int_model_probs.iloc[random_int]
                    .loc[int_model_probs["icustay_id"].isin(gender)]
                    .iloc[:, 1:]
                )

                while (sample_int_true == 0).all(axis=0).any():
                    random_sample = np.random.choice(
                        len(int_model_true), len(int_model_true), replace=True
                    )

                    sample_int_true = int_model_true.iloc[random_int].loc[
                        int_model_true["icustay_id"].isin(gender)
                    ]
                    sample_int_pred = int_model_probs.iloc[random_int].loc[
                        int_model_probs["icustay_id"].isin(gender)
                    ]

                int_ind_auroc = []
                int_ind_auprc = []
                int_ind_sens = []
                int_ind_spec = []
                int_ind_ppv = []
                int_ind_npv = []

                for column in int_model_true.columns[1:]:

                    score_int = roc_auc_score(
                        sample_int_true.loc[:, column], sample_int_pred.loc[:, column]
                    )
                    int_ind_auroc.append(score_int)

                    fpr, tpr, thresholds = roc_curve(
                        sample_int_true.loc[:, column], sample_int_pred.loc[:, column]
                    )
                    J = tpr - fpr
                    ix = np.argmax(J)
                    best_thresh = thresholds[ix]

                    y_pred_class = (
                        sample_int_pred.loc[:, column] >= best_thresh
                    ).astype("int")
                    cf_class = confusion_matrix(
                        sample_int_true.loc[:, column], y_pred_class
                    )
                    spec = cf_class[0, 0] / (cf_class[0, 0] + cf_class[0, 1])
                    npv = cf_class[0, 0] / (cf_class[0, 0] + cf_class[1, 0])
                    ppv = cf_class[1, 1] / (cf_class[1, 1] + cf_class[0, 1])
                    sens = cf_class[1, 1] / (cf_class[1, 1] + cf_class[1, 0])

                    int_ind_sens.append(sens)
                    int_ind_spec.append(spec)
                    int_ind_ppv.append(ppv)
                    int_ind_npv.append(npv)

                    prec, rec, _ = precision_recall_curve(
                        sample_int_true.loc[:, column], sample_int_pred.loc[:, column]
                    )
                    score_int = auc(rec, prec)
                    int_ind_auprc.append(score_int)

                int_ind_auroc = np.array(int_ind_auroc).reshape(1, -1)
                int_ind_auprc = np.array(int_ind_auprc).reshape(1, -1)
                int_ind_sens = np.array(int_ind_sens).reshape(1, -1)
                int_ind_spec = np.array(int_ind_spec).reshape(1, -1)
                int_ind_ppv = np.array(int_ind_ppv).reshape(1, -1)
                int_ind_npv = np.array(int_ind_npv).reshape(1, -1)

                auroc_int = pd.concat(
                    [
                        auroc_int,
                        pd.DataFrame(
                            data=int_ind_auroc, columns=int_model_true.columns[1:]
                        ),
                    ],
                    axis=0,
                )

                auprc_int = pd.concat(
                    [
                        auprc_int,
                        pd.DataFrame(
                            data=int_ind_auprc, columns=int_model_true.columns[1:]
                        ),
                    ],
                    axis=0,
                )

                sens_int = pd.concat(
                    [
                        sens_int,
                        pd.DataFrame(
                            data=int_ind_sens, columns=int_model_true.columns[1:]
                        ),
                    ],
                    axis=0,
                )

                spec_int = pd.concat(
                    [
                        spec_int,
                        pd.DataFrame(
                            data=int_ind_spec, columns=int_model_true.columns[1:]
                        ),
                    ],
                    axis=0,
                )

                ppv_int = pd.concat(
                    [
                        ppv_int,
                        pd.DataFrame(
                            data=int_ind_ppv, columns=int_model_true.columns[1:]
                        ),
                    ],
                    axis=0,
                )

                npv_int = pd.concat(
                    [
                        npv_int,
                        pd.DataFrame(
                            data=int_ind_npv, columns=int_model_true.columns[1:]
                        ),
                    ],
                    axis=0,
                )

                print(f"Iteration {i+1}")

            metrics = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV"]

            # Validation metrics

            auroc_int_sum = auroc_int.apply(
                lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )
            auprc_int_sum = auprc_int.apply(
                lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )
            sens_int_sum = sens_int.apply(
                lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )
            spec_int_sum = spec_int.apply(
                lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )
            ppv_int_sum = ppv_int.apply(
                lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )
            npv_int_sum = npv_int.apply(
                lambda x: f"{x.mean():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                axis=0,
            )

            int_metrics = pd.concat(
                [
                    auroc_int_sum,
                    auprc_int_sum,
                    sens_int_sum,
                    spec_int_sum,
                    ppv_int_sum,
                    npv_int_sum,
                ],
                axis=1,
            )

            int_metrics.columns = metrics

            int_metrics.to_csv(
                f"{MODEL_DIR}/subgroup/{value}_{cohort}_metrics.csv",
                index=None,
            )

            auroc_int.to_csv(
                f"{MODEL_DIR}/subgroup/{value}_{cohort}_auroc.csv",
                index=None,
            )

# %%
