#%% Import libraries
import pandas as pd
import numpy as np
import h5py
import time
import os
import pickle
import torch

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

from variables import MODEL_DIR, OUTPUT_DIR, ANALYSIS_DIR, PROSP_DATA_DIR
from apricotm import ApricotM

#%% Create results directory if needed
if not os.path.exists(f"{MODEL_DIR}/results_pe_ablation"):
    os.makedirs(f"{MODEL_DIR}/results_pe_ablation")

#%% Load model architecture and best hyperparameters
model_architecture = torch.load(f"{MODEL_DIR}/apricotm_architecture.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(f"{MODEL_DIR}/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

seq_len = best_params["seq_len"]
batch_size = best_params["batch_size"]

print(f"Best Parameters: {best_params}")

#%% Define cohorts
cohorts = [["validation", "int"], ["external_test", "ext"], ["temporal_test", "temp"]]
prospective_cohort = ["prospective", "prosp"]

#%% Function to load cohort data
def load_cohort_data(cohort_name, prospective=False):
    if prospective:
        file_path = f"{PROSP_DATA_DIR}/dataset.h5"
    else:
        file_path = f"{OUTPUT_DIR}/final_data/dataset.h5"
    
    with h5py.File(file_path, "r") as f:
        data = f[cohort_name]
        X = data["X"][:]
        static = data["static"][:]
        y_trans = data["y_trans"][:]
        y_main = data["y_main"][:]
    
    icustay_ids = X[:, 0, 3].reshape(-1, 1)
    if not prospective:
        static = static[:, 1:]
    y = np.concatenate([y_main, y_trans], axis=1)
    
    return X, static, y, icustay_ids

#%% Function to evaluate model
def evaluate_model(pe_condition):
    # Initialize model
    model = ApricotM(
        d_model=model_architecture["d_model"],
        d_hidden=model_architecture["d_hidden"],
        d_input=model_architecture["d_input"],
        d_static=model_architecture["d_static"],
        max_code=model_architecture["max_code"],
        n_layer=model_architecture["n_layer"],
        device=DEVICE,
        dropout=model_architecture["dropout"],
        pe=pe_condition,
    ).to(DEVICE)

    # Load correct weights depending on PE
    if pe_condition:
        checkpoint_path = f"{MODEL_DIR}/apricotm_weights.pth"
    else:
        checkpoint_path = f"{ANALYSIS_DIR}/positional_enc/apricotm_pe_False.pth"

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    results = {}

    # Standard cohorts + prospective
    # for cohort in cohorts + [prospective_cohort]:
    for cohort in [prospective_cohort]:

        cohort_key = cohort[1]
        prospective = (cohort == prospective_cohort)
        
        X, static, y, icustay_ids = load_cohort_data(cohort[0], prospective=prospective)

        X_arr = X.copy()  # Keep original for saving
        X = torch.FloatTensor(X)
        static = torch.FloatTensor(static)
        y = torch.FloatTensor(y)

        y_true_class = np.zeros((len(X), 12))
        y_pred_prob = np.zeros((len(X), 12))

        start_time = time.time()

        for patient in range(0, len(X), batch_size):
            inputs = []
            for sample in X[patient : patient + batch_size]:
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1]
                else:
                    padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]), dtype=sample.dtype)
                    adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs).to(DEVICE)
            static_input = static[patient : patient + batch_size].to(DEVICE)

            pred_y = model(inputs, static_input)

            y_true_class[patient : patient + batch_size, :] = y[patient : patient + batch_size].cpu().numpy()
            y_pred_prob[patient : patient + batch_size, :] = pred_y.cpu().detach().numpy()

        inference_time = time.time() - start_time
        print(f"Total inference time on {cohort_key} ({'PE' if pe_condition else 'no-PE'}): {inference_time:.2f} sec")

        # Calculate metrics
        aucs_roc = []
        for i in range(y_pred_prob.shape[1]):
            if len(np.unique(y_true_class[:, i])) > 1:
                ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
                aucs_roc.append(ind_auc)

        aucs_pr = []
        for i in range(y_pred_prob.shape[1]):
            if len(np.unique(y_true_class[:, i])) > 1:
                precision, recall, _ = precision_recall_curve(y_true_class[:, i], y_pred_prob[:, i])
                val_pr_auc = auc(recall, precision)
                aucs_pr.append(val_pr_auc)

        overall_roc = np.mean(np.array(aucs_roc)[[0,3,5,7,9,11]])
        overall_pr = np.mean(np.array(aucs_pr)[[0,3,5,7,9,11]])

        print("-" * 40)
        print(f"Validation {cohort_key} - PE: {pe_condition}")
        print(f"AUROC: {overall_roc:.4f}, AUPRC: {overall_pr:.4f}")

        results[cohort_key] = {
            "AUROC": overall_roc,
            "AUPRC": overall_pr,
        }

        # Save predictions
        cols = [
            "icustay_id", "discharge", "stable", "unstable", "dead",
            "unstable-stable", "stable-unstable", "mv-no mv", "no mv-mv",
            "vp-no vp", "no vp- vp", "crrt-no crrt", "no crrt-crrt",
        ]

        pred_labels = np.concatenate([icustay_ids, y_pred_prob], axis=1)
        pred_labels = pd.DataFrame(pred_labels, columns=cols)

        true_labels = np.concatenate([icustay_ids, y_true_class], axis=1)
        true_labels = pd.DataFrame(true_labels, columns=cols)

        pe_suffix = "pe_true" if pe_condition else "pe_false"
        pred_labels.to_csv(f"{MODEL_DIR}/results_pe_ablation/{cohort_key}_pred_labels_{pe_suffix}.csv", index=None)
        true_labels.to_csv(f"{MODEL_DIR}/results_pe_ablation/{cohort_key}_true_labels_{pe_suffix}.csv", index=None)

    return results

#%% Run evaluation for PE True and PE False
results_pe_true = evaluate_model(pe_condition=True)
results_pe_false = evaluate_model(pe_condition=False)

#%% Save overall results
final_results = {
    "PE_True": results_pe_true,
    "PE_False": results_pe_false
}

results_df = pd.DataFrame({
    (outerKey, innerKey): values
    for outerKey, innerDict in final_results.items()
    for innerKey, values in innerDict.items()
}).T

results_df.to_csv(f"{MODEL_DIR}/results_pe_ablation/pe_ablation_results_summary.csv")
print(results_df)

#%%

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from variables import MODEL_DIR

# Define constants
RESULTS_DIR = f"{MODEL_DIR}/results_pe_ablation"
COHORT_KEYS = ["int", "ext", "temp", "prosp"]
PE_CONDITIONS = ["pe_true", "pe_false"]
TASK_COLUMNS = [
    "discharge", "stable", "unstable", "dead",
    "unstable-stable", "stable-unstable", "mv-no mv", "no mv-mv",
    "vp-no vp", "no vp- vp", "crrt-no crrt", "no crrt-crrt",
]
SELECTED_INDICES = [0, 3, 5, 7, 9, 11]  # task indices for mean AUROC/AUPRC

def calculate_metrics(true_df, pred_df):
    aurocs = []
    auprcs = []

    for i in range(len(TASK_COLUMNS)):
        y_true = true_df[TASK_COLUMNS[i]].values
        y_pred = pred_df[TASK_COLUMNS[i]].values

        # Skip tasks with only one class
        if len(np.unique(y_true)) < 2:
            continue

        auroc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall, precision)

        aurocs.append(auroc)
        auprcs.append(auprc)

    # Mean across all tasks
    mean_auroc = np.mean(aurocs)
    mean_auprc = np.mean(auprcs)
    
    return mean_auroc, mean_auprc

# Aggregate results
results = []

for cohort_key in COHORT_KEYS:
    for pe in PE_CONDITIONS:
        true_path = os.path.join(RESULTS_DIR, f"{cohort_key}_true_labels_{pe}.csv")
        pred_path = os.path.join(RESULTS_DIR, f"{cohort_key}_pred_labels_{pe}.csv")

        true_df = pd.read_csv(true_path)
        pred_df = pd.read_csv(pred_path)

        # Drop icustay_id column
        true_df = true_df[TASK_COLUMNS]
        pred_df = pred_df[TASK_COLUMNS]

        mean_auroc, mean_auprc = calculate_metrics(true_df, pred_df)

        results.append({
            "Cohort": cohort_key,
            "PE Condition": pe,
            "Mean AUROC": mean_auroc,
            "Mean AUPRC": mean_auprc,
        })

# Create summary dataframe
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save to CSV
# results_df.to_csv(os.path.join(RESULTS_DIR, "summary_pe_ablation_metrics.csv"), index=False)


# %%

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from variables import MODEL_DIR

# Define constants
RESULTS_DIR = f"{MODEL_DIR}/results_pe_ablation"
COHORT_KEYS = ["int", "ext", "temp", "prosp"]
PE_CONDITIONS = ["pe_true", "pe_false"]
TASK_COLUMNS = [
    "discharge", "stable", "unstable", "dead",
    "unstable-stable", "stable-unstable", "no mv-mv",
    "no vp- vp", "no crrt-crrt",
]

def get_task_metrics(true_df, pred_df):
    task_auroc = []
    task_auprc = []

    for task in TASK_COLUMNS:
        y_true = true_df[task].values
        y_pred = pred_df[task].values

        if len(np.unique(y_true)) < 2:
            task_auroc.append(np.nan)
            task_auprc.append(np.nan)
            continue

        auroc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall, precision)

        task_auroc.append(auroc)
        task_auprc.append(auprc)

    return task_auroc, task_auprc

# Initialize storage
task_metrics = {pe: {"AUROC": [], "AUPRC": []} for pe in PE_CONDITIONS}

# Loop through cohorts and PE conditions
for pe in PE_CONDITIONS:
    for cohort_key in COHORT_KEYS:
        true_path = os.path.join(RESULTS_DIR, f"{cohort_key}_true_labels_{pe}.csv")
        pred_path = os.path.join(RESULTS_DIR, f"{cohort_key}_pred_labels_{pe}.csv")

        true_df = pd.read_csv(true_path)[TASK_COLUMNS]
        pred_df = pd.read_csv(pred_path)[TASK_COLUMNS]

        aurocs, auprcs = get_task_metrics(true_df, pred_df)
        task_metrics[pe]["AUROC"].append(aurocs)
        task_metrics[pe]["AUPRC"].append(auprcs)

# Average over cohorts (axis=0)
summary_rows = []
for pe in PE_CONDITIONS:
    mean_aurocs = np.nanmean(task_metrics[pe]["AUROC"], axis=0)
    mean_auprcs = np.nanmean(task_metrics[pe]["AUPRC"], axis=0)

    for i, task in enumerate(TASK_COLUMNS):
        summary_rows.append({
            "Task": task,
            "PE Condition": pe,
            "Mean AUROC": mean_aurocs[i],
            "Mean AUPRC": mean_auprcs[i],
        })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_rows)
summary_df["Task"] = pd.Categorical(summary_df["Task"], categories=TASK_COLUMNS, ordered=True)
summary_df = summary_df.sort_values(by=["Task", "PE Condition"])
print(summary_df.to_string(index=False))

# Save to CSV
# summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_taskwise_pe_ablation.csv"), index=False)

# %%

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from variables import MODEL_DIR

# Define constants
RESULTS_DIR = f"{MODEL_DIR}/results_pe_ablation"
COHORT_KEYS = ["int", "ext", "temp", "prosp"]
PE_CONDITIONS = ["pe_true", "pe_false"]
TASK_COLUMNS = [
    "discharge", "stable", "unstable", "dead",
    "unstable-stable", "stable-unstable", "no mv-mv",
    "no vp- vp", "no crrt-crrt",
]

def get_task_aurocs(true_df, pred_df):
    task_auroc = []
    for task in TASK_COLUMNS:
        y_true = true_df[task].values
        y_pred = pred_df[task].values
        if len(np.unique(y_true)) < 2:
            task_auroc.append(np.nan)
        else:
            task_auroc.append(roc_auc_score(y_true, y_pred))
    return task_auroc

# Initialize results dictionary
results = {}

for cohort_key in COHORT_KEYS:
    results[cohort_key] = {}
    for pe in PE_CONDITIONS:
        true_path = os.path.join(RESULTS_DIR, f"{cohort_key}_true_labels_{pe}.csv")
        pred_path = os.path.join(RESULTS_DIR, f"{cohort_key}_pred_labels_{pe}.csv")

        true_df = pd.read_csv(true_path)[TASK_COLUMNS]
        pred_df = pd.read_csv(pred_path)[TASK_COLUMNS]

        aurocs = get_task_aurocs(true_df, pred_df)
        results[cohort_key][pe] = aurocs

# Format into long-form DataFrame
rows = []
for cohort in COHORT_KEYS:
    for i, task in enumerate(TASK_COLUMNS):
        row = {
            "Cohort": cohort,
            "Task": task,
            "AUROC_pe_true": results[cohort]["pe_true"][i],
            "AUROC_pe_false": results[cohort]["pe_false"][i],
        }
        rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df = summary_df[["Cohort", "Task", "AUROC_pe_true", "AUROC_pe_false"]]
summary_df["Task"] = pd.Categorical(summary_df["Task"], categories=TASK_COLUMNS, ordered=True)
summary_df["Cohort"] = pd.Categorical(summary_df["Cohort"], categories=COHORT_KEYS, ordered=True)
summary_df = summary_df.sort_values(by=["Cohort", "Task"])
summary_df[["AUROC_pe_true", "AUROC_pe_false"]] = summary_df[["AUROC_pe_true", "AUROC_pe_false"]].round(2)
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv(os.path.join(RESULTS_DIR, "auroc_by_task_and_cohort.csv"), index=False)

print(sum(summary_df["AUROC_pe_true"] > summary_df["AUROC_pe_false"])/len(summary_df))
print(sum(summary_df["AUROC_pe_false"] > summary_df["AUROC_pe_true"])/len(summary_df))

# %%

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from variables import MODEL_DIR

# Define constants
RESULTS_DIR = f"{MODEL_DIR}/results_pe_ablation"
COHORT_KEYS = ["int", "ext", "temp", "prosp"]
PE_CONDITIONS = ["pe_true", "pe_false"]
TASK_COLUMNS = [
    "discharge", "stable", "unstable", "dead",
    "unstable-stable", "stable-unstable", "no mv-mv",
    "no vp- vp", "no crrt-crrt",
]

def get_task_aurocs(true_df, pred_df):
    task_auroc = []
    for task in TASK_COLUMNS:
        y_true = true_df[task].values
        y_pred = pred_df[task].values
        if len(np.unique(y_true)) < 2:
            task_auroc.append(np.nan)
        else:
            task_auroc.append(roc_auc_score(y_true, y_pred))
    return task_auroc

# Initialize storage
task_metrics = {pe: [] for pe in PE_CONDITIONS}

# Loop through PE conditions and cohorts
for pe in PE_CONDITIONS:
    for cohort_key in COHORT_KEYS:
        true_path = os.path.join(RESULTS_DIR, f"{cohort_key}_true_labels_{pe}.csv")
        pred_path = os.path.join(RESULTS_DIR, f"{cohort_key}_pred_labels_{pe}.csv")

        true_df = pd.read_csv(true_path)[TASK_COLUMNS]
        pred_df = pd.read_csv(pred_path)[TASK_COLUMNS]

        aurocs = get_task_aurocs(true_df, pred_df)
        task_metrics[pe].append(aurocs)

# Compute mean AUROC across cohorts for each PE condition
mean_aurocs_by_pe = {pe: np.nanmean(task_metrics[pe], axis=0) for pe in PE_CONDITIONS}

# Format into single DataFrame with PE conditions as columns
summary_df = pd.DataFrame({
    "Task": TASK_COLUMNS,
    "AUROC_pe_true": mean_aurocs_by_pe["pe_true"],
    "AUROC_pe_false": mean_aurocs_by_pe["pe_false"],
})

print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv(os.path.join(RESULTS_DIR, "taskwise_mean_auroc_pe_comparison.csv"), index=False)
summary_df.to_csv("taskwise_mean_auroc_pe_comparison.csv", index=False)

# %%


import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from variables import MODEL_DIR

# Define constants
RESULTS_DIR = f"{MODEL_DIR}/results_pe_ablation"
COHORT_KEYS = ["int", "ext", "temp", "prosp"]
PE_CONDITIONS = ["pe_true", "pe_false"]
TASK_COLUMNS = [
    "discharge", "stable", "unstable", "dead",
    "unstable-stable", "stable-unstable", "no mv-mv",
    "no vp- vp", "no crrt-crrt",
]

def bootstrap_auroc(y_true, y_pred, n_iter=100):
    """Compute bootstrap median and 95% CI of AUROC."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        try:
            score = roc_auc_score(y_true[idx], y_pred[idx])
            scores.append(score)
        except:
            continue
    if len(scores) == 0:
        return np.nan
    lower, upper = np.percentile(scores, [2.5, 97.5])
    median = np.median(scores)
    return f"{median:.2f} ({lower:.2f}–{upper:.2f})"

# Initialize results dictionary
results = []

for cohort_key in COHORT_KEYS:
    for pe in PE_CONDITIONS:
        true_path = os.path.join(RESULTS_DIR, f"{cohort_key}_true_labels_{pe}.csv")
        pred_path = os.path.join(RESULTS_DIR, f"{cohort_key}_pred_labels_{pe}.csv")

        true_df = pd.read_csv(true_path)[TASK_COLUMNS]
        pred_df = pd.read_csv(pred_path)[TASK_COLUMNS]

        for task in TASK_COLUMNS:
            y_true = true_df[task].values
            y_pred = pred_df[task].values
            auroc_ci = bootstrap_auroc(y_true, y_pred)
            results.append({
                "Cohort": cohort_key,
                "Task": task,
                f"AUROC_{pe}": auroc_ci
            })

# Convert to DataFrame
df_true = pd.DataFrame([r for r in results if "AUROC_pe_true" in r])
df_false = pd.DataFrame([r for r in results if "AUROC_pe_false" in r])

# Merge PE condition columns
summary_df = pd.merge(df_true, df_false, on=["Cohort", "Task"], how="outer")
summary_df = summary_df[["Cohort", "Task", "AUROC_pe_true", "AUROC_pe_false"]]
summary_df["Task"] = pd.Categorical(summary_df["Task"], categories=TASK_COLUMNS, ordered=True)
summary_df["Cohort"] = pd.Categorical(summary_df["Cohort"], categories=COHORT_KEYS, ordered=True)
summary_df = summary_df.sort_values(by=["Cohort", "Task"])

# Display and save
print(summary_df.to_string(index=False))
summary_df.to_csv("auroc_ci_by_task_and_cohort.csv", index=False)
