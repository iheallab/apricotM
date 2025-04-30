#%%
# Import libraries
import pandas as pd
import numpy as np
import h5py
import os
import time
import torch
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

from variables import time_window, PROSP_DATA_DIR, MODEL_DIR, ANALYSIS_DIR
from apricott import ApricotT

#%%
# Create directory to save results
if not os.path.exists(f"{MODEL_DIR}/apricott/seq_len_results"):
    os.makedirs(f"{MODEL_DIR}/apricott/seq_len_results")

#%%
# Load prospective dataset
with h5py.File(f"{ANALYSIS_DIR}/dataset_prospective.h5", "r") as f:
    data = f["prospective"]
    X = data["X"][:]
    static = data["static"][:]
    y_trans = data["y_trans"][:]
    y_main = data["y_main"][:]

icustay_ids = X[:, 0, 3].reshape(-1, 1)

#%%
# Load model parameters
with open(f"{MODEL_DIR}/apricott/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fixed_params = {
    "d_model": 92,
    "d_hidden": 28,
    "h": 4,
    "q": 23,
    "v": 23,
    "N": 2,
    "learning_rate": 3e-4,
    "batch_size": 128,
    "dropout": 0.2,
    "max_code": 49,
}

d_input = 512
d_static = 22

# Sequence lengths to evaluate
seq_lengths = [64, 128, 256, 512]

#%%
# Convert prospective data to tensors
X_arr = X.copy()  # Save original for later
X = torch.FloatTensor(X)
static = torch.FloatTensor(static)

y = np.concatenate([y_main, y_trans], axis=1)
y = torch.FloatTensor(y)

#%%
# Loop over sequence lengths
for seq_len in seq_lengths:
    print(f"Evaluating sequence length: {seq_len}")

    # Load model architecture
    model = ApricotT(
        d_model=fixed_params["d_model"],
        d_hidden=fixed_params["d_hidden"],
        d_input=d_input,
        d_static=d_static,
        max_code=fixed_params["max_code"],
        q=fixed_params["q"],
        v=fixed_params["v"],
        h=fixed_params["h"],
        N=fixed_params["N"],
        dropout=fixed_params["dropout"],
        device=DEVICE,
    ).to(DEVICE)

    # Load corresponding model weights
    model.load_state_dict(torch.load(f"{MODEL_DIR}/apricott/seq_len_weights/model_seq_len_{seq_len}.pth", map_location=DEVICE))
    model.eval()

    # Initialize arrays
    y_true_class = np.zeros((len(X), 12))
    y_pred_prob = np.zeros((len(X), 12))

    BATCH_SIZE = fixed_params["batch_size"]

    start_time = time.time()

    # Inference loop
    for patient in range(0, len(X), BATCH_SIZE):
        inputs = []
        for sample in X[patient : patient + BATCH_SIZE]:
            last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
            if last_non_zero_index >= seq_len:
                adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :]
            else:
                padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]), dtype=sample.dtype)
                adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
            inputs.append(adjusted_sample)

        inputs = torch.stack(inputs).to(DEVICE)
        static_input = static[patient : patient + BATCH_SIZE].to(DEVICE)
        labels = y[patient : patient + BATCH_SIZE].to(DEVICE)

        pred_y = model(inputs, static_input)

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = pred_y.to("cpu").detach().numpy()

    inference_time = time.time() - start_time
    inference_time_batch = inference_time / (int(len(X) / BATCH_SIZE))

    print(f"Total inference time: {inference_time:.2f} seconds")
    print(f"Batch inference time: {inference_time_batch:.2f} seconds")

    # Calculate metrics
    print("-" * 40)
    print(f"Prospective Validation - Sequence length {seq_len}")

    tasks = [
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

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        if len(np.unique(y_true_class[:, i])) > 1:
            ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
            aucs.append(f"{ind_auc:.3f}")
        else:
            aucs.append("N/A")

    auroc_results = pd.DataFrame(data={"Task": tasks, "AUROC": aucs})
    print(auroc_results)

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        if len(np.unique(y_true_class[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(y_true_class[:, i], y_pred_prob[:, i])
            val_pr_auc = auc(recall, precision)
            aucs.append(f"{val_pr_auc:.3f}")
        else:
            aucs.append("N/A")

    auprc_results = pd.DataFrame(data={"Task": tasks, "AUPRC": aucs})
    print("-" * 40)
    print("AUPRC results")
    print(auprc_results)

    # Save results
    cols = [
        "icustay_id",
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

    pred_labels = np.concatenate([icustay_ids, y_pred_prob], axis=1)
    pred_labels = pd.DataFrame(pred_labels, columns=cols)

    true_labels = np.concatenate([icustay_ids, y_true_class], axis=1)
    true_labels = pd.DataFrame(true_labels, columns=cols)

    pred_labels.to_csv(f"{MODEL_DIR}/apricott/seq_len_results/prosp_pred_labels_seq_len_{seq_len}.csv", index=None)
    true_labels.to_csv(f"{MODEL_DIR}/apricott/seq_len_results/prosp_true_labels_seq_len_{seq_len}.csv", index=None)
