import pandas as pd
import numpy as np
import h5py
import time
import torch
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable
from variables import time_window, ANALYSIS_DIR, MODEL_DIR
from apricotm import ApricotM
import pickle

# Create directory to save results
if not os.path.exists(f"{MODEL_DIR}/apricotm/seq_len_results"):
    os.makedirs(f"{MODEL_DIR}/apricotm/seq_len_results")

# Load best parameters
with open(f"{MODEL_DIR}/apricotm/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Load data
cohorts = [["validation", "int"], ["external_test", "ext"], ["temporal_test", "temp"]]

# Sequence lengths to evaluate
seq_lengths = [64, 128, 256, 512]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fixed_params = {
    "d_model": 92,
    "d_hidden": 28,
    "n_layer": 2,
    "learning_rate": 3e-4,
    "batch_size": 128,
    "dropout": 0.2,
    "max_code": 49,
}

d_input = 512
d_static = 22

for seq_len in seq_lengths:
    print(f"Evaluating sequence length: {seq_len}")

    # Load model architecture
    model = ApricotM(
        d_model=fixed_params["d_model"],
        d_hidden=fixed_params["d_hidden"],
        d_input=d_input,
        max_code=fixed_params["max_code"],
        d_static=d_static,
        n_layer=fixed_params["n_layer"],
        dropout=fixed_params["dropout"],
        device=DEVICE,
    ).to(DEVICE)

    # Load model weights for the current sequence length
    model.load_state_dict(torch.load(f"{MODEL_DIR}/apricotm/seq_len_weights/model_seq_len_{seq_len}.pth", map_location=DEVICE))

    model.eval()

    for cohort in cohorts:
        with h5py.File(f"{ANALYSIS_DIR}/final_data/dataset.h5", "r") as f:
            data = f[cohort[0]]
            X_int = data["X"][:]
            static_int = data["static"][:]
            y_trans_int = data["y_trans"][:]
            y_main_int = data["y_main"][:]

        static_int = static_int[:, 1:]

        # Convert data to tensors
        BATCH_SIZE = best_params["batch_size"]

        X_int_arr = X_int.copy()
        X_int = torch.FloatTensor(X_int)
        static_int = torch.FloatTensor(static_int)
        y_int = np.concatenate([y_main_int, y_trans_int], axis=1)
        y_int = torch.FloatTensor(y_int)

        y_true_class = np.zeros((len(X_int), 12))
        y_pred_prob = np.zeros((len(X_int), 12))

        start_time = time.time()

        for patient in range(0, len(X_int), BATCH_SIZE):
            inputs = []
            for sample in X_int[patient : patient + fixed_params["batch_size"]]:
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :]
                else:
                    padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]), dtype=sample.dtype)
                    adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs).to(DEVICE)
            static_input = static_int[patient : patient + fixed_params["batch_size"]].to(DEVICE)
            labels = y_int[patient : patient + fixed_params["batch_size"]].to(DEVICE)

            pred_y = model(inputs, static_input)

            y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
            y_pred_prob[patient : patient + BATCH_SIZE, :] = pred_y.to("cpu").detach().numpy()

        inference_time = time.time() - start_time
        inference_time_batch = inference_time / (int(len(X_int) / BATCH_SIZE))

        print(f"Total inference time: {inference_time} seconds")
        print(f"Batch inference time: {inference_time_batch} seconds")

        print("-" * 40)
        print(f"Validation {cohort[1]}")

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
            aucs.append(ind_auc)

        print(f"val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(y_true_class[:, i], y_pred_prob[:, i])
            val_pr_auc = auc(recall, precision)
            aucs.append(val_pr_auc)

        print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

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

        pred_labels = np.concatenate([X_int_arr[:, 0, 3].reshape(-1, 1), y_pred_prob], axis=1)
        pred_labels = pd.DataFrame(pred_labels, columns=cols)

        true_labels = np.concatenate([X_int_arr[:, 0, 3].reshape(-1, 1), y_true_class], axis=1)
        true_labels = pd.DataFrame(true_labels, columns=cols)

        pred_labels.to_csv(f"{MODEL_DIR}/apricotm/seq_len_results/{cohort[1]}_pred_labels_seq_len_{seq_len}.csv", index=None)
        true_labels.to_csv(f"{MODEL_DIR}/apricotm/seq_len_results/{cohort[1]}_true_labels_seq_len_{seq_len}.csv", index=None)