#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import time

from variables import time_window, MODEL_DIR

if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

#%%

# Load data

groups = [
    ["validation", "int"],
    ["external_test", "ext"],
    ["temporal_test", "temp"],
    ["prospective", "prosp"],
]

# Load model

from model.model import Transformer
import torch

# Load model architecture

model_architecture = torch.load("%s/model_architecture.pth" % MODEL_DIR)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Transformer(
    q=model_architecture["q"],
    v=model_architecture["v"],
    h=model_architecture["h"],
    N=model_architecture["N"],
    device=DEVICE,
    dropout=model_architecture["dropout"],
).to(DEVICE)

# Load best parameters

import pickle

with open("%s/best_params.pkl" % MODEL_DIR, "rb") as f:
    best_params = pickle.load(f)

# Load model weights

model.load_state_dict(torch.load("%s/model_weights.pth" % MODEL_DIR))

for group in groups:

    with h5py.File("%s/dataset_transformer.h5" % MODEL_DIR, "r") as f:
        data = f[group[0]]
        X = data["X"][:]
        y_trans = data["y_trans"][:]
        y_main = data["y_main"][:]

    y = np.concatenate([y_main, y_trans], axis=1)

    # Convert data to tensors

    BATCH_SIZE = best_params["batch_size"]

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # Run validation

    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    from torch.autograd import Variable

    y_true_class = np.zeros((len(X), 12))
    y_pred_prob = np.zeros((len(X), 12))

    start_time = time.time()

    for patient in range(0, len(X), BATCH_SIZE):

        inputs = Variable(X[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y[patient : patient + BATCH_SIZE]).to(DEVICE)

        pred_y = model(inputs)

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

    inference_time = time.time() - start_time
    inference_time_batch = inference_time / (int(len(X) / BATCH_SIZE))

    print(f"Total inference time: {inference_time} seconds")
    print(f"Batch inference time: {inference_time_batch} seconds")

    print("-" * 40)
    print(f"Validation {group[1]}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    cols = [
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

    pred_labels = pd.DataFrame(y_pred_prob, columns=cols)
    true_labels = pd.DataFrame(y_true_class, columns=cols)

    true_labels.to_csv(f"{MODEL_DIR}/results/{group[1]}_true_labels.csv", index=None)
    pred_labels.to_csv(f"{MODEL_DIR}/results/{group[1]}_pred_labels.csv", index=None)

# %%
