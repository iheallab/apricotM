#%%

# Import libraries
import pandas as pd
import numpy as np
import h5py
import os

from variables import time_window, DATA_DIR, MODEL_DIR

#%%
# Load dataset

with h5py.File("%s/dataset.h5" % DATA_DIR, "r") as f:
    data = f["prospective"]
    X = data["X"][:]
    static = data["static"][:]
    y_trans = data["y_trans"][:]
    y_main = data["y_main"][:]


#%%
# Load model

from apricott import ApricotT
import torch

# Load model architecture

model_architecture = torch.load(f"{MODEL_DIR}/apricott_architecture.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ApricotT(
    d_model=model_architecture["d_model"],
    d_hidden=model_architecture["d_hidden"],
    d_input=model_architecture["d_input"],
    d_static=model_architecture["d_static"],
    max_code=model_architecture["max_code"],
    N=model_architecture["N"],
    h=model_architecture["h"],
    q=model_architecture["q"],
    v=model_architecture["v"],
    device=DEVICE,
    dropout=model_architecture["dropout"],
).to(DEVICE)


# Load model weights

model.load_state_dict(
    torch.load(f"{MODEL_DIR}/apricott_weights.pth", map_location=DEVICE)
)

#%%

# Convert data to tensors

X = torch.FloatTensor(X)
static = torch.FloatTensor(static)

y = np.concatenate([y_main, y_trans], axis=1)
y = torch.FloatTensor(y)

#%%

# Prospective validation
import pickle

with open(f"{MODEL_DIR}/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

BATCH_SIZE = best_params["batch_size"]

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

y_true_class = np.zeros((len(X), 12))
y_pred_prob = np.zeros((len(X), 12))

for patient in range(0, len(X), BATCH_SIZE):
    inputs = Variable(X[patient : patient + BATCH_SIZE]).to(DEVICE)
    static_input = Variable(static[patient : patient + BATCH_SIZE]).to(DEVICE)
    labels = Variable(y[patient : patient + BATCH_SIZE]).to(DEVICE)

    pred_y = model(inputs, static_input)

    y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
    y_pred_prob[patient : patient + BATCH_SIZE, :] = pred_y.to("cpu").detach().numpy()


#%%

# Calculate metrics

print("-" * 40)
print("Prospective Validation")

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
        ind_auc = "N/A"
        aucs.append(ind_auc)

auroc_results = pd.DataFrame(data={"Task": tasks, "AUROC": aucs})

print("-" * 40)
print("AUROC results")

print(auroc_results)


aucs = []
for i in range(y_pred_prob.shape[1]):
    if len(np.unique(y_true_class[:, i])) > 1:
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(f"{val_pr_auc:.3f}")
    else:
        val_pr_auc = "N/A"
        aucs.append(val_pr_auc)

auprc_results = pd.DataFrame(data={"Task": tasks, "AUPRC": aucs})

print("-" * 40)
print("AUPRC results")

print(auprc_results)

#%%

# Save results

icustay_ids = X[:, 0, 3].reshape(-1, 1)

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

if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

true_labels.to_csv(f"{MODEL_DIR}/results/prosp_true_labels.csv", index=None)
pred_labels.to_csv(f"{MODEL_DIR}/results/prosp_pred_labels.csv", index=None)
