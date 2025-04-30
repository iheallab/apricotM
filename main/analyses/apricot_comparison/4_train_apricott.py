#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import optuna
import time
import json

from torch import optim
from torch.autograd import Variable
import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

#%%

from variables import time_window, MODEL_DIR, ANALYSIS_DIR

#%%

# Create directory for saving model

if not os.path.exists(f"{MODEL_DIR}/apricott/seq_len_weights"):
    os.makedirs(f"{MODEL_DIR}/apricott/seq_len_weights")

#%%

# Load data

with h5py.File("%s/final_data/dataset.h5" % ANALYSIS_DIR, "r") as f:
    data = f["training"]
    X_train = data["X"][:]
    static_train = data["static"][:]
    y_trans_train = data["y_trans"][:]
    y_main_train = data["y_main"][:]

    data = f["validation"]
    X_val = data["X"][:]
    static_val = data["static"][:]
    y_trans_val = data["y_trans"][:]
    y_main_val = data["y_main"][:]

static_train = static_train[:, 1:]
static_val = static_val[:, 1:]


# %%

# Shuffle training data

from sklearn.utils import shuffle

X_train, static_train, y_main_train, y_trans_train = shuffle(
    X_train, static_train, y_main_train, y_trans_train
)


#%%

# Merge targets

y_train = np.concatenate([y_main_train, y_trans_train], axis=1)
y_val = np.concatenate([y_main_val, y_trans_val], axis=1)

#%%

# Define hyper parameters and model parameters

import torch
import math
import torch.nn as nn
from apricott import ApricotT

EPOCH = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"use DEVICE: {DEVICE}")

# Convert data to tensors

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
static_train = torch.FloatTensor(static_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
static_val = torch.FloatTensor(static_val)


class_weights = []

d_input = X_train.shape[1]
d_static = static_train.shape[1]
d_output = 12

for i in range(d_output):

    negative = np.unique(y_train[:, i], return_counts=True)[1][0]
    positive = np.unique(y_train[:, i], return_counts=True)[1][1]

    class_weight = negative / positive

    # class_weight = torch.tensor(negative/positive)
    class_weights.append(class_weight)

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

loss_bin = BCELoss()
loss_bin_mort = BCEWithLogitsLoss()


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        # self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, roc_pr_auc, model):

        if self.best_score is None:
            self.best_score = roc_pr_auc
            # self.save_checkpoint(roc_pr_auc, model)
        elif roc_pr_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # self.save_checkpoint(roc_pr_auc, model)
            self.best_score = roc_pr_auc
            self.counter = 0

    # def save_checkpoint(self, roc_pr_auc, model):
    #     if self.verbose:
    #         print(f'Validation AUROC-AUPRC increased ({self.best_score:.6f} --> {roc_pr_auc:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)


# Fixed hyperparameter values
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

# Sequence lengths to evaluate
seq_lengths = [64, 128, 256, 512]

# Loop through each sequence length
for seq_len in seq_lengths:
    print(f"Training with sequence length: {seq_len}")
    
    nb_batches = int(len(X_train) / fixed_params["batch_size"]) + 1
    nb_batches_val = int(len(X_val) / fixed_params["batch_size"]) + 1

    # Initialize model
    net = ApricotT(
        d_model=fixed_params["d_model"],
        d_hidden=fixed_params["d_hidden"],
        d_input=d_input,
        max_code=fixed_params["max_code"],
        d_static=d_static,
        q=fixed_params["q"],
        v=fixed_params["v"],
        h=fixed_params["h"],
        N=fixed_params["N"],
        dropout=fixed_params["dropout"],
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=fixed_params["learning_rate"], weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=2, verbose=True)

    best_score = 0
    best_auroc = 0
    best_auprc = 0

    for epoch in range(EPOCH):
        train_loss = 0
        y_true_class = np.zeros((len(X_train), d_output))
        y_pred_prob = np.zeros((len(X_train), d_output))

        # Training loop
        for patient in tqdm.trange(0, len(X_train), fixed_params["batch_size"]):
            inputs = []
            for sample in X_train[patient : patient + fixed_params["batch_size"]]:
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :]
                else:
                    padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]), dtype=sample.dtype)
                    adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs).to(DEVICE)
            static_input = static_train[patient : patient + fixed_params["batch_size"]].to(DEVICE)
            labels = y_train[patient : patient + fixed_params["batch_size"]].to(DEVICE)

            optimizer.zero_grad()
            pred_y = net(inputs, static_input)

            loss = 0
            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]
                loss += loss_class

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            y_true_class[patient : patient + fixed_params["batch_size"], :] = labels.cpu().numpy()
            y_pred_prob[patient : patient + fixed_params["batch_size"], :] = pred_y.cpu().detach().numpy()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss / nb_batches:.6f}")

        # Validation loop
        val_loss = 0
        y_true_class_val = np.zeros((len(X_val), d_output))
        y_pred_prob_val = np.zeros((len(X_val), d_output))

        for patient in range(0, len(X_val), fixed_params["batch_size"]):
            inputs = []
            for sample in X_val[patient : patient + fixed_params["batch_size"]]:
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :]
                else:
                    padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]), dtype=sample.dtype)
                    adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs).to(DEVICE)
            static_input = static_val[patient : patient + fixed_params["batch_size"]].to(DEVICE)
            labels = y_val[patient : patient + fixed_params["batch_size"]].to(DEVICE)

            pred_y = net(inputs, static_input)

            loss = 0
            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]
                loss += loss_class

            val_loss += loss.item()
            y_true_class_val[patient : patient + fixed_params["batch_size"], :] = labels.cpu().numpy()
            y_pred_prob_val[patient : patient + fixed_params["batch_size"], :] = pred_y.cpu().detach().numpy()

        print(f"Validation Loss: {val_loss / nb_batches_val:.6f}")

        # Calculate metrics
        val_roc_auc = np.mean([roc_auc_score(y_true_class_val[:, i], y_pred_prob_val[:, i]) for i in range(d_output)])
        val_pr_auc = np.mean([auc(*precision_recall_curve(y_true_class_val[:, i], y_pred_prob_val[:, i])[1::-1]) for i in range(d_output)])
        
        print(f"Validation AUROC: {val_roc_auc:.6f}, AUPRC: {val_pr_auc:.6f}")

        roc_pr_auc = val_roc_auc + val_pr_auc
        if roc_pr_auc > best_score:
            best_score = roc_pr_auc
            best_auroc = val_roc_auc
            best_auprc = val_pr_auc

        early_stopping(roc_pr_auc, net)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Best AUROC: {best_auroc:.6f}, Best AUPRC: {best_auprc:.6f}")

    # Save the model for this sequence length
    model_path = f"{MODEL_DIR}/apricott/seq_len_weights/model_seq_len_{seq_len}.pth"
    torch.save(net.state_dict(), model_path)
    print(f"Model saved at {model_path}")
