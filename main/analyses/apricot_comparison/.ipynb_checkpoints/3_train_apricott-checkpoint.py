#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import optuna
import time
import json

#%%

from variables import time_window, MODEL_DIR, ANALYSIS_DIR

#%%

# Create directory for saving model

if not os.path.exists(f"{MODEL_DIR}/apricott"):
    os.makedirs(f"{MODEL_DIR}/apricott")

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
from models.apricott import ApricotT

EPOCH = 100
BATCH_SIZE = 256
LR = 5e-4
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


from torch import optim
from torch.autograd import Variable
import tqdm
import optuna

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

# Define objective function
def objective(trial):
    # Define hyperparameters to be optimized
    d_model = trial.suggest_int("d_model", 64, 256, step=2)
    h = trial.suggest_int("h", 2, 16, step=2)
    q = int(d_model / h)
    v = int(d_model / h)
    d_hidden = trial.suggest_int("d_hidden", 8, 64, step=2)
    N = trial.suggest_int("N", 1, 3)
    LR = trial.suggest_loguniform("learning_rate", 1e-4, 5e-4)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.2, 0.3, 0.4])
    seq_len = trial.suggest_categorical("seq_len", [64, 128, 256, 512])
    max_code = 49
    
    nb_batches = int(len(X_train) / BATCH_SIZE) + 1
    nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

    # Initialize model, loss, and optimizer
    net = ApricotT(
        d_model=d_model,
        d_hidden=d_hidden,
        d_input=d_input,
        max_code=max_code,
        d_static=d_static,
        q=q,
        v=v,
        h=h,
        N=N,
        dropout=dropout,
        device=DEVICE,
    ).to(DEVICE)

    optimizer_name = "Adam"

    if optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    elif optimizer_name == "RMS":
        optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=2, verbose=True)

    best_score = 0
    best_auroc = 0
    best_auprc = 0

    for epoch in range(EPOCH):
        train_loss = 0
        # total_accuracy = 0
        count = 0
        y_true = np.zeros((len(X_train)))
        y_true_class = np.zeros((len(X_train), d_output))
        y_pred = np.zeros((len(X_train)))
        y_pred_prob = np.zeros((len(X_train), d_output))
        for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):
            # for patient in range(0, len(X_train), BATCH_SIZE):
            # Adjust sequence size to 256 for each sample
            inputs = []
            for sample in X_train[patient : patient + BATCH_SIZE]:

                # Find the last non-zero index in the first dimension (512)
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

                # Take the last 256 values if the last non-zero index is greater than or equal to 256
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[
                        last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                    ]
                # Pad the sequence with zeros if the last non-zero index is less than 256
                else:
                    padding = torch.zeros(
                        (seq_len - last_non_zero_index - 1, sample.shape[1]),
                        dtype=sample.dtype,
                    )
                    adjusted_sample = torch.cat(
                        (sample[: last_non_zero_index + 1], padding), dim=0
                    )

                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs)
            inputs = Variable(inputs).to(DEVICE)

            # inputs = Variable(X_train[patient:patient+BATCH_SIZE]).to(DEVICE)
            static_input = Variable(static_train[patient : patient + BATCH_SIZE]).to(
                DEVICE
            )
            labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
            optimizer.zero_grad()
            pred_y = net(inputs, static_input)

            loss = 0

            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

                loss = loss + loss_class

            loss.backward()
            train_loss += loss.data
            optimizer.step()

            y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
            y_pred_prob[patient : patient + BATCH_SIZE, :] = (
                pred_y.to("cpu").detach().numpy()
            )

            count += 1

        print("-" * 40)
        print(f"Epoch {epoch+1}")

        print("Train performance")
        print("loss: {}".format(train_loss / nb_batches))

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
            aucs.append(ind_auc)

        print(f"train_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(
                y_true_class[:, i], y_pred_prob[:, i]
            )
            val_pr_auc = auc(recall, precision)
            aucs.append(val_pr_auc)

        print(f"train_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        y_true = np.zeros((len(X_val)))
        y_true_class = np.zeros((len(X_val), d_output))
        y_pred = np.zeros((len(X_val)))
        y_pred_prob = np.zeros((len(X_val), d_output))
        val_loss = 0
        for patient in range(0, len(X_val), BATCH_SIZE):

            inputs = []
            for sample in X_val[patient : patient + BATCH_SIZE]:

                # Find the last non-zero index in the first dimension (512)
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

                # Take the last 256 values if the last non-zero index is greater than or equal to 256
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[
                        last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                    ]
                # Pad the sequence with zeros if the last non-zero index is less than 256
                else:
                    padding = torch.zeros(
                        (seq_len - last_non_zero_index - 1, sample.shape[1]),
                        dtype=sample.dtype,
                    )
                    adjusted_sample = torch.cat(
                        (sample[: last_non_zero_index + 1], padding), dim=0
                    )

                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs)
            inputs = Variable(inputs).to(DEVICE)

            # inputs = Variable(X_val[patient:patient+BATCH_SIZE]).to(DEVICE)
            static_input = Variable(static_val[patient : patient + BATCH_SIZE]).to(
                DEVICE
            )
            labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)

            pred_y = net(inputs, static_input)

            loss = 0

            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

                loss = loss + loss_class

            val_loss += loss.data

            y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
            y_pred_prob[patient : patient + BATCH_SIZE, :] = (
                pred_y.to("cpu").detach().numpy()
            )

        print("-" * 40)
        print("Val performance")

        print("val_loss: {}".format(val_loss / nb_batches_val))

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
            aucs.append(ind_auc)

        print(f"val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = np.array(aucs)

        roc_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

        print(f"Overall AUROC: {roc_auc_mean}")

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(
                y_true_class[:, i], y_pred_prob[:, i]
            )
            val_pr_auc = auc(recall, precision)
            aucs.append(val_pr_auc)

        print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = np.array(aucs)

        pr_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

        print(f"Overall AUPRC: {pr_auc_mean}")

        roc_pr_auc = roc_auc_mean + pr_auc_mean

        if roc_pr_auc > best_score:
            best_score = roc_pr_auc
            best_auroc = roc_auc_mean
            best_auprc = pr_auc_mean

        early_stopping(roc_pr_auc, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return best_score


# Create study
study = optuna.create_study(direction="maximize")

# Start the timer
start_time = time.time()

# Optimize hyperparameters
study.optimize(objective, n_trials=10)

# End the timer
end_time = time.time()
elapsed_time = end_time - start_time

# Log the study details
study_details = {
    "best_params": study.best_params,
    "best_value": study.best_value,
    "best_trial_number": study.best_trial.number,
    "total_trials": len(study.trials),
    "optimization_duration_seconds": elapsed_time,
    "all_trials": []
}

# Save details of each trial
for trial in study.trials:
    trial_details = {
        "trial_number": trial.number,
        "params": trial.params,
        "value": trial.value,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        "state": trial.state.name
    }
    study_details["all_trials"].append(trial_details)

# Save study details to a JSON file
with open(f"{MODEL_DIR}/apricott/study_details.json", "w") as f:
    json.dump(study_details, f, indent=4)

print(f"Study completed in {elapsed_time:.2f} seconds.")
print(f"Best parameters: {study.best_params}")


# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

import pickle

with open("%s/apricott/best_params.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(best_params, f, protocol=2)


# Define hyperparameters to be optimized
d_model = best_params["d_model"]
d_hidden = best_params["d_hidden"]
h = best_params["h"]
q = int(d_model / h)
v = int(d_model / h)
N = best_params["N"]
LR = best_params["learning_rate"]
BATCH_SIZE = best_params["batch_size"]
dropout = best_params["dropout"]
seq_len = best_params["seq_len"]
max_code = 49

nb_batches = int(len(X_train) / BATCH_SIZE) + 1
nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

# Initialize model, loss, and optimizer
net = ApricotT(
    d_model=d_model,
    d_hidden=d_hidden,
    d_input=d_input,
    max_code=max_code,
    d_static=d_static,
    N=N,
    h=h,
    q=q,
    v=v,
    dropout=dropout,
    device=DEVICE,
).to(DEVICE)

optimizer_name = "Adam"

if optimizer_name == "Adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "RMS":
    optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path=""):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, roc_pr_auc, model):

        if self.best_score is None:
            self.best_score = roc_pr_auc
            self.save_checkpoint(roc_pr_auc, model)
        elif roc_pr_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(roc_pr_auc, model)
            self.best_score = roc_pr_auc
            self.counter = 0

    def save_checkpoint(self, roc_pr_auc, model):
        if self.verbose:
            print(
                f"Validation AUROC-AUPRC increased ({self.best_score:.6f} --> {roc_pr_auc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)


early_stopping = EarlyStopping(
    patience=2, verbose=True, path=f"{MODEL_DIR}/apricott/model_weights.pth"
)

best_score = 0
best_auroc = 0
best_auprc = 0

for epoch in range(EPOCH):
    train_loss = 0
    # total_accuracy = 0
    count = 0
    y_true = np.zeros((len(X_train)))
    y_true_class = np.zeros((len(X_train), d_output))
    y_pred = np.zeros((len(X_train)))
    y_pred_prob = np.zeros((len(X_train), d_output))
    for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):
        # for patient in range(0, len(X_train), BATCH_SIZE):
        # Adjust sequence size to 256 for each sample
        inputs = []
        for sample in X_train[patient : patient + BATCH_SIZE]:

            # Find the last non-zero index in the first dimension (512)
            last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

            # Take the last 256 values if the last non-zero index is greater than or equal to 256
            if last_non_zero_index >= seq_len:
                adjusted_sample = sample[
                    last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                ]
            # Pad the sequence with zeros if the last non-zero index is less than 256
            else:
                padding = torch.zeros(
                    (seq_len - last_non_zero_index - 1, sample.shape[1]),
                    dtype=sample.dtype,
                )
                adjusted_sample = torch.cat(
                    (sample[: last_non_zero_index + 1], padding), dim=0
                )

            inputs.append(adjusted_sample)

        inputs = torch.stack(inputs)
        inputs = Variable(inputs).to(DEVICE)

        # inputs = Variable(X_train[patient:patient+BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        optimizer.zero_grad()
        pred_y = net(inputs, static_input)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        loss.backward()
        train_loss += loss.data
        optimizer.step()

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

        count += 1

    print("-" * 40)
    print(f"Epoch {epoch+1}")

    print("Train performance")
    print("loss: {}".format(train_loss / nb_batches))

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"train_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"train_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    y_true = np.zeros((len(X_val)))
    y_true_class = np.zeros((len(X_val), d_output))
    y_pred = np.zeros((len(X_val)))
    y_pred_prob = np.zeros((len(X_val), d_output))
    val_loss = 0
    for patient in range(0, len(X_val), BATCH_SIZE):

        inputs = []
        for sample in X_val[patient : patient + BATCH_SIZE]:

            # Find the last non-zero index in the first dimension (512)
            last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

            # Take the last 256 values if the last non-zero index is greater than or equal to 256
            if last_non_zero_index >= seq_len:
                adjusted_sample = sample[
                    last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                ]
            # Pad the sequence with zeros if the last non-zero index is less than 256
            else:
                padding = torch.zeros(
                    (seq_len - last_non_zero_index - 1, sample.shape[1]),
                    dtype=sample.dtype,
                )
                adjusted_sample = torch.cat(
                    (sample[: last_non_zero_index + 1], padding), dim=0
                )

            inputs.append(adjusted_sample)

        inputs = torch.stack(inputs)
        inputs = Variable(inputs).to(DEVICE)

        # inputs = Variable(X_val[patient:patient+BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_val[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)

        pred_y = net(inputs, static_input)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        val_loss += loss.data

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

    print("-" * 40)
    print("Val performance")

    print("val_loss: {}".format(val_loss / nb_batches_val))

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = np.array(aucs)

    roc_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

    print(f"Overall AUROC: {roc_auc_mean}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = np.array(aucs)

    pr_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

    print(f"Overall AUPRC: {pr_auc_mean}")

    roc_pr_auc = roc_auc_mean + pr_auc_mean

    if roc_pr_auc > best_score:
        best_score = roc_pr_auc
        best_auroc = roc_auc_mean
        best_auprc = pr_auc_mean

    early_stopping(roc_pr_auc, net)

    if early_stopping.early_stop:
        print("Early stopping")
        break


model_architecture = {
    "d_model": d_model,
    "d_hidden": d_hidden,
    "d_input": d_input,
    "max_code": max_code,
    "d_static": d_static,
    "q": q,
    "v": v,
    "h": h,
    "N": N,
    "dropout": dropout,
}

torch.save(model_architecture, f"{MODEL_DIR}/apricott/model_architecture.pth")

# %%
